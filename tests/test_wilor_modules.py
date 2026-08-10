"""``perception/wilor.py`` + ``perception/moge.py`` 的单测。不需要 GPU、不需要模型。

能这么测是因为 WiLoR 的 ``predict`` 和 MoGe 的 ``infer`` 都是注入进来的，这里用假的顶上。

重点钉三类**错了不会报错**的地方：

1. **两条取深度路径的像素取整方式不同**（``round`` vs ``int()`` 截断），这是从原脚本
   照抄的差异。谁"顺手统一"一下，HO-3D 那份 11.0 cm 的数字就变了，而代码照样跑。
2. **反投影必须用取整后的像素**。深度是在整数像素处取的，用亚像素坐标反投影会得到一个
   和深度不对应的 XY —— 误差只有半个像素量级，肉眼和单测都容易放过去。
3. **patch 取中位不取均值**。手边缘的像素落在深度断层上，均值会被背景拽走几十厘米，
   而中位免疫。这是原脚本对的地方，得钉住。
"""
import sys
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from web2robot.perception import moge as MG  # noqa: E402
from web2robot.perception import wilor as WL  # noqa: E402
from web2robot.perception.to_clip import HAND_LEFT, HAND_RIGHT, N_JOINTS  # noqa: E402


def _det(is_right=1.0, kp2d=None, kp3d=None, cam_t=None):
    """一条假的 WiLoR 检测。默认 21 个关键点全在 (10,10)。"""
    return {
        "is_right": is_right,
        "wilor_preds": {
            "pred_keypoints_2d": (np.full((N_JOINTS, 2), 10.0) if kp2d is None
                                  else np.asarray(kp2d, float)),
            "pred_keypoints_3d": (np.zeros((N_JOINTS, 3)) if kp3d is None
                                  else np.asarray(kp3d, float)),
            "pred_cam_t_full": (np.zeros(3) if cam_t is None
                                else np.asarray(cam_t, float)),
        },
    }


class TestSamplePointmap(unittest.TestCase):

    def test_returns_the_median_of_the_patch(self):
        pm = np.zeros((20, 20, 3))
        pm[7:14, 7:14] = [1.0, 2.0, 3.0]
        np.testing.assert_allclose(MG.sample_pointmap(pm, (10, 10)), [1, 2, 3])

    def test_median_survives_half_the_patch_being_background(self):
        """一半像素落在背景深度上时中位仍取到手 —— 均值会被拽走。"""
        pm = np.zeros((20, 20, 3))
        pm[..., 2] = 5.0                      # 背景 5 m
        pm[7:14, 7:11, 2] = 0.5               # patch 左侧 4/7 列是手，0.5 m
        got = MG.sample_pointmap(pm, (10, 10))[2]
        self.assertAlmostEqual(got, 0.5, places=6)
        patch = pm[7:14, 7:14, 2]
        self.assertGreater(patch.mean(), 1.0, "这组输入下均值确实会被背景拽走")

    def test_non_finite_points_are_dropped_not_propagated(self):
        pm = np.full((20, 20, 3), np.nan)
        pm[10, 10] = [1.0, 1.0, 1.0]
        np.testing.assert_allclose(MG.sample_pointmap(pm, (10, 10)), [1, 1, 1])

    def test_all_nan_patch_gives_nan_not_zero(self):
        """给 0 会是一个合法的相机系坐标，静默错下去；必须是 NaN。"""
        got = MG.sample_pointmap(np.full((20, 20, 3), np.nan), (10, 10))
        self.assertTrue(np.isnan(got).all())

    def test_patch_is_clipped_at_both_borders(self):
        """边缘上的关键点仍取到数（patch 双侧 clip），不越界报错。"""
        pm = np.arange(5 * 5 * 3, dtype=float).reshape(5, 5, 3)
        for uv in [(0, 0), (4, 4), (-3, -3), (7, 7)]:
            self.assertTrue(np.isfinite(MG.sample_pointmap(pm, uv)).all(), uv)

    def test_a_keypoint_far_outside_the_image_gives_nan_not_a_border_pixel(self):
        """中心像素**不**clip，所以完全出画的关键点 patch 是空的 → NaN。

        这是原脚本的行为，而且是对的：手已经出画了，返回边缘像素的深度是编数。
        和 `sample_depth` **故意不同** —— 那条会把中心 clip 进画内，永远返回一个值。
        """
        pm = np.ones((5, 5, 3))
        self.assertTrue(np.isnan(MG.sample_pointmap(pm, (99, 99))).all())
        self.assertAlmostEqual(MG.sample_depth(np.ones((5, 5)), (99, 99)), 1.0)

    def test_rounds_the_pixel(self):
        """``round``：9.6 → 10。和 ``sample_depth`` 的截断行为**故意不同**。"""
        pm = np.zeros((20, 20, 3))
        pm[10, 10] = [1.0, 0.0, 0.0]          # 只有正中一个像素非零
        a = MG.sample_pointmap(pm, (9.6, 9.6), r=0)
        np.testing.assert_allclose(a, [1, 0, 0])

    def test_rejects_a_depth_map_passed_by_mistake(self):
        with self.assertRaises(ValueError):
            MG.sample_pointmap(np.zeros((20, 20)), (10, 10))


class TestSampleDepth(unittest.TestCase):

    def test_returns_the_median_of_the_patch(self):
        d = np.zeros((20, 20))
        d[7:14, 7:14] = 0.8
        self.assertAlmostEqual(MG.sample_depth(d, (10, 10)), 0.8)

    def test_truncates_the_pixel_instead_of_rounding(self):
        """这里是 ``int()`` 截断：9.6 → 9，**不是** 10。原脚本如此，不许统一。"""
        d = np.zeros((20, 20))
        d[9, 9] = 1.0
        d[10, 10] = 2.0
        self.assertAlmostEqual(MG.sample_depth(d, (9.6, 9.6), r=0), 1.0)
        self.assertAlmostEqual(MG.sample_pointmap(
            np.dstack([d, d, d]), (9.6, 9.6), r=0)[0], 2.0)

    def test_the_two_paths_really_do_differ_on_this_input(self):
        """确认上一条不是在测两个恒等函数 —— 差异必须是可观测的。"""
        d = np.zeros((20, 20))
        d[9, 9], d[10, 10] = 1.0, 2.0
        self.assertNotEqual(MG.sample_depth(d, (9.6, 9.6), r=0),
                            MG.sample_pointmap(np.dstack([d, d, d]), (9.6, 9.6), r=0)[0])

    def test_all_nan_patch_gives_nan(self):
        self.assertTrue(np.isnan(MG.sample_depth(np.full((20, 20), np.nan), (10, 10))))

    def test_rejects_a_pointmap_passed_by_mistake(self):
        with self.assertRaises(ValueError):
            MG.sample_depth(np.zeros((20, 20, 3)), (10, 10))


class TestUnproject(unittest.TestCase):

    K = np.array([[600.0, 0, 320.0], [0, 600.0, 240.0], [0, 0, 1.0]])

    def test_principal_point_maps_to_the_optical_axis(self):
        np.testing.assert_allclose(MG.unproject((320, 240), 1.5, self.K), [0, 0, 1.5])

    def test_one_focal_length_off_axis_is_one_depth_off_axis(self):
        got = MG.unproject((320 + 600, 240), 2.0, self.K)
        np.testing.assert_allclose(got, [2.0, 0.0, 2.0])

    def test_nan_depth_gives_a_nan_point_not_the_origin(self):
        got = MG.unproject((320, 240), float("nan"), self.K)
        self.assertTrue(np.isnan(got).all())

    def test_must_use_the_rounded_pixel_that_the_depth_came_from(self):
        """亚像素坐标反投影会得到和深度不对应的 XY —— 差半像素，很容易放过去。"""
        uv = (100.7, 50.2)
        px = MG.pixel_index(uv, (480, 640))
        self.assertEqual(px, (100, 50))
        good = MG.unproject(px, 1.0, self.K)
        bad = MG.unproject(uv, 1.0, self.K)
        self.assertFalse(np.allclose(good, bad))

    def test_bad_intrinsics_raise(self):
        with self.assertRaises(ValueError):
            MG.unproject((0, 0), 1.0, np.eye(4))


class TestSceneAnchor(unittest.TestCase):

    def test_median_of_medians(self):
        maps = [np.full((4, 4), v) for v in (1.0, 2.0, 3.0)]
        self.assertAlmostEqual(MG.scene_depth_anchor(maps), 2.0)

    def test_all_nan_raises_instead_of_returning_nan(self):
        with self.assertRaises(ValueError):
            MG.scene_depth_anchor([np.full((4, 4), np.nan)])

    def test_frame_indices_match_the_original_stride(self):
        """``frames[::N//6 or 1][:6]`` 的逐字行为。"""
        self.assertEqual(MG.anchor_frame_indices(60), [0, 10, 20, 30, 40, 50])
        self.assertEqual(MG.anchor_frame_indices(3), [0, 1, 2])   # N//6==0 → step 1
        self.assertEqual(MG.anchor_frame_indices(1), [0])
        with self.assertRaises(ValueError):
            MG.anchor_frame_indices(0)


class TestHandSlot(unittest.TestCase):

    def test_threshold_is_a_half(self):
        self.assertEqual(WL.hand_slot(_det(is_right=0.51)), HAND_RIGHT)
        self.assertEqual(WL.hand_slot(_det(is_right=0.5)), HAND_LEFT)
        self.assertEqual(WL.hand_slot(_det(is_right=0.0)), HAND_LEFT)


class TestJointExtraction(unittest.TestCase):

    def test_pointmap_path_fills_all_21_joints(self):
        """原脚本只取了 [0,4,8] 三个；下游 clip 契约要 21 个，默认取全。"""
        pm = np.zeros((40, 40, 3))
        pm[..., 2] = 0.7
        kp = np.stack([np.arange(N_JOINTS) + 5, np.full(N_JOINTS, 20)], 1)
        got = WL.joints_from_pointmap(_det(kp2d=kp), pm)
        self.assertEqual(got.shape, (N_JOINTS, 3))
        self.assertTrue(np.isfinite(got).all())

    def test_subset_matches_the_full_run_on_those_joints(self):
        """取子集和取全集在同样的关节上必须逐位一致（原脚本只取三个）。"""
        pm = np.random.default_rng(0).normal(0, 1, (40, 40, 3))
        kp = np.stack([np.arange(N_JOINTS) + 5, np.full(N_JOINTS, 20)], 1)
        det = _det(kp2d=kp)
        full = WL.joints_from_pointmap(det, pm)
        sub = WL.joints_from_pointmap(det, pm, joint_indices=[0, 4, 8])
        for j in (0, 4, 8):
            np.testing.assert_array_equal(sub[j], full[j])
        self.assertTrue(np.isnan(sub[1]).all(), "没要的关节要留 NaN")

    def test_native_3d_adds_cam_translation(self):
        kp3d = np.zeros((N_JOINTS, 3))
        kp3d[0] = [0.1, 0.2, 0.3]
        got = WL.native_joints_3d(_det(kp3d=kp3d, cam_t=[1.0, 2.0, 3.0]))
        np.testing.assert_allclose(got[0], [1.1, 2.2, 3.3])
        np.testing.assert_allclose(got[1], [1.0, 2.0, 3.0])

    def test_wrong_keypoint_count_raises(self):
        with self.assertRaises(ValueError):
            WL.keypoints_2d(_det(kp2d=np.zeros((15, 2))))
        with self.assertRaises(ValueError):
            WL.native_joints_3d(_det(kp3d=np.zeros((15, 3))))


class TestGlobalScale(unittest.TestCase):

    def test_scale_is_scene_over_wrist_median(self):
        J = np.full((4, 2, N_JOINTS, 3), np.nan)
        J[:, HAND_RIGHT, 0, 2] = 2.0
        self.assertAlmostEqual(WL.global_scale(J, scene_depth=1.0), 0.5)

    def test_both_hands_go_into_the_same_median(self):
        """``step3d`` 是把两只手的手腕深度混在一起取中位，不是各算各的。"""
        J = np.full((4, 2, N_JOINTS, 3), np.nan)
        J[:, HAND_LEFT, 0, 2] = 1.0
        J[:, HAND_RIGHT, 0, 2] = 3.0
        self.assertAlmostEqual(WL.global_scale(J, scene_depth=2.0), 1.0)   # 中位=2

    def test_non_positive_wrist_depth_falls_back_to_no_scaling(self):
        J = np.full((4, 2, N_JOINTS, 3), np.nan)
        J[:, HAND_RIGHT, 0, 2] = -1.0
        self.assertEqual(WL.global_scale(J, scene_depth=2.0), 1.0)

    def test_no_detections_at_all_falls_back_to_no_scaling(self):
        J = np.full((4, 2, N_JOINTS, 3), np.nan)
        self.assertEqual(WL.global_scale(J, scene_depth=1.0), 1.0)

    def test_hand_nearer_than_the_scene_median_shrinks_the_whole_hand(self):
        """global-scale 那 6.5 倍缩小的机理，用合成输入钉住（不用 GPU）。

        手比"场景深度中位"近的时候，尺度 = 场景/手腕 > 1 会**放大**；而 WiLoR 原生
        手腕深度是非度量的大数（实测 ~13 m），于是尺度远小于 1，整只手被按比例缩掉。
        ABF12 上量到骨长 0.45 cm（真手 2~4 cm），就是这么来的。
        """
        J = np.full((4, 2, N_JOINTS, 3), np.nan)
        J[:, HAND_RIGHT, :, :] = 0.0
        J[:, HAND_RIGHT, 0, 2] = 13.0        # WiLoR 原生手腕深度的实测量级
        s = WL.global_scale(J, scene_depth=0.5)   # 真实场景 0.5 m
        self.assertAlmostEqual(s, 0.5 / 13.0, places=6)
        self.assertLess(s, 0.05, "尺度远小于 1 → 手被缩掉，这正是 global-scale 的病")


class TestWilorToJoints(unittest.TestCase):

    def _images(self, n=5):
        return [np.zeros((40, 40, 3), np.uint8) for _ in range(n)]

    def _infer(self, depth=0.7):
        pm = np.zeros((40, 40, 3))
        pm[..., 2] = depth
        return lambda rgb: {"points": pm, "depth": pm[..., 2]}

    def test_pointmap_mode_puts_hands_in_their_own_slots(self):
        def predict(rgb):
            return [_det(is_right=1.0), _det(is_right=0.0)]
        J = WL.wilor_to_joints(self._images(), predict, self._infer())
        self.assertEqual(J.shape, (5, 2, N_JOINTS, 3))
        self.assertTrue(np.isfinite(J[:, HAND_LEFT]).all())
        self.assertTrue(np.isfinite(J[:, HAND_RIGHT]).all())

    def test_undetected_frames_stay_nan(self):
        def predict(rgb):
            return []
        J = WL.wilor_to_joints(self._images(), predict, self._infer())
        self.assertTrue(np.isnan(J).all())

    def test_a_crashing_frame_does_not_kill_the_clip(self):
        """逐帧检测器在个别帧上炸掉是常态，整段中断反而丢掉其余可用帧。"""
        calls = {"n": 0}

        def predict(rgb):
            calls["n"] += 1
            if calls["n"] == 2:
                raise RuntimeError("WiLoR 炸了")
            return [_det()]
        J = WL.wilor_to_joints(self._images(), predict, self._infer())
        self.assertTrue(np.isnan(J[1, HAND_RIGHT]).all())
        self.assertTrue(np.isfinite(J[0, HAND_RIGHT]).all())
        self.assertTrue(np.isfinite(J[2, HAND_RIGHT]).all())
        self.assertEqual(WL.valid_frame_counts(J), {"left": 0, "right": 4})

    def test_intrinsics_switch_to_the_depth_plus_unproject_path(self):
        K = np.array([[600.0, 0, 20.0], [0, 600.0, 20.0], [0, 0, 1.0]])
        J = WL.wilor_to_joints(self._images(), lambda r: [_det()], self._infer(),
                              K=K)
        # kp2d 全在 (10,10)，深度 0.7 → x=(10-20)/600*0.7
        np.testing.assert_allclose(J[0, HAND_RIGHT, 0],
                                   [(10 - 20) / 600 * 0.7, (10 - 20) / 600 * 0.7, 0.7])

    def test_global_scale_mode_scales_the_native_3d(self):
        kp3d = np.zeros((N_JOINTS, 3))
        kp3d[:, 2] = 2.0
        J = WL.wilor_to_joints(self._images(), lambda r: [_det(kp3d=kp3d)],
                              self._infer(depth=1.0),
                              depth_mode=WL.DEPTH_GLOBAL_SCALE)
        np.testing.assert_allclose(J[:, HAND_RIGHT, 0, 2], 1.0)   # 2.0 * (1.0/2.0)

    def test_unknown_depth_mode_raises(self):
        with self.assertRaises(ValueError):
            WL.wilor_to_joints(self._images(), lambda r: [], self._infer(),
                              depth_mode="flow3r")

    def test_empty_input_raises(self):
        with self.assertRaises(ValueError):
            WL.wilor_to_joints([], lambda r: [], self._infer())

    def test_output_feeds_write_clip_unchanged(self):
        """契约层是 to_clip，这里只验形状/dtype 能直接喂进去。"""
        import tempfile

        from web2robot.perception.to_clip import write_clip
        J = WL.wilor_to_joints(self._images(), lambda r: [_det()], self._infer())
        with tempfile.TemporaryDirectory() as d:
            r = write_clip(Path(d) / "seq", J, fps=15.0)
            self.assertEqual(r["meta"]["joints_shape"], [5, 2, N_JOINTS, 3])


class TestAperture(unittest.TestCase):

    def test_same_definition_as_hawor(self):
        from web2robot.perception.hawor import aperture as hawor_aperture
        J = np.random.default_rng(3).normal(0, 1, (6, N_JOINTS, 3))
        np.testing.assert_array_equal(WL.aperture(J), hawor_aperture(J))


if __name__ == "__main__":
    unittest.main()
