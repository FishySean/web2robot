"""``perception/`` 两个模块的单测。不需要 GPU、不需要 HaWoR 仓库。

能这么测是因为迁移时刻意把依赖注入化了：``world_to_camera`` / ``aperture`` /
``write_clip`` 全是纯 numpy + json，HaWoR 的三个函数（``run_mano`` /
``run_mano_left`` / ``load_slam_cam``）是参数传进来的，这里用假的顶上。

重点钉两处**错了不会报错**的地方：

1. ``world_to_camera`` 的 einsum 转置方向 —— 转置反了相当于用逆旋转，手会跑到相机
   后面，深度全负，但流水线照样往下跑到底。
2. ``hand_joints.bin`` 的形状/dtype 与 ``hand_meta.json`` 的 ``joints_shape`` 一致 ——
   上游 ``np.fromfile`` + reshape，不一致不抛异常，只会 reshape 出错位的轨迹。
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from web2robot.perception import hawor as H  # noqa: E402
from web2robot.perception import to_clip as TC  # noqa: E402


def _rotz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


class TestWorldToCamera(unittest.TestCase):

    def test_identity_pose_is_a_pure_translation(self):
        T, K = 5, 21
        J = np.random.default_rng(0).normal(0, .3, (T, K, 3))
        R = np.tile(np.eye(3), (T, 1, 1))
        t = np.tile([0.1, -0.2, 0.5], (T, 1))
        got = H.world_to_camera(J, R, t)
        np.testing.assert_allclose(got, J + t[:, None, :], atol=1e-12)

    def test_matches_the_per_frame_per_joint_reference_loop(self):
        """和最笨的三重循环对齐 —— einsum 的下标顺序只能这么钉。"""
        rng = np.random.default_rng(1)
        T, K = 4, 7
        J = rng.normal(0, 1, (T, K, 3))
        R = np.stack([_rotz(a) for a in rng.uniform(-np.pi, np.pi, T)])
        t = rng.normal(0, 1, (T, 3))
        want = np.empty_like(J)
        for ti in range(T):
            for ki in range(K):
                want[ti, ki] = R[ti] @ J[ti, ki] + t[ti]
        np.testing.assert_allclose(H.world_to_camera(J, R, t), want, atol=1e-12)

    def test_transposing_R_gives_a_different_answer(self):
        """确认这个测试真能分辨转置方向 —— 否则上一条只是在测 einsum 会不会跑。"""
        rng = np.random.default_rng(2)
        T, K = 3, 5
        J = rng.normal(0, 1, (T, K, 3))
        R = np.stack([_rotz(a) for a in (0.7, -1.2, 2.0)])
        t = np.zeros((T, 3))
        right = H.world_to_camera(J, R, t)
        wrong = np.einsum("tji,tkj->tki", R, J)          # R 转置版
        self.assertFalse(np.allclose(right, wrong),
                         "转置反了结果却一样？那这组输入分辨不出方向，换一组")

    def test_shape_mismatch_raises_instead_of_broadcasting(self):
        J = np.zeros((4, 21, 3))
        with self.assertRaises(ValueError):                       # 位姿只有 3 帧
            H.world_to_camera(J, np.tile(np.eye(3), (3, 1, 1)), np.zeros((3, 3)))
        with self.assertRaises(ValueError):                       # J 少一维
            H.world_to_camera(np.zeros((4, 3)), np.tile(np.eye(3), (4, 1, 1)),
                              np.zeros((4, 3)))


class TestAperture(unittest.TestCase):

    def test_is_the_thumb_tip_to_index_tip_distance(self):
        J = np.zeros((3, 21, 3))
        J[:, TC.MANO_THUMB_TIP] = [0.0, 0.0, 0.0]
        J[:, TC.MANO_INDEX_TIP] = [0.03, 0.04, 0.0]     # 3-4-5 直角三角形 → 5cm
        np.testing.assert_allclose(H.aperture(J), [0.05] * 3, atol=1e-12)

    def test_uses_joints_4_and_8_not_some_other_pair(self):
        """把 4/8 之外的点全打成 NaN，开合信号仍然要算得出来。"""
        J = np.full((2, 21, 3), np.nan)
        J[:, TC.MANO_THUMB_TIP] = 0.0
        J[:, TC.MANO_INDEX_TIP] = [0.1, 0.0, 0.0]
        np.testing.assert_allclose(H.aperture(J), [0.1, 0.1], atol=1e-12)


class TestWriteClip(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.out = Path(self.tmp.name) / "seq01"
        self.J = TC.empty_joints(6)
        rng = np.random.default_rng(4)
        self.J[:, TC.HAND_RIGHT] = rng.normal(0, .2, (6, 21, 3))
        self.J[2:4, TC.HAND_LEFT] = rng.normal(0, .2, (2, 21, 3))

    def tearDown(self):
        self.tmp.cleanup()

    def test_roundtrip_through_fromfile_reshape_like_upstream_does(self):
        """按上游 clip_io 的读法读回来，必须逐位还原（含 NaN 的位置）。"""
        TC.write_clip(self.out, self.J, fps=15.0, focal=612.5)
        meta = json.loads((self.out / "hand_meta.json").read_text())
        raw = np.fromfile(self.out / "hand_joints.bin", dtype=np.float32)
        back = raw.reshape(meta["joints_shape"])
        self.assertTrue(np.array_equal(back, self.J.astype(np.float32), equal_nan=True))
        self.assertEqual(back.shape, (6, 2, 21, 3))

    def test_bin_size_matches_the_declared_shape(self):
        """字节数不对就是形状对不上 —— 这是 reshape 静默错位的唯一早期信号。"""
        TC.write_clip(self.out, self.J, fps=15.0)
        meta = json.loads((self.out / "hand_meta.json").read_text())
        want = int(np.prod(meta["joints_shape"])) * 4      # float32
        self.assertEqual((self.out / "hand_joints.bin").stat().st_size, want)

    def test_is_right_per_frame_is_one_row_per_frame_and_left0_right1(self):
        TC.write_clip(self.out, self.J, fps=15.0)
        meta = json.loads((self.out / "hand_meta.json").read_text())
        self.assertEqual(meta["n_frames"], 6)
        self.assertEqual(len(meta["is_right_per_frame"]), 6)
        self.assertEqual(meta["is_right_per_frame"][0], [False, True])

    def test_scene_defaults_and_overrides(self):
        r = TC.write_clip(self.out, self.J, fps=30.0)
        self.assertEqual(r["scene"]["camera"]["focal"], TC.DEFAULT_FOCAL)
        self.assertEqual(r["scene"]["camera"]["gravity_up"], TC.DEFAULT_GRAVITY_UP)
        self.assertEqual(r["scene"]["fps"], 30.0)
        self.assertEqual(r["scene"]["id"], "seq01")          # 默认取目录名
        r = TC.write_clip(self.out, self.J, fps=15.0, focal=700.0,
                          gravity_up=[0.0, 0.0, 1.0], clip_id="custom")
        self.assertEqual(r["scene"]["camera"]["gravity_up"], [0.0, 0.0, 1.0])
        self.assertEqual(r["scene"]["id"], "custom")

    def test_float64_input_is_written_as_float32(self):
        TC.write_clip(self.out, self.J.astype(np.float64), fps=15.0)
        self.assertEqual((self.out / "hand_joints.bin").stat().st_size,
                         6 * 2 * 21 * 3 * 4)

    def test_wrong_shape_raises(self):
        with self.assertRaises(ValueError):
            TC.write_clip(self.out, np.zeros((6, 21, 3)), fps=15.0)
        with self.assertRaises(ValueError):
            TC.write_clip(self.out, np.zeros((6, 2, 20, 3)), fps=15.0)

    def test_all_nan_clip_is_rejected(self):
        with self.assertRaises(ValueError):
            TC.write_clip(self.out, TC.empty_joints(6), fps=15.0)

    def test_valid_frame_counts(self):
        self.assertEqual(TC.valid_frame_counts(self.J), {"left": 2, "right": 6})


class TestReadFocal(unittest.TestCase):

    def test_reads_first_token(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "est_focal.txt").write_text("612.5 extra junk\n")
            self.assertAlmostEqual(H.read_focal(Path(d)), 612.5)

    def test_missing_or_garbage_falls_back(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(H.read_focal(Path(d)), TC.DEFAULT_FOCAL)
            (Path(d) / "est_focal.txt").write_text("not-a-number\n")
            self.assertEqual(H.read_focal(Path(d)), TC.DEFAULT_FOCAL)
            (Path(d) / "est_focal.txt").write_text("")
            self.assertEqual(H.read_focal(Path(d)), TC.DEFAULT_FOCAL)


class _FakeTensor:
    """够用的假 torch 张量：只要 ``[slice]`` / ``.to()`` / ``.detach().cpu().numpy()``。"""

    def __init__(self, a):
        self.a = np.asarray(a)

    def __getitem__(self, k):
        return _FakeTensor(self.a[k])

    @property
    def shape(self):
        return self.a.shape

    def to(self, _device):
        return self

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.a

    def astype(self, d):
        return self.a.astype(d)


class TestManoJointsCamera(unittest.TestCase):
    """左右手用对了 runner、valid 掩码和 NaN 的处理 —— 用假 runner 测。"""

    T = 5

    def _params(self, valid_left=None, valid_right=None):
        v = np.ones((2, self.T), bool)
        if valid_left is not None:
            v[0] = valid_left
        if valid_right is not None:
            v[1] = valid_right
        z = _FakeTensor(np.zeros((2, self.T, 3)))
        return [z, z, z, z, _FakeTensor(v)]

    @staticmethod
    def _runner(tag):
        """假 run_mano：把 tag 写进 x 坐标，这样能验出哪只手用了哪个 runner。"""
        def fn(trans, rot, hpose, betas=None):
            T = trans.shape[1]
            J = np.zeros((1, T, 21, 3))
            J[..., 0] = tag
            return {"joints": _FakeTensor(J)}
        return fn

    def test_left_and_right_use_their_own_runner(self):
        R = np.tile(np.eye(3), (self.T, 1, 1))
        t = np.zeros((self.T, 3))
        left = H.mano_joints_camera(self._params(), "left", R, t,
                                    self._runner(-1.0), device="cpu")
        right = H.mano_joints_camera(self._params(), "right", R, t,
                                     self._runner(+1.0), device="cpu")
        self.assertTrue(np.all(left[..., 0] == -1.0))
        self.assertTrue(np.all(right[..., 0] == +1.0))

    def test_invalid_frames_become_nan(self):
        R = np.tile(np.eye(3), (self.T, 1, 1))
        t = np.zeros((self.T, 3))
        v = np.array([True, False, True, True, False])
        got = H.mano_joints_camera(self._params(valid_right=v), "right", R, t,
                                   self._runner(1.0), device="cpu")
        self.assertTrue(np.isnan(got[~v]).all())
        self.assertFalse(np.isnan(got[v]).any())

    def test_non_finite_geometry_is_masked_even_when_hawor_says_valid(self):
        """HaWoR 标 valid 的帧偶尔带 inf；inf 会一路传到 IK 里静默失败。"""
        R = np.tile(np.eye(3), (self.T, 1, 1))
        t = np.zeros((self.T, 3))
        t[3] = np.inf
        got = H.mano_joints_camera(self._params(), "right", R, t,
                                   self._runner(1.0), device="cpu")
        self.assertTrue(np.isnan(got[3]).all())
        self.assertFalse(np.isnan(np.delete(got, 3, axis=0)).any())


class TestSlamPath(unittest.TestCase):

    def test_filename_carries_the_frame_count(self):
        p = H.slam_path(Path("/x/ho3d_SMu41"), 55)
        self.assertEqual(p.name, "hawor_slam_w_scale_0_55.npz")
        self.assertEqual(p.parent.name, "SLAM")


if __name__ == "__main__":
    unittest.main()
