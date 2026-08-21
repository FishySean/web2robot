"""``web2robot.twin`` 的单测 —— 物体 6D 位姿（EgoEngine §3.1 数字孪生）。

判据分三类，每类都对着一个真实会犯的错：

1. **几何换算**（``mats_to_posquat`` / ``posquat_to_mats``）—— 四元数顺序、符号、
   180° 附近的数值稳定、坏帧不许把整段搞崩。位姿约定写错是最安静的一类 bug。
2. **官方文件布局**—— ``object_pose.bin`` 是 ``(T, n_obj, 4, 4)`` 而不是
   ``(n_obj, T, 4, 4)``，两种读法都不报错但后者全错（实测差 1.35 弧度/米量级）。
   长度对不上必须抛异常，不许猜。物体 id 不连续（0/2/3/4/5）不许当下标用。
3. **落盘往返**—— 模块二只认 ``object_poses.npz``，所以"写进去再读出来必须一模
   一样"是它的地基。

跑法::

    envs/rt_env/bin/python -m unittest tests.test_twin_object_pose -v
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from web2robot.twin import (
    QUAT_ORDER, CameraIntrinsics, ObjectPoseSet, ObjectTrack, SOURCES,
    load_object_poses, mats_to_posquat, posquat_to_mats, read_official_twin,
    save_object_poses, select_task_object, track_objects,
)
from web2robot.twin.sources import read_camera, read_sam2_foundationpose_twin
from web2robot.twin.viz import (
    box_edges, draw_box, draw_pose_axes, obb_in_object_frame,
    overlay_object_poses, project,
)


# ── 造数据的小工具 ────────────────────────────────────────────────────────────

def rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


def make_mats(T=5, n_obj=2, seed=0):
    """``(T, n_obj, 4, 4)`` 一串合法的刚体变换，每个物体一条不同的螺旋轨迹。"""
    rng = np.random.default_rng(seed)
    out = np.tile(np.eye(4, dtype=np.float32), (T, n_obj, 1, 1))
    for j in range(n_obj):
        base = rng.normal(size=3) * 0.1 + np.array([0.0, 0.0, 1.5])
        for t in range(T):
            R = rot_z(0.2 * t + j) @ rot_x(0.1 * t)
            out[t, j, :3, :3] = R
            out[t, j, :3, 3] = base + np.array([0.01 * t * (j + 1), 0.0, 0.0])
    return out


def write_clip(dirpath: Path, T=6, oids=(0, 2, 3), *, with_signals=True,
               with_obb=True, with_pose_track=True, with_camera=True,
               with_meshes=(0,), states=None, layout="TN", seed=0):
    """在磁盘上造一个"官方片段"，只放 twin 那几个文件。

    ``layout="NT"`` 故意按错误的 ``(n_obj, T, 4, 4)`` 写，用来验证我们能不能靠
    长度校验挡住 —— 挡不住的话就得靠人眼发现"轨迹有点怪"，那是最坏的情况。
    """
    dirpath.mkdir(parents=True, exist_ok=True)
    n = len(oids)
    mats = make_mats(T, n, seed=seed)
    (mats if layout == "TN" else mats.transpose(1, 0, 2, 3)).astype(
        np.float32).tofile(dirpath / "object_pose.bin")

    (dirpath / "hand_meta.json").write_text(json.dumps({"n_frames": T}))
    (dirpath / "scene.json").write_text(json.dumps(
        {"fps": 15.0, "camera": {"focal": 500.0, "gravity_up": [0, -1, 0]}}))

    if with_camera:
        (dirpath / "camera.json").write_text(json.dumps(
            {"focal": 500.0, "cx": 320.0, "cy": 240.0, "width": 640,
             "height": 480, "gravity_up": [0.0, -1.0, 0.0]}))
    if with_obb:
        obb = np.zeros((n, 8, 3), np.float32)
        for j in range(n):
            c = mats[0, j, :3, 3]
            corners = np.array([[sx, sy, sz] for sx in (-.05, .05)
                                for sy in (-.04, .04) for sz in (-.03, .03)])
            obb[j] = c + corners @ mats[0, j, :3, :3].T
        obb.tofile(dirpath / "object_obb.bin")
    if with_pose_track:
        pt = {}
        for j, oid in enumerate(oids):
            pt[str(oid)] = {
                "mode": "position_first", "anchor_t": 0, "n_observed": T,
                "tracking_status": "ok", "T_seq": mats[:, j].tolist(),
                "scale_correction": 0.3 + 0.1 * j,
                "state_per_frame": (states[j] if states else ["static"] * T),
            }
        (dirpath / "pose_track.json").write_text(json.dumps(pt))
    if with_signals:
        per = {}
        for j, oid in enumerate(oids):
            trust = [True] * T
            if j == 1:                       # 第二个物体故意有几帧不可信
                trust[0] = trust[1] = False
            per[str(oid)] = {
                "trust": trust,
                "state": (states[j] if states else ["static"] * T),
            }
        (dirpath / "signals.json").write_text(json.dumps(
            {"n_frames": T, "scene": {}, "per_object": per}))
    objs = dirpath / "objects"
    objs.mkdir(exist_ok=True)
    for oid in with_meshes:
        (objs / f"obj_{oid}.ply").write_text("ply\n")
    return mats


# ── 1. 几何换算 ───────────────────────────────────────────────────────────────

class TestPoseConversions(unittest.TestCase):
    def test_identity(self):
        p = mats_to_posquat(np.eye(4))
        np.testing.assert_allclose(p, [0, 0, 0, 1, 0, 0, 0], atol=1e-6)

    def test_quat_order_is_wxyz(self):
        """绕 z 转 90°：wxyz 下是 (0.7071, 0, 0, 0.7071)。

        如果实现偷偷用了 xyzw，这个断言会立刻炸 —— 而任何"位置对、姿态错"的
        可视化人眼是看不出来的（三轴还是三轴，只是转错了）。
        """
        self.assertEqual(QUAT_ORDER, "wxyz")
        T = np.eye(4); T[:3, :3] = rot_z(np.pi / 2)
        q = mats_to_posquat(T)[3:]
        np.testing.assert_allclose(q, [np.sqrt(.5), 0, 0, np.sqrt(.5)], atol=1e-6)

    def test_roundtrip_random(self):
        mats = make_mats(T=17, n_obj=3, seed=7)
        back = posquat_to_mats(mats_to_posquat(mats))
        np.testing.assert_allclose(back, mats, atol=1e-5)

    def test_roundtrip_near_180_degrees(self):
        """接近 180° 是朴素公式会掉精度的地方，Shepperd 分支就是为这个。"""
        for axis_rot in (rot_x, rot_z):
            for ang in (np.pi - 1e-4, np.pi, np.pi + 1e-4):
                T = np.eye(4); T[:3, :3] = axis_rot(ang); T[:3, 3] = [1, 2, 3]
                back = posquat_to_mats(mats_to_posquat(T))
                np.testing.assert_allclose(back, T, atol=1e-5,
                                           err_msg=f"{axis_rot.__name__} {ang}")

    def test_quat_sign_canonical(self):
        """整段轨迹的 qw 恒 >= 0，符号不许乱跳（下游要做时序差分）。"""
        mats = make_mats(T=40, n_obj=2, seed=3)
        q = mats_to_posquat(mats)[..., 3:]
        self.assertTrue((q[..., 0] >= 0).all())

    def test_bad_frame_does_not_poison_the_rest(self):
        """一帧坏（NaN / 非旋转矩阵）只该让那一帧变 NaN。"""
        mats = make_mats(T=6, n_obj=1, seed=1).copy()
        mats[2, 0, :3, :3] = np.nan
        mats[4, 0, :3, :3] = 0.0            # det=0，不是旋转矩阵
        p = mats_to_posquat(mats)[:, 0]
        self.assertTrue(np.isnan(p[2, 3:]).all())
        self.assertTrue(np.isnan(p[4, 3:]).all())
        for t in (0, 1, 3, 5):
            self.assertTrue(np.isfinite(p[t]).all(), t)

    def test_shape_errors(self):
        with self.assertRaises(ValueError):
            mats_to_posquat(np.zeros((3, 3)))
        with self.assertRaises(ValueError):
            posquat_to_mats(np.zeros((5, 6)))

    def test_posquat_handles_zero_quat(self):
        out = posquat_to_mats(np.array([1.0, 2, 3, 0, 0, 0, 0]))
        self.assertTrue(np.isnan(out[:3, :3]).all())
        np.testing.assert_allclose(out[:3, 3], [1, 2, 3])

    def test_batch_shapes_preserved(self):
        self.assertEqual(mats_to_posquat(np.zeros((4, 5, 4, 4))).shape, (4, 5, 7))
        self.assertEqual(posquat_to_mats(np.zeros((4, 5, 7))).shape, (4, 5, 4, 4))


# ── 2. 数据结构 / 任务物体 ────────────────────────────────────────────────────

class TestObjectTrack(unittest.TestCase):
    def test_length_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            ObjectTrack(0, np.zeros((5, 7), np.float32), np.ones(4, bool))
        with self.assertRaises(ValueError):
            ObjectTrack(0, np.zeros((5, 7), np.float32), np.ones(5, bool),
                        state=np.array(["static"] * 3))

    def test_grasped_frac_without_state_is_zero_not_guessed(self):
        tr = ObjectTrack(0, np.zeros((4, 7), np.float32), np.ones(4, bool))
        self.assertEqual(tr.grasped_frac, 0.0)

    def test_grasped_frac_counts_all_three_grasp_states(self):
        st = ["static", "grasped_l", "grasped_r", "grasped_both", "moving"]
        tr = ObjectTrack(0, np.zeros((5, 7), np.float32), np.ones(5, bool),
                         state=np.array(st))
        self.assertAlmostEqual(tr.grasped_frac, 3 / 5)

    def test_travel_only_counts_valid_frames(self):
        p = np.zeros((4, 7), np.float32)
        p[:, 0] = [0.0, 1.0, 2.0, 3.0]
        all_valid = ObjectTrack(0, p, np.ones(4, bool))
        self.assertAlmostEqual(all_valid.travel, 3.0, places=5)
        some = ObjectTrack(0, p, np.array([True, False, False, True]))
        self.assertAlmostEqual(some.travel, 3.0, places=5)   # 首尾两点，直接距离

    def test_obb_reshaped_to_8x3(self):
        tr = ObjectTrack(0, np.zeros((2, 7), np.float32), np.ones(2, bool),
                         obb=np.arange(24, dtype=np.float32))
        self.assertEqual(tr.obb.shape, (8, 3))


class TestSelectTaskObject(unittest.TestCase):
    def _tr(self, oid, states, trust=None, xs=None, T=5):
        p = np.zeros((T, 7), np.float32)
        p[:, 3] = 1.0
        if xs is not None:
            p[:, 0] = xs
        return ObjectTrack(oid, p,
                           np.ones(T, bool) if trust is None else np.array(trust),
                           state=np.array(states))

    def test_picks_the_most_grasped(self):
        a = self._tr(0, ["static"] * 5)
        b = self._tr(2, ["static", "grasped_l", "grasped_l", "static", "static"])
        c = self._tr(3, ["grasped_r"] * 5)
        self.assertEqual(select_task_object([a, b, c]), 3)

    def test_tie_broken_by_trust_then_travel(self):
        a = self._tr(0, ["grasped_l"] * 5, trust=[1, 1, 0, 0, 0])
        b = self._tr(2, ["grasped_l"] * 5, trust=[1] * 5)
        self.assertEqual(select_task_object([a, b]), 2)
        c = self._tr(3, ["grasped_l"] * 5, trust=[1] * 5, xs=[0, 1, 2, 3, 4])
        self.assertEqual(select_task_object([b, c]), 3)

    def test_all_static_still_returns_something_and_is_deterministic(self):
        a = self._tr(7, ["static"] * 5)
        b = self._tr(4, ["static"] * 5)
        self.assertEqual(select_task_object([a, b]), 4)      # 全并列 → 最小 id
        self.assertEqual(select_task_object([b, a]), 4)      # 和顺序无关

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            select_task_object([])


class TestObjectPoseSet(unittest.TestCase):
    def test_frame_count_mismatch_rejected(self):
        tr = ObjectTrack(0, np.zeros((5, 7), np.float32), np.ones(5, bool))
        with self.assertRaises(ValueError):
            ObjectPoseSet(tracks=[tr], n_frames=6, fps=15.0)

    def test_task_object_auto_selected(self):
        a = ObjectTrack(0, np.zeros((4, 7), np.float32), np.ones(4, bool),
                        state=np.array(["static"] * 4))
        b = ObjectTrack(5, np.zeros((4, 7), np.float32), np.ones(4, bool),
                        state=np.array(["grasped_both"] * 4))
        s = ObjectPoseSet(tracks=[a, b], n_frames=4, fps=15.0)
        self.assertEqual(s.task_object_id, 5)
        self.assertIs(s.task_track, b)

    def test_track_lookup_by_id_not_index(self):
        s = ObjectPoseSet(
            tracks=[ObjectTrack(oid, np.zeros((3, 7), np.float32), np.ones(3, bool))
                    for oid in (0, 2, 5)],
            n_frames=3, fps=15.0)
        self.assertEqual(s.oids, [0, 2, 5])
        self.assertEqual(s.track(5).oid, 5)
        with self.assertRaises(KeyError):
            s.track(1)


# ── 3. 官方 backend ───────────────────────────────────────────────────────────

class TestOfficialSource(unittest.TestCase):
    def test_reads_TN_layout_and_matches_pose_track(self):
        """读出来的位姿必须和 ``pose_track.json`` 里的 ``T_seq`` 逐位一致。

        这正是当初定布局的判据：``(T,n,4,4)`` 差 0，``(n,T,4,4)`` 差 1.35。
        """
        with tempfile.TemporaryDirectory() as td:
            d = Path(td) / "clip"
            mats = write_clip(d, T=6, oids=(0, 2, 3))
            ps = read_official_twin(d)
            self.assertEqual(ps.n_frames, 6)
            self.assertEqual(ps.oids, [0, 2, 3])
            self.assertEqual(ps.source, "official")
            self.assertEqual(ps.frame, "camera")
            self.assertEqual(ps.fps, 15.0)
            for j, oid in enumerate([0, 2, 3]):
                got = posquat_to_mats(ps.track(oid).poses)
                np.testing.assert_allclose(got, mats[:, j], atol=1e-5)

    def test_wrong_layout_is_caught_not_silently_read(self):
        """按 ``(n_obj,T,4,4)`` 写的文件长度一样，靠长度校验挡不住 —— 但内容会和
        ``T_seq`` 不符。这里验证的是：物体数对不上时必须抛 ValueError。"""
        with tempfile.TemporaryDirectory() as td:
            d = Path(td) / "clip"
            write_clip(d, T=6, oids=(0, 2, 3))
            # 把 bin 截短一个物体的量，模拟"文件和元信息不一致"
            raw = np.fromfile(d / "object_pose.bin", dtype=np.float32)
            raw[: 6 * 2 * 16].tofile(d / "object_pose.bin")
            with self.assertRaises(ValueError) as cm:
                read_official_twin(d)
            self.assertIn("(T, n_obj, 4, 4)", str(cm.exception))

    def test_layout_NT_gives_different_numbers(self):
        """把同样的数据按错误布局写进去，读出来必须和 ``T_seq`` 不一致 ——
        这条是留给以后的人的：证明布局选错不会自己暴露，只能靠比对。"""
        with tempfile.TemporaryDirectory() as td:
            d = Path(td) / "clip"
            mats = write_clip(d, T=6, oids=(0, 2, 3), layout="NT")
            ps = read_official_twin(d)
            got = posquat_to_mats(ps.track(0).poses)
            self.assertGreater(np.abs(got - mats[:, 0]).max(), 0.1)

    def test_missing_object_pose_bin_raises_filenotfound(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td) / "clip"
            write_clip(d, T=4, oids=(0,))
            (d / "object_pose.bin").unlink()
            with self.assertRaises(FileNotFoundError) as cm:
                read_official_twin(d)
            self.assertIn("data/clips_official", str(cm.exception))

    def test_trust_from_signals_becomes_valid(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td) / "clip"
            write_clip(d, T=6, oids=(0, 2, 3))
            ps = read_official_twin(d)
            self.assertTrue(ps.track(0).valid.all())
            np.testing.assert_array_equal(
                ps.track(2).valid, [False, False, True, True, True, True])

    def test_no_signals_marks_a_note(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td) / "clip"
            write_clip(d, T=4, oids=(0,), with_signals=False)
            ps = read_official_twin(d)
            self.assertTrue(ps.track(0).valid.all())
            self.assertTrue(any("signals" in n for n in ps.notes),
                            "没有 signals.json 必须留 note，否则 valid 全 True "
                            "会被误读成'都很好'")

    def test_state_falls_back_to_pose_track(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td) / "clip"
            st = [["grasped_l"] * 4]
            write_clip(d, T=4, oids=(0,), with_signals=False, states=st)
            ps = read_official_twin(d)
            self.assertAlmostEqual(ps.track(0).grasped_frac, 1.0)

    def test_non_contiguous_ids_and_meshes(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td) / "clip"
            write_clip(d, T=4, oids=(0, 2, 5), with_meshes=(0, 5))
            ps = read_official_twin(d)
            self.assertEqual(ps.oids, [0, 2, 5])
            self.assertTrue(ps.track(0).mesh_path.endswith("obj_0.ply"))
            self.assertIsNone(ps.track(2).mesh_path)
            self.assertTrue(ps.track(5).mesh_path.endswith("obj_5.ply"))

    def test_mesh_scale_is_carried_through(self):
        """官方 mesh 在归一化 canonical 系里，公制尺寸靠 ``scale_correction``。
        漏了这个系数拿 mesh 去渲染会差 3 倍多，所以必须读出来、存下去。"""
        with tempfile.TemporaryDirectory() as td:
            d = Path(td) / "clip"
            write_clip(d, T=4, oids=(0, 2))
            ps = read_official_twin(d)
            self.assertAlmostEqual(ps.track(0).scale, 0.3, places=6)
            self.assertAlmostEqual(ps.track(2).scale, 0.4, places=6)

    def test_missing_scale_defaults_to_one(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td) / "clip"
            write_clip(d, T=4, oids=(0,))
            pt = json.loads((d / "pose_track.json").read_text())
            del pt["0"]["scale_correction"]
            (d / "pose_track.json").write_text(json.dumps(pt))
            self.assertEqual(read_official_twin(d).track(0).scale, 1.0)

    def test_obb_shape_mismatch_is_ignored_not_fatal(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td) / "clip"
            write_clip(d, T=4, oids=(0, 2))
            np.zeros(7, np.float32).tofile(d / "object_obb.bin")
            ps = read_official_twin(d)          # 包围盒只用来画图，不该拦住主流程
            self.assertIsNone(ps.track(0).obb)

    def test_without_pose_track_ids_are_inferred(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td) / "clip"
            write_clip(d, T=4, oids=(0, 2), with_pose_track=False,
                       with_signals=False)
            ps = read_official_twin(d)
            self.assertEqual(ps.oids, [0, 1])
            self.assertTrue(any("pose_track" in n for n in ps.notes))

    def test_explicit_n_frames_and_fps_win(self):
        """调用方（``test.py``）已经从 SamplesSequence 拿到帧数/帧率，口径必须一致。"""
        with tempfile.TemporaryDirectory() as td:
            d = Path(td) / "clip"
            write_clip(d, T=6, oids=(0,))
            ps = read_official_twin(d, n_frames=6, fps=30.0)
            self.assertEqual(ps.fps, 30.0)
            with self.assertRaises(ValueError):
                read_official_twin(d, n_frames=7)

    def test_camera_from_camera_json_then_scene_json(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td) / "clip"
            write_clip(d, T=4, oids=(0,))
            cam = read_camera(d)
            self.assertEqual((cam.width, cam.height), (640, 480))
            self.assertEqual((cam.cx, cam.cy), (320.0, 240.0))
            (d / "camera.json").unlink()
            cam2 = read_camera(d)
            self.assertEqual(cam2.focal, 500.0)
            self.assertEqual((cam2.width, cam2.height), (0, 0))

    def test_tracking_status_not_ok_is_noted(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td) / "clip"
            write_clip(d, T=4, oids=(0,))
            pt = json.loads((d / "pose_track.json").read_text())
            pt["0"]["tracking_status"] = "failed"
            (d / "pose_track.json").write_text(json.dumps(pt))
            ps = read_official_twin(d)
            self.assertTrue(any("跟踪不 ok" in n for n in ps.notes))


class TestSourceRegistry(unittest.TestCase):
    def test_registry_contents(self):
        self.assertEqual(sorted(SOURCES), ["official", "sam2_foundationpose"])

    def test_unknown_source_raises(self):
        with self.assertRaises(ValueError):
            track_objects("/nonexistent", source="whatever")

    def test_sam2_backend_refuses_loudly(self):
        """未实现的 backend 必须报错，**不许静默退化到 official** ——
        否则"我们跑的孪生"和"官方发布的孪生"会在结果里混成一团。"""
        with self.assertRaises(NotImplementedError) as cm:
            read_sam2_foundationpose_twin("/nonexistent")
        msg = str(cm.exception)
        self.assertIn("mesh", msg)
        self.assertIn("FoundationStereo", msg)
        with self.assertRaises(NotImplementedError):
            track_objects("/nonexistent", source="sam2_foundationpose")


# ── 4. 落盘往返 ───────────────────────────────────────────────────────────────

class TestSaveLoad(unittest.TestCase):
    def _make(self, T=6, oids=(0, 2, 5)):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td) / "clip"
            write_clip(d, T=T, oids=oids, with_meshes=(0, 5),
                       states=[["static"] * T, ["grasped_l"] * T, ["static"] * T])
            return read_official_twin(d)

    def test_roundtrip_identical(self):
        ps = self._make()
        with tempfile.TemporaryDirectory() as td:
            p = save_object_poses(Path(td) / "object_poses.npz", ps)
            back = load_object_poses(p)
        self.assertEqual(back.oids, ps.oids)
        self.assertEqual(back.task_object_id, ps.task_object_id)
        self.assertEqual(back.n_frames, ps.n_frames)
        self.assertEqual(back.fps, ps.fps)
        self.assertEqual(back.source, ps.source)
        self.assertEqual(back.frame, ps.frame)
        self.assertEqual(back.notes, ps.notes)
        for oid in ps.oids:
            a, b = ps.track(oid), back.track(oid)
            np.testing.assert_array_equal(a.poses, b.poses)
            np.testing.assert_array_equal(a.valid, b.valid)
            np.testing.assert_array_equal(a.state, b.state)
            np.testing.assert_allclose(a.obb, b.obb)
            self.assertEqual(a.mesh_path, b.mesh_path)
            self.assertAlmostEqual(a.scale, b.scale, places=6)

    def test_npz_keys_and_shapes(self):
        """接口是磁盘格式，键名和形状被下游依赖，改了要有人知道。"""
        ps = self._make(T=6, oids=(0, 2, 5))
        with tempfile.TemporaryDirectory() as td:
            p = save_object_poses(Path(td) / "object_poses.npz", ps)
            z = np.load(p, allow_pickle=False)
            self.assertEqual(z["object_poses"].shape, (6, 7))
            self.assertEqual(z["object_poses"].dtype, np.float32)
            self.assertEqual(z["object_valid"].shape, (6,))
            self.assertEqual(z["object_poses_all"].shape, (6, 3, 7))
            self.assertEqual(z["object_valid_all"].shape, (6, 3))
            self.assertEqual(z["object_ids"].tolist(), [0, 2, 5])
            self.assertEqual(z["object_obb"].shape, (3, 8, 3))
            self.assertEqual(z["object_scale"].shape, (3,))
            self.assertEqual(str(z["quat_order"]), "wxyz")
            self.assertEqual(str(z["frame"]), "camera")
            self.assertEqual(int(z["task_object_id"]), 2)
            self.assertEqual(int(z["n_frames"]), 6)

    def test_object_poses_is_the_task_object(self):
        ps = self._make()
        with tempfile.TemporaryDirectory() as td:
            z = np.load(save_object_poses(Path(td) / "x.npz", ps))
            i = ps.oids.index(ps.task_object_id)
            np.testing.assert_array_equal(z["object_poses"],
                                          z["object_poses_all"][:, i])

    def test_camera_survives_roundtrip(self):
        ps = self._make()
        with tempfile.TemporaryDirectory() as td:
            back = load_object_poses(save_object_poses(Path(td) / "x.npz", ps))
        self.assertAlmostEqual(back.camera.focal, 500.0)
        self.assertEqual((back.camera.width, back.camera.height), (640, 480))
        np.testing.assert_allclose(back.camera.gravity_up, [0, -1, 0])

    def test_roundtrip_without_camera_or_state(self):
        tr = ObjectTrack(3, np.zeros((4, 7), np.float32), np.ones(4, bool))
        ps = ObjectPoseSet(tracks=[tr], n_frames=4, fps=20.0, camera=None)
        with tempfile.TemporaryDirectory() as td:
            back = load_object_poses(save_object_poses(Path(td) / "x.npz", ps))
        self.assertIsNone(back.camera)
        self.assertIsNone(back.track(3).state)
        self.assertIsNone(back.track(3).obb)

    def test_empty_track_list_roundtrips(self):
        ps = ObjectPoseSet(tracks=[], n_frames=5, fps=15.0)
        with tempfile.TemporaryDirectory() as td:
            p = save_object_poses(Path(td) / "x.npz", ps)
            z = np.load(p)
            self.assertEqual(z["object_poses"].shape, (5, 7))
            self.assertFalse(z["object_valid"].any())
            back = load_object_poses(p)
        self.assertEqual(back.tracks, [])
        self.assertEqual(back.task_object_id, -1)

    def test_wrong_quat_order_on_disk_is_rejected(self):
        ps = self._make()
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "x.npz"
            save_object_poses(p, ps)
            z = dict(np.load(p))
            z["quat_order"] = np.array("xyzw")
            np.savez(p, **z)
            with self.assertRaises(ValueError):
                load_object_poses(p)

    def test_save_creates_parent_dirs(self):
        ps = self._make()
        with tempfile.TemporaryDirectory() as td:
            p = save_object_poses(Path(td) / "a" / "b" / "object_poses.npz", ps)
            self.assertTrue(p.exists())


# ── 5. 可视化 ─────────────────────────────────────────────────────────────────

class TestViz(unittest.TestCase):
    def test_project_basic_and_behind_camera(self):
        uv, ok = project(np.array([[0, 0, 2.0], [0, 0, -2.0], [1, 0, 2.0]]),
                         fx=500.0, cx=320.0, cy=240.0)
        self.assertTrue(ok[0] and not ok[1] and ok[2])
        np.testing.assert_allclose(uv[0], [320, 240])
        np.testing.assert_allclose(uv[2], [320 + 250, 240])

    def test_project_nan_is_invalid(self):
        uv, ok = project(np.array([[np.nan, 0, 2.0]]), 500.0, 0.0, 0.0)
        self.assertFalse(ok[0])

    def test_box_edges_is_order_independent(self):
        """角点顺序未知，所以棱是**算**出来的。打乱顺序后同一条棱集合。"""
        c = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-2, 2)
                      for sz in (-3, 3)], dtype=np.float64)
        e1 = box_edges(c)
        self.assertEqual(len(e1), 12)
        perm = np.array([5, 0, 7, 2, 1, 6, 3, 4])
        e2 = box_edges(c[perm])
        inv = np.argsort(perm)
        remap = sorted(tuple(sorted((int(inv[i]), int(inv[j])))) for i, j in e1)
        self.assertEqual(sorted(e2), remap)

    def _box(self, a, b, c, R=None):
        pts = np.array([[sx * a, sy * b, sz * c] for sx in (-1, 1)
                        for sy in (-1, 1) for sz in (-1, 1)], dtype=np.float64)
        return pts if R is None else pts @ R.T

    def test_box_edges_on_elongated_box(self):
        """细长盒是第一版实现画错的地方：薄面对角线比长棱短。

        判据不是"有 12 条"（错的实现也能凑出 12 条），而是**每条棱的长度只能是
        2a/2b/2c 三种之一** —— 面对角线混进来就会出现第四种长度。
        """
        for dims in [(0.086, 0.042, 0.226),      # --oo8_XIuOM 那根圆柱的实测半长
                     (0.01, 0.01, 1.0),
                     (1.0, 1.0, 0.02)]:
            for R in (None, rot_z(0.6) @ rot_x(0.3)):
                c = self._box(*dims, R=R)
                edges = box_edges(c)
                self.assertEqual(len(edges), 12, f"{dims} {R is not None}")
                lens = sorted({round(float(np.linalg.norm(c[i] - c[j])), 6)
                               for i, j in edges})
                want = sorted({round(2 * d, 6) for d in dims})
                self.assertEqual(lens, want, f"{dims} 画出了非棱的连线：{lens}")
                deg = np.bincount([i for e in edges for i in e], minlength=8)
                self.assertTrue((deg == 3).all(), f"{dims} 每个角点该连 3 条：{deg}")

    def test_box_edges_degenerate_does_not_raise(self):
        """退化盒（全部角点重合）凑不出 3 组 4 条，退回最近邻，不许抛。"""
        self.assertIsInstance(box_edges(np.zeros((8, 3))), list)

    def test_obb_in_object_frame_then_back(self):
        T = np.eye(4); T[:3, :3] = rot_z(0.7); T[:3, 3] = [0.1, 0.2, 1.5]
        pose0 = mats_to_posquat(T)
        corners = np.array([[sx, sy, sz] for sx in (-.05, .05)
                            for sy in (-.04, .04) for sz in (-.03, .03)])
        world = corners @ T[:3, :3].T + T[:3, 3]
        local = obb_in_object_frame(world, pose0)
        np.testing.assert_allclose(local, corners, atol=1e-6)

    def test_draw_helpers_do_not_crash_on_bad_input(self):
        img = np.zeros((64, 64, 3), np.uint8)
        self.assertFalse(draw_pose_axes(img, np.full(7, np.nan), 50.0, 32, 32))
        self.assertFalse(draw_pose_axes(
            img, np.array([0, 0, -1.0, 1, 0, 0, 0]), 50.0, 32, 32))
        self.assertTrue(draw_pose_axes(
            img, np.array([0, 0, 1.0, 1, 0, 0, 0]), 50.0, 32, 32, label="o0"))
        self.assertTrue(img.any(), "画完了图上应该有东西")
        draw_box(img, np.full((8, 3), np.nan), 50.0, 32, 32)   # 不许抛

    def test_overlay_runs_and_pads_short_video(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td) / "clip"
            write_clip(d, T=6, oids=(0, 2),
                       states=[["static"] * 6, ["grasped_l"] * 6])
            ps = read_official_twin(d)
        frames = [np.zeros((48, 64, 3), np.uint8) for _ in range(4)]  # 少 2 帧
        out = overlay_object_poses(frames, ps, axis_len=0.05)
        self.assertEqual(len(out), 6)
        self.assertEqual(out[0].shape, (48, 64, 3))
        self.assertTrue(any(f.any() for f in out))

    def test_overlay_empty_frames(self):
        ps = ObjectPoseSet(tracks=[], n_frames=3, fps=15.0)
        self.assertEqual(overlay_object_poses([], ps), [])


# ── 6. CLI ────────────────────────────────────────────────────────────────────

class TestCli(unittest.TestCase):
    def test_summarize_flags_task_object(self):
        from web2robot.twin.cli import summarize
        with tempfile.TemporaryDirectory() as td:
            d = Path(td) / "clip"
            write_clip(d, T=5, oids=(0, 2), with_meshes=(0,),
                       states=[["static"] * 5, ["grasped_both"] * 5])
            info = summarize(read_official_twin(d))
        self.assertEqual(info["task_object_id"], 2)
        self.assertEqual(info["n_objects"], 2)
        task = [r for r in info["objects"] if r["is_task"]]
        self.assertEqual(len(task), 1)
        self.assertEqual(task[0]["oid"], 2)
        self.assertEqual(task[0]["grasped_frac"], 1.0)
        self.assertTrue([r for r in info["objects"] if r["oid"] == 0][0]["has_mesh"])

    def test_main_outputs_exist(self):
        import contextlib
        import io
        from web2robot.twin.cli import main
        with tempfile.TemporaryDirectory() as td:
            d = Path(td) / "clip"
            write_clip(d, T=5, oids=(0, 2))
            out = Path(td) / "out"
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(["--clip", str(d), "--out", str(out)]), 0)
            self.assertTrue((out / "object_poses.npz").exists())
            info = json.loads((out / "object_poses.json").read_text())
            self.assertEqual(info["n_frames"], 5)
            self.assertEqual(load_object_poses(out / "object_poses.npz").n_frames, 5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
