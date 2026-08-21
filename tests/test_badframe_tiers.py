"""坏帧过滤的三个粒度（EgoSmith / EgoSteer arXiv 2607.09701）—— 新增两层的隔离测试。

这个测试守的东西按重要性排序：

1. **默认跑法什么都没变。** ``--bad_frame_tiers`` 默认只有 ``frame``，
   :func:`run_extra_tiers` 在那种情况下必须返回 ``{}`` —— 调用方据此**不写**报告文件，
   所以产物连"多一个文件"都不会发生。这条是逐字节不变那个硬要求在单测层面的凭据
   （端到端的凭据是 ``scripts/dev/check_tiers_yaml_bytes.sh`` 的 md5）。
2. **新增两层一个数都不改。** 每个调用前后都拿原数组的副本逐位比对。这是我们和原文
   最大的分歧（原文 discard，我们只警告/只标记），所以它必须是**测出来的**性质，
   不是 docstring 里的承诺。
3. 判据本身能在合成信号上把该抓的抓出来、把该放的放过（退化输入不炸）。

跑法::

    envs/rt_env/bin/python -m unittest tests.test_badframe_tiers -v
"""
import ast
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from web2robot.retarget.fallback import run_extra_tiers, save_tier_report  # noqa: E402
from web2robot.trajectory.tiers import (  # noqa: E402
    DEFAULT_TIERS, TIER_NAMES, episode_camera_check, parse_tiers,
    segment_spatial_check,
)


def _texture(h=120, w=160, seed=0):
    """低频纹理图 —— Farneback 在纯白噪声上跟不住，在这种块状图案上才稳。"""
    rng = np.random.default_rng(seed)
    small = rng.integers(0, 255, size=(h // 8, w // 8), dtype=np.uint8)
    import cv2
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def _panning_frames(shifts):
    """按 ``shifts`` 逐帧累计横移同一张纹理图 → 已知每对帧的"相机平移"量。"""
    base = _texture()
    out, acc = [], 0
    for s in [0] + list(shifts):
        acc += s
        out.append(np.roll(base, acc, axis=1))
    return out


def _straight_traj(T=60, quat=(1.0, 0.0, 0.0, 0.0)):
    """(T,7) 匀速直线手腕轨迹，四元数恒等。"""
    traj = np.zeros((T, 7), dtype=np.float64)
    traj[:, 0] = np.linspace(0.0, 0.10, T)      # x 缓慢前伸
    traj[:, 1] = 0.05
    traj[:, 2] = 0.40
    traj[:, 3:] = np.asarray(quat)
    return traj


def _joints_for(traj, spread=0.02, seed=1):
    """(T,21,3) 相机系关键点：手腕位置 + 一团固定的相对偏移（手形不变）。"""
    rng = np.random.default_rng(seed)
    offs = rng.normal(0.0, spread, size=(21, 3))
    return traj[:, None, :3] + offs[None]


class TestParseTiers(unittest.TestCase):
    def test_default_is_frame_only(self):
        """默认等于现状不变 —— 这是整个任务的前提，写死在这儿。"""
        self.assertEqual(DEFAULT_TIERS, ("frame",))
        self.assertEqual(parse_tiers("frame"), ("frame",))

    def test_order_is_normalised_coarse_to_fine(self):
        for spec in ("frame,episode,segment", "segment, frame ,episode",
                     "episode,segment,frame"):
            self.assertEqual(parse_tiers(spec), ("episode", "segment", "frame"),
                             spec)

    def test_duplicates_collapse(self):
        self.assertEqual(parse_tiers("frame,frame"), ("frame",))

    def test_empty_means_all_off(self):
        self.assertEqual(parse_tiers(""), ())
        self.assertEqual(parse_tiers(" , "), ())

    def test_typo_raises_instead_of_silently_disabling(self):
        """``epsiode`` 打错一个字母不能变成"什么都没开"而人以为开了。"""
        with self.assertRaises(ValueError) as cm:
            parse_tiers("epsiode,frame")
        self.assertIn("epsiode", str(cm.exception))
        for name in TIER_NAMES:
            self.assertIn(name, str(cm.exception))


class TestEpisodeTier(unittest.TestCase):
    def test_too_few_frames_is_not_a_warning(self):
        rep = episode_camera_check(_panning_frames([1, 1]))
        self.assertFalse(rep.warn)
        self.assertIn("帧数太少", rep.reason)
        self.assertIsNone(rep.flow_mad)

    def test_static_clip_does_not_divide_by_zero(self):
        """MAD=0（画面几乎全静止）时不能除 —— 那种段本来就不该报警。"""
        still = [_texture()] * 12
        rep = episode_camera_check(still)
        self.assertFalse(rep.warn)
        self.assertEqual(rep.outlier_frames, [])
        self.assertEqual(rep.max_robust_z, 0.0)

    def test_steady_pan_is_not_flagged(self):
        """整段匀速平移 = 分布里没有离群，哪怕运动量本身不小。

        这一层判的是**离群**，不是"动得多不多"（绝对阈值不可迁移的理由见
        tiers.py 的"我们做不到原文哪一步"第 4 条）。
        """
        rep = episode_camera_check(_panning_frames([2] * 20))
        self.assertFalse(rep.warn, rep.reason)

    def test_injected_lurches_are_flagged(self):
        shifts = [1] * 24
        for i in (7, 8, 15):
            shifts[i] = 22               # 三次突然的大幅镜头运动
        rep = episode_camera_check(_panning_frames(shifts))
        self.assertTrue(rep.warn, rep.reason)
        self.assertTrue(set(rep.outlier_frames) & {7, 8, 15},
                        rep.outlier_frames)
        self.assertIn("§V2", rep.reason)
        self.assertIn("§V3", rep.reason)

    def test_report_is_json_serialisable_and_cites_the_clause(self):
        rep = episode_camera_check(_panning_frames([1] * 12))
        d = rep.to_dict()
        self.assertEqual(d["clause"], "§V2/§V3")
        json.dumps(d)                     # 会直接写进 bad_frame_tiers.json


class TestSegmentTier(unittest.TestCase):
    def test_smooth_trajectory_has_no_findings(self):
        traj = _straight_traj()
        self.assertEqual(
            segment_spatial_check(traj, _joints_for(traj), fps=15.0, side="left"),
            [])

    def test_injected_wrist_lurch_is_marked(self):
        traj = _straight_traj()
        traj[20, :3] += np.array([0.0, 0.0, 0.6])    # 一帧手腕深度爆点
        found = segment_spatial_check(traj, None, fps=15.0, side="left")
        wrist = [f for f in found if f.kind == "wrist_outlier"]
        self.assertTrue(wrist, found)
        f = wrist[0]
        self.assertLessEqual(f.start, 20)
        self.assertGreaterEqual(f.end, 20)
        self.assertEqual(f.side, "left")
        self.assertEqual(f.clause, "§V2/§V3")
        self.assertGreater(f.score, 3.5)

    def test_injected_finger_jitter_is_marked(self):
        traj = _straight_traj()
        joints = _joints_for(traj)
        joints[33, 4] += np.array([0.25, 0.0, 0.0])   # 一根手指飞出去一帧
        found = segment_spatial_check(traj, joints, fps=15.0, side="right")
        kinds = {f.kind for f in found}
        self.assertIn("finger_outlier", kinds, found)
        self.assertEqual({f.side for f in found}, {"right"})

    def test_nothing_is_modified(self):
        """**只标记不修改**是这一层的定义性质，所以要逐位测出来。"""
        traj = _straight_traj()
        traj[20, :3] += np.array([0.0, 0.0, 0.6])
        joints = _joints_for(traj)
        joints[33, 4] += np.array([0.25, 0.0, 0.0])
        t0, j0 = traj.copy(), joints.copy()
        found = segment_spatial_check(traj, joints, fps=15.0, side="left")
        self.assertTrue(found)                        # 确实检出了东西
        self.assertTrue(np.array_equal(traj, t0, equal_nan=True))
        self.assertTrue(np.array_equal(joints, j0, equal_nan=True))

    def test_hold_filled_segment_with_one_spike_is_still_caught(self):
        """零阶保持填出来的一段是**逐位常数** → MAD 恰好 0。

        MAD 口径在这里是 0/0，会把"一帧爆点"判成"无离群" —— 而那正是最该抓的情形。
        所以稳健 z 在 MAD=0 时退回平均绝对偏差（Iglewicz–Hoaglin 给 MAD=0 的处方，
        见 tiers.py::_robust_z）。这个测试就是钉这条退路。
        """
        traj = np.zeros((40, 7))
        traj[:, :3] = np.array([0.1, 0.0, 0.45])      # 整段一模一样（保持填出来的）
        traj[:, 3] = 1.0
        traj[17, 2] += 0.5                            # 一帧深度爆点
        found = segment_spatial_check(traj, None, fps=15.0, side="left")
        self.assertTrue([f for f in found if f.kind == "wrist_outlier"], found)

    def test_truly_constant_segment_is_not_flagged(self):
        """两个尺度都是 0（全部相等）才是真退化 —— 那种段不该报任何东西。"""
        traj = np.zeros((40, 7))
        traj[:, :3] = np.array([0.1, 0.0, 0.45])
        traj[:, 3] = 1.0
        self.assertEqual(segment_spatial_check(traj, None, fps=15.0), [])

    def test_degenerate_inputs_do_not_raise(self):
        self.assertEqual(segment_spatial_check(np.zeros((0, 7)), None, 15.0), [])
        allnan = np.full((40, 7), np.nan)
        self.assertEqual(segment_spatial_check(allnan, None, 15.0), [])
        # 四元数全零（不可归一化）→ 转不出旋转矩阵，跳过而不是抛
        zeroq = _straight_traj()
        zeroq[:, 3:] = 0.0
        self.assertEqual(segment_spatial_check(zeroq, None, 15.0), [])

    def test_sparse_segment_is_skipped(self):
        """一小段里有效帧太少就不判 —— 3 个点的"分布"没有意义。"""
        traj = _straight_traj(T=30)
        traj[:, :] = np.nan
        traj[:3] = _straight_traj(T=30)[:3]           # 只剩 3 帧有效
        self.assertEqual(segment_spatial_check(traj, None, fps=15.0), [])


class TestRunExtraTiers(unittest.TestCase):
    def setUp(self):
        self.traj = _straight_traj()
        self.traj[20, :3] += np.array([0.0, 0.0, 0.6])
        self.joints = _joints_for(self.traj)

    def test_frame_only_returns_empty_report(self):
        """默认跑法：两层都没开 → ``{}`` → 调用方不写文件 → 产物逐字节不变。"""
        rep = run_extra_tiers(("frame",), self.traj, self.traj, 15.0,
                              frames=_panning_frames([1] * 12), log=lambda _m: None)
        self.assertEqual(rep, {})

    def test_no_tiers_at_all_returns_empty_report(self):
        self.assertEqual(run_extra_tiers((), self.traj, self.traj, 15.0,
                                         log=lambda _m: None), {})

    def test_segment_tier_marks_both_hands_without_touching_them(self):
        l0, r0, j0 = self.traj.copy(), self.traj.copy(), self.joints.copy()
        rep = run_extra_tiers(("segment",), self.traj, self.traj, 15.0,
                              joints_left=self.joints, joints_right=self.joints,
                              log=lambda _m: None)
        self.assertEqual(set(rep), {"tiers", "segment"})
        self.assertEqual(rep["segment"]["clause"], "§V2/§V3")
        sides = {f["side"] for f in rep["segment"]["findings"]}
        self.assertEqual(sides, {"left", "right"}, rep["segment"]["findings"])
        self.assertTrue(np.array_equal(self.traj, l0, equal_nan=True))
        self.assertTrue(np.array_equal(self.traj, r0, equal_nan=True))
        self.assertTrue(np.array_equal(self.joints, j0, equal_nan=True))

    def test_episode_tier_without_video_frames_degrades_to_a_note(self):
        rep = run_extra_tiers(("episode",), self.traj, self.traj, 15.0,
                              frames=None, log=lambda _m: None)
        self.assertFalse(rep["episode"]["warn"])
        self.assertIn("没有视频帧", rep["episode"]["reason"])

    def test_episode_warning_does_not_block_anything(self):
        """整段级检出 = 只警告。函数返回 report，不抛、不改、不返回"要不要继续"。"""
        shifts = [1] * 24
        for i in (7, 8, 15):
            shifts[i] = 22
        logged = []
        rep = run_extra_tiers(("episode", "segment"), self.traj, self.traj, 15.0,
                              frames=_panning_frames(shifts),
                              joints_left=self.joints, joints_right=self.joints,
                              log=logged.append)
        self.assertTrue(rep["episode"]["warn"], rep["episode"]["reason"])
        self.assertEqual(rep["tiers"], ["episode", "segment"])
        self.assertTrue(any("§V2/§V3" in m for m in logged), logged)
        self.assertTrue(any("不做任何自动处理" in m for m in logged), logged)

    def test_report_keys_are_only_the_three_tiers(self):
        """report 里不该出现轨迹数据 —— 这一层的输出是信号，不是数据。"""
        rep = run_extra_tiers(("episode", "segment", "frame"), self.traj, self.traj,
                              15.0, frames=_panning_frames([1] * 12),
                              log=lambda _m: None)
        self.assertLessEqual(set(rep), {"tiers", "episode", "segment"})
        json.dumps(rep)

    def test_saved_json_says_where_the_method_came_from_and_how_we_differ(self):
        rep = run_extra_tiers(("segment",), self.traj, self.traj, 15.0,
                              log=lambda _m: None)
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "bad_frame_tiers.json"
            save_tier_report(p, rep, clip="clip_x", robot="m7")
            got = json.loads(p.read_text())
        self.assertEqual(got["clip"], "clip_x")
        self.assertEqual(got["robot"], "m7")
        self.assertIn("2607.09701", got["source"])
        self.assertIn("§V2/§V3", got["source"])
        self.assertIn("discard", got["source"])       # 说清和原文的分歧


class TestUpstreamFlagDefault(unittest.TestCase):
    """上游 ``scripts/test.py`` 的 ``--bad_frame_tiers`` 默认值必须是 ``"frame"``。

    这是"默认行为不变"那条硬要求的**入口**：判据模块里的 DEFAULT_TIERS 再怎么对，
    命令行默认值一旦变了，所有历史 md5 就全部失效。用 AST 读而不是 import，是因为
    上游那个文件 import 就要 torch/mujoco 全家。
    """
    UPSTREAM = REPO / "external/EgoInfinity/retarget/scripts/test.py"

    @unittest.skipUnless(UPSTREAM.exists(), "上游仓库未挂载")
    def test_default_is_frame(self):
        tree = ast.parse(self.UPSTREAM.read_text())
        defaults = {}
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "add_argument"):
                continue
            if not (node.args and isinstance(node.args[0], ast.Constant)):
                continue
            flag = node.args[0].value
            for kw in node.keywords:
                if kw.arg == "default" and isinstance(kw.value, ast.Constant):
                    defaults[flag] = kw.value.value
        self.assertIn("--bad_frame_tiers", defaults,
                      "上游没有这个开关了？三层过滤的入口没了")
        self.assertEqual(defaults["--bad_frame_tiers"], "frame")
        self.assertEqual(parse_tiers(defaults["--bad_frame_tiers"]), DEFAULT_TIERS)


if __name__ == "__main__":
    unittest.main(verbosity=2)
