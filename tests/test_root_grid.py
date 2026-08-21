"""``retarget/root_grid.py`` 的隔离测试 —— 纯 numpy，不要 GPU、不要 checkpoint。

这个模块的输入是"末端位置数组 + 一个打分闭包"，所以可以完全在假 IK 下测：把
``score_fn`` 换成一个解析可算的可行性代理，就能钉住三类容易静默出错的性质。

1. **关键帧选取**：凸包顶点是不是真的取到了空间极值、退化轨迹（共线/共面/点太少）
   有没有走兜底、兜底走了有没有记在 ``source`` 里。
2. **剪枝是可采纳的**：上界剪枝版和穷举版必须给出**完全相同**的解。这条是本模块
   唯一一处"为了快而改变了搜索顺序"的地方，必须钉死，不然以后没人敢碰它。
3. **空操作会炸**：候选平移不起作用时（``workspace_center`` 那个坑）抛
   ``RuntimeError`` 而不是返回一个看起来没问题的位姿。这条是照着
   ``postik-smoother-noop`` 那次教训加的。

跑法::

    envs/rt_env/bin/python -m unittest tests.test_root_grid -v
"""
import sys
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from web2robot.retarget.root_grid import (  # noqa: E402
    build_translation_grid, estimate_reach, gravity_yaw_candidates,
    make_keyframe_scorer, select_extremal_keyframes, solve_root_pose_grid,
)


# ── 关键帧选取 ─────────────────────────────────────────────────────────────────

class TestKeyframeSelection(unittest.TestCase):

    def test_cube_corners_are_keyframes_interior_points_are_not(self):
        """立方体 8 个角 + 一堆内部点：只有角所在的帧进 K。

        这就是"covering the spatial extremes"的可检验版本 —— 内部点无论多少个，
        都不该占用关键帧预算（每个关键帧都要付一次 IK batch 的钱）。
        """
        corners = np.array([[x, y, z] for x in (0., 1.) for y in (0., 1.) for z in (0., 1.)])
        rng = np.random.default_rng(0)
        interior = rng.uniform(0.3, 0.7, size=(20, 3))       # 严格在内部
        traj = np.concatenate([corners, interior])           # 帧 0..7 是角

        kf = select_extremal_keyframes(traj)
        self.assertEqual(kf.source, "convex_hull")
        self.assertEqual(kf.indices.tolist(), list(range(8)))
        self.assertEqual(kf.n_frames, 28)

    def test_collinear_trajectory_falls_back_to_range_extremes(self):
        """一条直线上的轨迹：Qhull 会报退化，必须退到 range 兜底并如实标注。"""
        t = np.linspace(0, 1, 30)[:, None]
        traj = t * np.array([[1., 2., 3.]])                  # 全在一条直线上

        kf = select_extremal_keyframes(traj)
        self.assertEqual(kf.source, "range_extremes")
        self.assertEqual(kf.indices.tolist(), [0, 29])       # 两个端点

    def test_planar_trajectory_falls_back_too(self):
        """桌面上的平面轨迹（共面）也是退化 —— 这是常见输入，不是异常输入。"""
        th = np.linspace(0, 2 * np.pi, 40, endpoint=False)
        traj = np.stack([np.cos(th), np.sin(th), np.zeros_like(th)], axis=1)

        kf = select_extremal_keyframes(traj)
        self.assertEqual(kf.source, "range_extremes")
        self.assertGreaterEqual(len(kf.indices), 4)          # 至少 x/y 各两端

    def test_too_few_points_falls_back(self):
        """3 帧单手 = 3 个点，凸包要 4 个非共面点才成立。"""
        kf = select_extremal_keyframes(np.array([[0., 0, 0], [1, 0, 0], [0, 1, 0]]))
        self.assertEqual(kf.source, "range_extremes")

    def test_a_frame_enters_K_if_either_hand_is_extremal(self):
        """双手：只要有**一只**手贡献了凸包顶点，这一帧就进 K。

        底座要同时照顾两只手，"左手伸到最远"那一帧对底座的约束和右手的一样硬，
        所以不能要求两只手同时极端。
        """
        # 立方体 8 个角劈成两半：左手在帧 0..3 摸到 4 个角，右手在帧 4..7 摸到另 4 个，
        # 另一只手同时停在体心（内部点，不该进 K）。8 个角都在凸位置上，所以
        # 每一帧都靠**其中一只**手进 K —— 正好是要测的那条"或"语义。
        cube = np.array([[x, y, z] for x in (0., 1.) for y in (0., 1.) for z in (0., 1.)])
        center = np.full((4, 3), 0.5)
        left  = np.concatenate([cube[:4], center])                       # 帧 0..3 极端
        right = np.concatenate([center, cube[4:]])                       # 帧 4..7 极端
        traj = np.stack([left, right], axis=1)                          # (8, 2, 3)

        kf = select_extremal_keyframes(traj)
        self.assertEqual(kf.source, "convex_hull")
        self.assertEqual(kf.indices.tolist(), list(range(8)))

    def test_max_keyframes_keeps_the_farthest_ones(self):
        """截断时留离质心最远的，因为它们对底座的约束最紧。"""
        corners = np.array([[x, y, z] for x in (-1., 1.) for y in (-1., 1.) for z in (-1., 1.)])
        traj = np.concatenate([corners, np.array([[3., 0, 0], [-3., 0, 0]])])  # 帧 8,9 最远

        kf = select_extremal_keyframes(traj, max_keyframes=2)
        self.assertEqual(kf.indices.tolist(), [8, 9])

    def test_rejects_bad_shapes(self):
        for bad in (np.zeros((5, 2)), np.zeros((5, 2, 4)), np.zeros((0, 3))):
            with self.assertRaises(ValueError):
                select_extremal_keyframes(bad)


# ── 候选网格 ───────────────────────────────────────────────────────────────────

class TestTranslationGrid(unittest.TestCase):

    def test_centroid_itself_is_always_a_candidate(self):
        """质心必须**恰好**落在格点上，不能靠格距碰巧对齐。

        "最优底座就在质心"是完全可能的情形（人站着不动干活），如果格子是
        ``arange(-r, r, step)`` 这种不含 0 的排法，这个解就只能被近似碰到。
        """
        c = np.array([0.37, -1.21, 0.88])
        grid, spec = build_translation_grid(c, r_max=0.8, spacing=0.05)
        self.assertTrue(np.isclose(np.abs(grid - c).sum(axis=1), 0).any())
        np.testing.assert_allclose(spec.centroid, c)

    def test_shape_and_extent(self):
        """半径/格距 → 每轴 ``2*floor(r/s)+1`` 个点，且不超出半径。"""
        grid, spec = build_translation_grid(np.zeros(3), r_max=0.5, spacing=0.1)
        self.assertEqual(spec.shape[:2], (11, 11))           # -0.5..0.5 步长 0.1
        self.assertLessEqual(np.abs(grid[:, :2]).max(), 0.5 + 1e-9)
        self.assertEqual(len(grid), spec.n_translations)

    def test_z_axis_is_searched_more_narrowly_by_default(self):
        """竖直方向默认只搜横向的一半（先验，不是 URDF 约束 —— 见模块 docstring）。"""
        grid, spec = build_translation_grid(np.zeros(3), r_max=0.8, spacing=0.05)
        self.assertAlmostEqual(spec.z_radius, 0.4)
        self.assertLess(spec.shape[2], spec.shape[0])
        self.assertLessEqual(np.abs(grid[:, 2]).max(), 0.4 + 1e-9)

    def test_z_radius_can_be_opened_up(self):
        _, spec = build_translation_grid(np.zeros(3), r_max=0.8, spacing=0.05, z_radius=0.8)
        self.assertEqual(spec.shape[2], spec.shape[0])

    def test_radius_smaller_than_spacing_degenerates_to_one_layer(self):
        """``z_radius=0`` 是合法输入：只搜质心所在的那一层高度。"""
        grid, spec = build_translation_grid(np.zeros(3), r_max=0.3, spacing=0.1, z_radius=0.0)
        self.assertEqual(spec.shape[2], 1)
        self.assertTrue((grid[:, 2] == 0).all())

    def test_rejects_bad_params(self):
        for kw in ({"r_max": 0.0}, {"r_max": -1.0}, {"spacing": 0.0},
                   {"z_radius": -0.1}, {"z_spacing": 0.0}):
            with self.assertRaises(ValueError):
                build_translation_grid(np.zeros(3), **{"r_max": 0.5, "spacing": 0.1, **kw})
        with self.assertRaises(ValueError):
            build_translation_grid(np.zeros(2), r_max=0.5, spacing=0.1)


# ── 求解 ───────────────────────────────────────────────────────────────────────

def _fake_traj(seed: int = 0, T: int = 40) -> np.ndarray:
    """一段双手轨迹：两只手各绕一个偏心圆走，人为拉开左右不对称。"""
    rng = np.random.default_rng(seed)
    th = np.linspace(0, 2 * np.pi, T, endpoint=False)
    left  = np.stack([0.25 * np.cos(th) - 0.2, 0.25 * np.sin(th) + 0.3, 0.10 * np.cos(2 * th)], 1)
    right = np.stack([0.30 * np.cos(th) + 0.2, 0.30 * np.sin(th) + 0.3, 0.12 * np.sin(2 * th)], 1)
    traj = np.stack([left, right], axis=1) + rng.normal(0, 0.002, (T, 2, 3))
    return traj


class _ShellScorer:
    """假 IK：目标点落在以底座为心的球壳 ``[inner, outer]`` 内就算"可行"。

    为什么用球壳而不是随手编个函数：它必须满足 ``score ≤ reach_bound``（否则测
    "剪枝可采纳"就没有意义了）。取 ``outer < reach_frac * r_max`` 就天然满足，
    同时保留了真实 IK 的两个关键性质 —— 太远解不出来、太近也解不出来。
    """

    def __init__(self, kf_pts, inner=0.10, outer=0.45):
        self.kf_pts, self.inner, self.outer = kf_pts, inner, outer
        self.n_calls = 0

    def __call__(self, R_batch, t_batch):
        self.n_calls += 1
        d = np.linalg.norm(self.kf_pts[None, :, :] - t_batch[:, None, :], axis=2)
        return ((d >= self.inner) & (d <= self.outer)).mean(axis=1)


def _kf_points(traj):
    kf = select_extremal_keyframes(traj)
    return traj[kf.indices].reshape(-1, 3)


class TestSolveRootPoseGrid(unittest.TestCase):

    def setUp(self):
        self.traj = _fake_traj()
        self.kw = dict(r_max=0.8, spacing=0.1, log=lambda *_: None)

    def test_output_contract_is_a_single_static_pose(self):
        """输出是**一个常量位姿**，不是逐帧序列 —— 这是这条支路的定义特征。"""
        sol = solve_root_pose_grid(self.traj, _ShellScorer(_kf_points(self.traj)), **self.kw)
        self.assertEqual(sol.mode, "static")
        self.assertEqual(sol.t.shape, (3,))
        self.assertEqual(sol.R.shape, (3, 3))
        np.testing.assert_allclose(sol.R, np.eye(3))          # 默认只搜平移
        self.assertTrue(0.0 <= sol.ik_rate <= 1.0)
        self.assertEqual(sol.keyframes.source, "convex_hull")

    def test_pruned_search_returns_exactly_the_brute_force_answer(self):
        """上界剪枝必须**可采纳**：和 ``exhaustive=True`` 逐位相同。

        剪枝依据是"目标点超出臂展 → 必然不可行"，所以上界 ≥ 真实可行率，被剪掉的
        候选不可能是最优解。这条一旦破掉，所有跑出来的数字都不再是 argmax，
        而单看结果是发现不了的（照样返回一个位姿、照样有个可行率）。
        """
        s_fast = _ShellScorer(_kf_points(self.traj))
        s_full = _ShellScorer(_kf_points(self.traj))
        fast = solve_root_pose_grid(self.traj, s_fast, **self.kw)
        full = solve_root_pose_grid(self.traj, s_full, exhaustive=True, **self.kw)

        self.assertAlmostEqual(fast.ik_rate, full.ik_rate, places=12)
        np.testing.assert_allclose(fast.t, full.t)
        self.assertLess(fast.n_scored, full.n_scored)          # 确实剪掉了东西
        self.assertEqual(full.n_scored, full.n_candidates)

    def test_it_finds_the_planted_optimum(self):
        """把"唯一最优底座"种在网格上，看能不能捡回来。

        这里把球壳收窄到只有种下的那个格点能覆盖全部关键帧，argmax 因此唯一。
        """
        kf_pts = _kf_points(self.traj)
        planted = kf_pts.mean(axis=0)                          # 关键帧的几何中心
        r = np.linalg.norm(kf_pts - planted, axis=1).max()
        scorer = _ShellScorer(kf_pts, inner=0.0, outer=r + 1e-6)

        sol = solve_root_pose_grid(self.traj, scorer, r_max=0.8, spacing=0.05,
                                   log=lambda *_: None)
        self.assertAlmostEqual(sol.ik_rate, 1.0, places=12)
        self.assertLess(np.linalg.norm(sol.t - planted), 0.05 * np.sqrt(3))

    def test_constant_scores_raise_instead_of_returning_a_plausible_pose(self):
        """所有候选同分 → 炸。这就是 ``workspace_center`` 那个坑的探测器。

        真实场景：``score_fn`` 里给 ``cam_to_root_targets`` 传了 ``workspace_center``，
        位置被 ``pos - pos.mean(0) + center`` 重新居中，候选 ``t`` 被完全抹掉。
        搜索退化成空操作，但结果长得一点问题都没有 —— 和
        ``uniform_filter1d(size=1)`` 那次一模一样。
        """
        def blind(R_batch, t_batch):
            return np.full(len(t_batch), 0.73)                 # 完全无视 t

        with self.assertRaises(RuntimeError) as cm:
            solve_root_pose_grid(self.traj, blind, exhaustive=True, **self.kw)
        self.assertIn("workspace_center", str(cm.exception))

    def test_orientation_candidates_are_searched_too(self):
        """给了 ``R_candidates`` 就连朝向一起搜，并记下选中的是第几个。"""
        kf_pts = _kf_points(self.traj)

        class RSensitive(_ShellScorer):
            """只认第 1 个朝向（下标 1），别的朝向一律 0 分。"""
            def __call__(self, R_batch, t_batch):
                base = super().__call__(R_batch, t_batch)
                yaw90 = np.array([[0., -1, 0], [1, 0, 0], [0, 0, 1]])
                is_target = np.isclose(R_batch, yaw90).all(axis=(1, 2))
                return base * is_target

        yaw90 = np.array([[0., -1, 0], [1, 0, 0], [0, 0, 1]])
        sol = solve_root_pose_grid(self.traj, RSensitive(kf_pts),
                                   R_candidates=[np.eye(3), yaw90], **self.kw)
        self.assertEqual(sol.best_R_index, 1)
        np.testing.assert_allclose(sol.R, yaw90)
        self.assertEqual(sol.n_candidates, 2 * sol.grid.n_translations)

    def test_result_is_deterministic(self):
        """同一份输入跑两次给完全一样的解（并列时靠稳定排序打破）。

        这是它和生成模型那条路最大的行为差异：那边要 best-of-N 采样，这边一次到位。
        """
        a = solve_root_pose_grid(self.traj, _ShellScorer(_kf_points(self.traj)), **self.kw)
        b = solve_root_pose_grid(self.traj, _ShellScorer(_kf_points(self.traj)), **self.kw)
        np.testing.assert_allclose(a.t, b.t)
        self.assertEqual(a.ik_rate, b.ik_rate)
        self.assertEqual(a.n_scored, b.n_scored)

    def test_chunking_does_not_change_the_answer(self):
        ref = solve_root_pose_grid(self.traj, _ShellScorer(_kf_points(self.traj)), **self.kw)
        for chunk in (1, 7, 100000):
            sol = solve_root_pose_grid(self.traj, _ShellScorer(_kf_points(self.traj)),
                                       chunk=chunk, **self.kw)
            np.testing.assert_allclose(sol.t, ref.t)
            self.assertAlmostEqual(sol.ik_rate, ref.ik_rate, places=12)

    # ── 平局：公式 (3) 的 argmax 是个集合，不是一个点 ──────────────────────────

    def _plateau_scorer(self, lo, hi):
        """一个"高原"打分函数：t 落在盒 ``[lo, hi]`` 里就满分，外面 0 分。

        真实数据上这个高原是自然出现的（K 只覆盖位置极值，容易的片段上成千上万个
        候选都是 100%），所以这不是人造病例，是必须定义清楚的常态。

        和 :class:`_ShellScorer` 一样，这里也必须守住 ``score ≤ reach_bound``：
        上界是"目标点在臂展内的 (帧,手) 占比"，一个盒内候选如果有关键帧超出臂展，
        它的上界就 < 1，给它满分等于伪造一个上界扛不住的分数 —— 那时剪枝被判失效
        不是代码的错，是打分函数违约。所以盒内分数再乘一次同一条可达判据。
        """
        kf_pts = _kf_points(self.traj)
        r_max = self.kw["r_max"]

        def score(R_batch, t_batch):
            inside = np.all((t_batch >= lo) & (t_batch <= hi), axis=1).astype(float)
            d = np.linalg.norm(kf_pts[None, :, :] - t_batch[:, None, :], axis=2)
            return np.minimum(inside, (d <= r_max).mean(axis=1))
        return score

    def test_it_counts_how_many_candidates_tie_at_the_optimum(self):
        lo, hi = np.array([-0.1, -0.1, -0.1]), np.array([0.1, 0.1, 0.1])
        sol = solve_root_pose_grid(self.traj, self._plateau_scorer(lo, hi),
                                   tie_break="first", **self.kw)
        cand_t = sol.extras["candidate_t"]
        d = np.linalg.norm(_kf_points(self.traj)[None] - cand_t[:, None, :], axis=2)
        tied = (np.all((cand_t >= lo) & (cand_t <= hi), axis=1)
                & (d <= self.kw["r_max"]).all(axis=1))         # 盒内 ∩ 全部关键帧可达
        self.assertEqual(sol.n_tied, int(tied.sum()))
        self.assertGreater(sol.n_tied, 1)           # 不然这个测试没在测东西

    def test_plateau_tie_break_lands_in_the_middle_of_the_plateau(self):
        """默认 tie_break 应当挑高原**内部**的点，而不是排序里最先撞上的那个。

        理由不是美观：K 是按位置选的，没进 K 的帧（尤其腕部朝向别扭那些）在边界点
        上更容易掉出可行域。实测过一次 —— 两个解在 K 上都 100%，全轨迹一个 100%
        一个 66.7%。
        """
        lo, hi = np.array([-0.15, -0.15, -0.15]), np.array([0.25, 0.25, 0.25])
        centre = (lo + hi) / 2
        plateau = solve_root_pose_grid(self.traj, self._plateau_scorer(lo, hi),
                                       tie_break="plateau", **self.kw)
        first = solve_root_pose_grid(self.traj, self._plateau_scorer(lo, hi),
                                     tie_break="first", **self.kw)
        self.assertEqual(plateau.ik_rate, first.ik_rate)      # 目标函数值一样 —— 都是精确 argmax
        self.assertLess(np.linalg.norm(plateau.t - centre),
                        np.linalg.norm(first.t - centre))
        self.assertLess(np.linalg.norm(plateau.t - centre), self.kw["spacing"])

    def test_pruning_keeps_the_tied_candidates_so_the_count_is_right(self):
        """剪枝门槛是 ``>=`` 而不是 ``>``：同分候选不能被剪掉，否则 n_tied 会少数、
        高原内部点也选不到。剪枝版和穷举版的 n_tied 必须一致。"""
        lo, hi = np.array([-0.15, -0.15, -0.15]), np.array([0.25, 0.25, 0.25])
        fast = solve_root_pose_grid(self.traj, self._plateau_scorer(lo, hi), **self.kw)
        full = solve_root_pose_grid(self.traj, self._plateau_scorer(lo, hi),
                                    exhaustive=True, **self.kw)
        self.assertEqual(fast.n_tied, full.n_tied)
        np.testing.assert_allclose(fast.t, full.t)

    def test_first_tie_break_prunes_harder_and_says_so(self):
        """``tie_break="first"`` 用 ``>`` 剪枝：饱和时少打很多分，代价是 n_tied 变下界。

        这条钉的是那笔账本身 —— 要准确的 argmax 集合大小就得几乎穷举，要快就只能
        拿到下界。哪天有人把两条路的门槛改成一样的，这里会红。
        """
        lo, hi = np.array([-0.15, -0.15, -0.15]), np.array([0.25, 0.25, 0.25])
        plateau = solve_root_pose_grid(self.traj, self._plateau_scorer(lo, hi),
                                       tie_break="plateau", **self.kw)
        first = solve_root_pose_grid(self.traj, self._plateau_scorer(lo, hi),
                                     tie_break="first", **self.kw)
        self.assertEqual(plateau.ik_rate, first.ik_rate)       # 目标函数值一样
        self.assertLess(first.n_scored, plateau.n_scored)      # 真的少打了
        self.assertLessEqual(first.n_tied, plateau.n_tied)     # 下界 ≤ 准确值

    def test_both_tie_breaks_agree_when_the_optimum_is_unique(self):
        """最优唯一时两条路必须给出同一个位姿 —— 剪枝强度不许改变答案。"""
        kf_pts = _kf_points(self.traj)
        planted = kf_pts.mean(axis=0)
        r = np.linalg.norm(kf_pts - planted, axis=1).max()
        kw = dict(self.kw, spacing=0.05)
        a = solve_root_pose_grid(self.traj, _ShellScorer(kf_pts, 0.0, r + 1e-6),
                                 tie_break="plateau", **kw)
        b = solve_root_pose_grid(self.traj, _ShellScorer(kf_pts, 0.0, r + 1e-6),
                                 tie_break="first", **kw)
        np.testing.assert_allclose(a.t, b.t)
        self.assertAlmostEqual(a.ik_rate, b.ik_rate, places=12)

    def test_saturated_scores_are_not_mistaken_for_the_workspace_center_bug(self):
        """可行率饱和（打过分的候选全是满分）是**正常**情况，不许误报成空操作。

        判据是"分数不变而可达上界在变"：空操作时上界照样随 t 变化，饱和时打过分的
        候选本来就是上界最高那一批。这个测试盯的就是这条判据没被写成"只要同分就炸"。
        """
        sol = solve_root_pose_grid(self.traj, lambda R, t: np.ones(len(t)),
                                   chunk=8, **self.kw)
        self.assertEqual(sol.ik_rate, 1.0)
        self.assertGreater(sol.n_tied, 1)

    def test_score_fn_shape_mismatch_is_caught(self):
        with self.assertRaises(ValueError):
            solve_root_pose_grid(self.traj, lambda R, t: np.zeros(len(t) + 1), **self.kw)

    def test_rejects_bad_params(self):
        scorer = _ShellScorer(_kf_points(self.traj))
        for kw in ({"reach_frac": 0.0}, {"reach_frac": 1.5}, {"min_dist": -0.1},
                   {"chunk": 0}, {"tie_break": "middle"}):
            with self.assertRaises(ValueError):
                solve_root_pose_grid(self.traj, scorer, **{**self.kw, **kw})

    def test_summary_mentions_the_things_you_need_to_read_a_run(self):
        sol = solve_root_pose_grid(self.traj, _ShellScorer(_kf_points(self.traj)), **self.kw)
        text = sol.summary()
        for token in ("ik_rate", "convex_hull", "候选"):
            self.assertIn(token, text)


class TestMakeKeyframeScorer(unittest.TestCase):
    """胶水层：批处理约定和聚合口径。两个依赖都用纯 numpy 假货，不碰上游。"""

    def setUp(self):
        self.K = 4
        self.kf_l = np.concatenate([np.linspace(0, 1, self.K)[:, None] * np.ones((1, 3)),
                                    np.tile([1., 0, 0, 0], (self.K, 1))], axis=1)
        self.kf_r = self.kf_l + 0.5
        self.seen = []

        def to_root(left, right, R_pf, t_pf):
            """假 cam_to_root_targets：只做 p − t（够了，本层不解释坐标语义）。"""
            self.seen.append(("to_root", len(left)))
            return (left[:, :3] - t_pf, left[:, 3:],
                    right[:, :3] - t_pf, right[:, 3:])

        self.to_root = to_root

    def _scorer(self, thresh=0.6):
        def converged(side, pos, quat):
            self.seen.append((side, len(pos)))
            return np.linalg.norm(pos, axis=1) < thresh
        return make_keyframe_scorer(self.kf_l, self.kf_r, self.to_root, converged)

    def test_it_batches_all_candidates_into_one_ik_call(self):
        """B 个候选 × |K| 帧 → **一次** IK 调用，不是 B 次。

        逐候选串行调 IK 会慢两个数量级，网格搜索就不可能跑完。所以"只调一次"
        是这层胶水存在的理由，得钉住。
        """
        score = self._scorer()
        B = 5
        out = score(np.tile(np.eye(3), (B, 1, 1)), np.zeros((B, 3)))

        self.assertEqual(out.shape, (B,))
        ik_calls = [s for s in self.seen if s[0] in ("left", "right")]
        self.assertEqual(len(ik_calls), 2)                       # 左右各一次
        self.assertEqual({n for _, n in ik_calls}, {B * self.K})

    def test_aggregation_matches_upstream_select_best_anchor(self):
        """口径 = ``(左臂可行率 + 右臂可行率) / 2``，和上游聚类打分同一个式子。

        故意造成左右不对称（右手整体偏 +0.5，全都超阈值），这样"两臂平均"和
        "两臂都要收敛"这两种口径给出的数不同，测试才区分得开。
        """
        score = self._scorer(thresh=0.6)
        out = score(np.eye(3)[None], np.zeros((1, 3)))
        # 左手 4 帧 ‖p‖ = 0, .577, 1.155, 1.732 → 2/4 < 0.6 → 50%
        # 右手整体 +0.5（三轴）→ 最近的也有 0.866 → 0/4 → 0%
        # 两臂平均 = 25%。若换成"两臂都收敛才算"，答案会是 0%，所以这个例子能区分口径。
        self.assertAlmostEqual(float(out[0]), 0.25, places=12)

    def test_candidate_translation_actually_reaches_the_targets(self):
        """不同候选 t 必须给出不同分 —— 这层要是把 t 丢了，上面那个 no-op 断言
        就再也炸不出来了（因为分数会变成常量）。"""
        score = self._scorer(thresh=0.6)
        near = score(np.eye(3)[None], np.zeros((1, 3)))
        far  = score(np.eye(3)[None], np.full((1, 3), 5.0))
        self.assertNotAlmostEqual(float(near[0]), float(far[0]))
        self.assertEqual(float(far[0]), 0.0)

    def test_bad_converged_fn_shape_is_caught(self):
        bad = make_keyframe_scorer(self.kf_l, self.kf_r, self.to_root,
                                   lambda side, p, q: np.zeros(len(p) + 1, bool))
        with self.assertRaises(ValueError):
            bad(np.eye(3)[None], np.zeros((1, 3)))

    def test_rejects_empty_or_mismatched_keyframes(self):
        with self.assertRaises(ValueError):
            make_keyframe_scorer(self.kf_l[:0], self.kf_r[:0], self.to_root,
                                 lambda *a: np.zeros(0, bool))
        with self.assertRaises(ValueError):
            make_keyframe_scorer(self.kf_l, self.kf_r[:2], self.to_root,
                                 lambda *a: np.zeros(0, bool))


class TestGravityYawCandidates(unittest.TestCase):
    """朝向候选：竖直轴必须是重力给的，偏航必须真的转满一圈。"""

    def setUp(self):
        # 相机系里 "上" = -y（OpenCV 式相机：+y 朝下），手在 +z 前方
        self.up = np.array([0.0, -1.0, 0.0])
        T = 20
        self.left  = np.zeros((T, 7)); self.right = np.zeros((T, 7))
        self.left[:, :3]  = [0.2, -0.3, 1.0]        # 左手在 +x 侧
        self.right[:, :3] = [-0.2, -0.3, 1.0]

    def test_every_candidate_is_a_rotation_with_gravity_as_its_third_column(self):
        for R in gravity_yaw_candidates(self.up, self.left, self.right, n_yaw=7):
            np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-9)
            self.assertAlmostEqual(float(np.linalg.det(R)), 1.0, places=9)
            # 第三列 = 躯干"上"轴，必须就是重力方向，一个候选都不许偏
            np.testing.assert_allclose(R[:, 2], self.up, atol=1e-9)

    def test_zero_point_puts_the_left_hand_on_the_torso_left(self):
        """n_yaw=1 时那唯一一个候选，左手在躯干系里的 y 必须为正。"""
        R = gravity_yaw_candidates(self.up, self.left, self.right, n_yaw=1)[0]
        lm = R.T @ self.left[0, :3]
        rm = R.T @ self.right[0, :3]
        self.assertGreater(lm[1], rm[1])
        self.assertGreater(lm[1] - rm[1], 0.3)      # 手间距 0.4 m 基本全落在 y 上

    def test_yaws_cover_the_full_circle(self):
        """n_yaw 个候选的前向轴应当均匀铺满 360°，而不是挤在一小段里。"""
        Rs = gravity_yaw_candidates(self.up, self.left, self.right, n_yaw=12)
        fwd = np.stack([R[:, 0] for R in Rs])
        ang = np.sort(np.degrees(np.arctan2(fwd @ Rs[0][:, 1], fwd @ Rs[0][:, 0])))
        gaps = np.diff(np.concatenate([ang, ang[:1] + 360]))
        np.testing.assert_allclose(gaps, 30.0, atol=1e-6)

    def test_it_survives_the_two_hands_coinciding(self):
        """双手轨迹重合 → 零点没有物理含义，但仍须返回合法旋转（全圆枚举照样能搜到）。"""
        same = self.left.copy()
        Rs = gravity_yaw_candidates(self.up, same, same, n_yaw=4)
        self.assertEqual(len(Rs), 4)
        for R in Rs:
            np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-9)

    def test_bad_input_raises(self):
        with self.assertRaises(ValueError):
            gravity_yaw_candidates(self.up, self.left, self.right, n_yaw=0)
        with self.assertRaises(ValueError):
            gravity_yaw_candidates([0, 0], self.left, self.right)
        with self.assertRaises(ValueError):
            gravity_yaw_candidates([0, 0, 0], self.left, self.right)


class TestEstimateReach(unittest.TestCase):
    """臂展测量：用一个解析可算的假机器人，答案必须对得上。"""

    @staticmethod
    def _two_link(q):
        """平面两连杆，长 0.6 + 0.4。‖p‖ 在 q1=0 时最大（=1.0），完全折叠时最小 0.2。"""
        q = np.asarray(q, dtype=float)
        a = q[:, 0]
        b = q[:, 0] + q[:, 1]
        x = 0.6 * np.cos(a) + 0.4 * np.cos(b)
        y = 0.6 * np.sin(a) + 0.4 * np.sin(b)
        return np.stack([x, y, np.zeros_like(x)], axis=1)

    def test_it_finds_the_analytic_reach(self):
        lim = np.array([[-np.pi, np.pi], [-np.pi, np.pi]])
        r = estimate_reach(self._two_link, lim, n_random=20_000, seed=0)
        self.assertAlmostEqual(r, 1.0, places=3)

    def test_it_beats_corner_only_sampling(self):
        """最远构型落在限位内部时，只采角点会低估 —— 这正是 M7 上实测到的情形
        （角点 0.7176 m vs 随机 1.0067 m），所以随机采样不是可选项。"""
        lim = np.array([[-np.pi, np.pi], [-np.pi, np.pi]])
        corner_only = float(np.linalg.norm(self._two_link(
            np.array(np.meshgrid(*lim, indexing="ij")).reshape(2, -1).T), axis=1).max())
        self.assertLess(corner_only, 0.9)                       # 角点全是折叠构型
        self.assertGreater(estimate_reach(self._two_link, lim, n_random=20_000), 0.99)

    def test_batching_does_not_change_the_answer(self):
        lim = np.array([[-1.0, 1.0], [-2.0, 0.5]])
        a = estimate_reach(self._two_link, lim, n_random=5000, seed=3, batch=64)
        b = estimate_reach(self._two_link, lim, n_random=5000, seed=3, batch=100000)
        self.assertEqual(a, b)

    def test_seed_makes_it_reproducible(self):
        lim = np.array([[-1.0, 1.0], [-2.0, 0.5]])
        self.assertEqual(estimate_reach(self._two_link, lim, n_random=3000, seed=7),
                         estimate_reach(self._two_link, lim, n_random=3000, seed=7))

    def test_bad_input_raises(self):
        with self.assertRaises(ValueError):
            estimate_reach(self._two_link, np.zeros(3))
        with self.assertRaises(ValueError):
            estimate_reach(self._two_link, np.array([[1.0, -1.0]]))
        with self.assertRaises(ValueError):
            estimate_reach(lambda q: np.zeros((len(q), 2)), np.array([[0.0, 1.0]]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
