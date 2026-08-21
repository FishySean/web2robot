"""模块二（``--action_refine``）的单测 —— EgoEngine §3.2.2 的判决逻辑。

分六组，每组盯一类会真出错的事：

1. **误差公式**（``TestStepErrors``）—— e_t 的两个分量各自对不对、λ 起不起作用、
   坏帧变 NaN 而不是 0。误差算错，后面全套判决都是错的，还看不出来。
2. **位姿代数**（``TestPoseAlgebra``）—— 逆、复合、共轭换系。这里最容易左右乘写反，
   而写反之后误差**看起来还挺合理**（数量级不变），只有拿构造好的例子才抓得住。
3. **刚连预测**（``TestAttach``）—— 手没偏 → 物体不动；只有抓着的帧才受影响；
   两只手都抓时取偏差大的那只。
4. **切块**（``TestSplitBlocks``）—— 覆盖完整、余块不丢。丢帧是静默的。
5. **判决**（``TestPlan``）—— 早停位置、unknown 不当成 ok、两块联合窗口真的会把
   好块拖下来、``none`` 下 escalate 照标但 mode 不动。
6. **接线**（``TestModesAndRun``）—— mpc/rl 明确报错不静默降级；落盘的三个文件
   键名齐全；CLI 缺文件时报错而不是拿参考当执行。
"""
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from web2robot.refine import (
    ErrorWeights, H_DEFAULT, MODES, RefineConfig, conjugate_delta, mpc_solve,
    plan_blocks, pose_compose, pose_delta, pose_inverse, predict_object_poses,
    replay_solve, reward, rl_solve, score_block, split_blocks, step_errors,
)
from web2robot.refine.attach import grasp_hands
from web2robot.refine.run import refine_run
from web2robot.twin.object_pose import (
    CameraIntrinsics, ObjectPoseSet, ObjectTrack, mats_to_posquat, save_object_poses,
)

IDENT = np.array([0.0, 0, 0, 1, 0, 0, 0])


def quat_z(deg):
    a = np.deg2rad(deg) / 2
    return np.array([np.cos(a), 0.0, 0.0, np.sin(a)])


def pose(xyz, q=(1.0, 0, 0, 0)):
    return np.concatenate([np.asarray(xyz, float), np.asarray(q, float)])


def straight_line(T, step=0.001):
    """一段沿 x 匀速平移、姿态不变的物体轨迹 (T,7)。"""
    p = np.tile(IDENT, (T, 1))
    p[:, 0] = np.arange(T) * step
    return p


def hands(T, ref=None, ach=None):
    """(T,2,7) 两只手，默认两只都在原点、参考=实际（零偏差）。"""
    base = np.tile(IDENT, (T, 2, 1))
    return (base.copy() if ref is None else ref), (base.copy() if ach is None else ach)


class TestStepErrors(unittest.TestCase):
    def test_translation_only(self):
        ref = np.tile(IDENT, (3, 1))
        ach = ref.copy()
        ach[:, 0] = [0.0, 0.1, 0.2]
        out = step_errors(ref, ach)
        np.testing.assert_allclose(out["ep"], [0.0, 0.1, 0.2], atol=1e-12)
        np.testing.assert_allclose(out["eR"], 0.0, atol=1e-7)
        np.testing.assert_allclose(out["e"], [0.0, 0.1, 0.2], atol=1e-7)

    def test_rotation_only_is_geodesic_in_radians(self):
        ref = np.tile(IDENT, (2, 1))
        ach = ref.copy()
        ach[0, 3:] = quat_z(30)
        ach[1, 3:] = quat_z(150)
        out = step_errors(ref, ach)
        np.testing.assert_allclose(out["eR"], np.deg2rad([30, 150]), atol=1e-9)
        np.testing.assert_allclose(out["ep"], 0.0, atol=1e-12)

    def test_quat_sign_flip_is_not_an_error(self):
        """q 和 −q 是同一个旋转 —— 不取绝对值的话这里会给出 2π−θ。"""
        ref = np.tile(IDENT, (1, 1))
        ach = ref.copy()
        ach[0, 3:] = -quat_z(20)
        self.assertAlmostEqual(float(step_errors(ref, ach)["eR"][0]),
                               np.deg2rad(20), places=9)

    def test_geodesic_is_capped_at_pi(self):
        ref = np.tile(IDENT, (1, 1))
        ach = ref.copy()
        ach[0, 3:] = quat_z(359)
        self.assertLessEqual(float(step_errors(ref, ach)["eR"][0]), np.pi + 1e-9)

    def test_lambdas_weight_the_two_terms(self):
        ref = np.tile(IDENT, (1, 1))
        ach = ref.copy()
        ach[0, 0] = 0.3
        ach[0, 3:] = quat_z(np.rad2deg(0.4))
        e1 = step_errors(ref, ach, weights=ErrorWeights(1.0, 1.0))["e"][0]
        e2 = step_errors(ref, ach, weights=ErrorWeights(1.0, 0.0))["e"][0]
        self.assertAlmostEqual(float(e1), np.hypot(0.3, 0.4), places=6)
        self.assertAlmostEqual(float(e2), 0.3, places=6)

    def test_invalid_frames_become_nan_not_zero(self):
        """用 0 填无效帧会被下游读成"这帧完美"，是最坏的一种坏。"""
        ref = np.tile(IDENT, (4, 1))
        ach = ref.copy()
        ach[1, 0] = 0.5
        ach[2, :3] = np.nan
        out = step_errors(ref, ach, valid=np.array([True, False, True, True]))
        self.assertTrue(np.isnan(out["e"][1]))   # valid=False
        self.assertTrue(np.isnan(out["e"][2]))   # NaN 位姿
        self.assertEqual(float(out["e"][0]), 0.0)

    def test_negative_lambda_rejected(self):
        with self.assertRaises(ValueError):
            ErrorWeights(-1.0, 1.0)

    def test_shape_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            step_errors(np.tile(IDENT, (3, 1)), np.tile(IDENT, (4, 1)))

    def test_reward_is_C_minus_e(self):
        np.testing.assert_allclose(reward(np.array([0.0, 0.2]), 1.0), [1.0, 0.8])


class TestPoseAlgebra(unittest.TestCase):
    """位姿代数。容差 1e-6 不是 1e-9：位姿一路是 float32（和 ``root_frames.npz``、
    ``object_poses.npz`` 落盘的精度一致），链式换算下来相对误差就在 1e-7 量级。
    关心的误差是厘米级，float32 够用；把容差写成 1e-9 只会测出浮点精度。"""

    def test_inverse_roundtrip(self):
        p = np.array([pose([0.1, 0.2, 0.3], quat_z(37))])
        back = pose_compose(p, pose_inverse(p))
        np.testing.assert_allclose(back[0, :3], 0, atol=1e-6)
        self.assertAlmostEqual(abs(float(back[0, 3])), 1.0, places=6)

    def test_compose_order_is_A_then_B_as_matrices(self):
        """``pose_compose(a,b)`` 必须等于矩阵 ``A @ B``，不是 ``B @ A``。"""
        a = np.array([pose([1.0, 0, 0], quat_z(90))])
        b = np.array([pose([0.0, 1.0, 0])])
        got = pose_compose(a, b)
        # A@B：先把 b 的平移按 a 的旋转转过去，再加 a 的平移 → (1,0,0)+R90·(0,1,0)=(0,0,0)
        np.testing.assert_allclose(got[0, :3], [0.0, 0.0, 0.0], atol=1e-6)

    def test_delta_is_left_multiplied(self):
        """D = T_ach ∘ T_ref⁻¹。右乘写法在这个例子上会给出不同的平移。"""
        ref = np.array([pose([1.0, 0, 0])])
        ach = np.array([pose([1.0, 0, 0], quat_z(90))])
        d = pose_delta(ref, ach)
        # 左乘：绕原点转 90° 再把 ref 的位置搬回去 → 平移 (1,-1,0)
        np.testing.assert_allclose(d[0, :3], [1.0, -1.0, 0.0], atol=1e-6)

    def test_delta_applied_to_ref_recovers_ach(self):
        ref = np.array([pose([0.2, -0.1, 0.4], quat_z(20))])
        ach = np.array([pose([0.25, -0.05, 0.4], quat_z(35))])
        got = pose_compose(pose_delta(ref, ach), ref)
        np.testing.assert_allclose(got[0, :3], ach[0, :3], atol=1e-6)
        self.assertAlmostEqual(
            float(step_errors(got, ach)["eR"][0]), 0.0, places=6)

    def test_conjugate_commutes_with_the_frame_change(self):
        """真正的判据：``D_cam ∘ T = T ∘ D_root``（先偏后换系 = 先换系后偏）。

        "范数不变"是错的判据 —— 共轭后的平移是 ``R·d + (I − R_conj)·t``，绕一个离
        原点远的点转一点点，换系之后看就是一大段平移。这条交换律才抓得住 R/Rᵀ 写反。
        """
        d = np.array([pose([0.03, -0.01, 0.02], quat_z(11))])
        ang = 0.7
        R = np.array([[[np.cos(ang), -np.sin(ang), 0],
                       [np.sin(ang), np.cos(ang), 0], [0, 0, 1.0]]])
        t = np.array([[0.4, -1.2, 0.9]])
        T = np.concatenate([t, mats_to_posquat(
            np.concatenate([np.concatenate([R, t[:, :, None]], 2),
                            np.array([[[0, 0, 0, 1.0]]])], 1))[:, 3:]], 1)
        lhs = pose_compose(conjugate_delta(d, R, t), T)
        rhs = pose_compose(T, d)
        np.testing.assert_allclose(lhs[:, :3], rhs[:, :3], atol=1e-6)
        self.assertAlmostEqual(float(step_errors(lhs, rhs)["eR"][0]), 0.0, places=6)

    def test_conjugate_preserves_the_rotation_angle(self):
        d = np.array([pose([0.03, -0.01, 0.02], quat_z(11))])
        ang = 0.7
        R = np.array([[[np.cos(ang), -np.sin(ang), 0],
                       [np.sin(ang), np.cos(ang), 0], [0, 0, 1.0]]])
        c = conjugate_delta(d, R, np.array([[0.4, -1.2, 0.9]]))
        self.assertAlmostEqual(float(step_errors(np.array([IDENT]), c)["eR"][0]),
                               np.deg2rad(11), places=6)

    def test_conjugate_by_identity_is_a_noop(self):
        d = np.array([pose([0.05, 0.0, 0.0], quat_z(15))])
        c = conjugate_delta(d, np.eye(3)[None], np.zeros((1, 3)))
        np.testing.assert_allclose(c, d, atol=1e-6)

    def test_pure_translation_delta_ignores_t(self):
        """纯平移偏差才轮到 ``t`` 掉出去 —— 带旋转时它是该出现的，见上面那条。"""
        d = np.array([pose([0.05, 0.0, 0.0])])
        c1 = conjugate_delta(d, np.eye(3)[None], np.zeros((1, 3)))
        c2 = conjugate_delta(d, np.eye(3)[None], np.array([[10.0, -3.0, 7.0]]))
        np.testing.assert_allclose(c1, c2, atol=1e-6)


class TestAttach(unittest.TestCase):
    def test_zero_hand_error_means_object_unmoved(self):
        obj = straight_line(5)
        hr, ha = hands(5)
        np.testing.assert_allclose(predict_object_poses(obj, hr, ha), obj, atol=1e-9)
        self.assertTrue(np.all(step_errors(obj, predict_object_poses(obj, hr, ha))["e"]
                               < 1e-9))

    def test_hand_offset_moves_the_object_by_the_same_amount(self):
        obj = straight_line(4)
        hr, ha = hands(4)
        ha[:, 0, 0] += 0.07          # 左手整体偏 7cm
        got = predict_object_poses(obj, hr, ha)
        np.testing.assert_allclose(got[:, 0] - obj[:, 0], 0.07, atol=1e-9)

    def test_only_grasped_frames_are_affected(self):
        obj = straight_line(4)
        hr, ha = hands(4)
        ha[:, 0, 0] += 0.07
        grasp = np.zeros((4, 2), bool)
        grasp[2:, 0] = True
        got = predict_object_poses(obj, hr, ha, grasp=grasp)
        np.testing.assert_allclose(got[:2], obj[:2], atol=1e-12)
        np.testing.assert_allclose(got[2:, 0] - obj[2:, 0], 0.07, atol=1e-9)

    def test_both_hands_takes_the_larger_deviation(self):
        obj = straight_line(3)
        hr, ha = hands(3)
        ha[:, 0, 0] += 0.02          # 左手偏 2cm
        ha[:, 1, 0] += 0.09          # 右手偏 9cm ← 应该用这只
        got = predict_object_poses(obj, hr, ha, grasp=np.ones((3, 2), bool))
        np.testing.assert_allclose(got[:, 0] - obj[:, 0], 0.09, atol=1e-9)

    def test_rotation_of_the_hand_rotates_the_object(self):
        obj = np.tile(pose([1.0, 0, 0]), (2, 1))
        hr, ha = hands(2)
        ha[:, 0, 3:] = quat_z(90)    # 左手绕原点转 90°
        got = predict_object_poses(obj, hr, ha)
        np.testing.assert_allclose(got[:, :3], [[0, 1.0, 0]] * 2, atol=1e-9)

    def test_grasp_hands_parses_the_state_strings(self):
        g = grasp_hands(["static", "grasped_l", "grasped_r", "grasped_both", "", "moving"])
        np.testing.assert_array_equal(g, [[0, 0], [1, 0], [0, 1], [1, 1], [0, 0], [0, 0]])

    def test_shape_errors_are_loud(self):
        obj = straight_line(3)
        hr, ha = hands(3)
        with self.assertRaises(ValueError):
            predict_object_poses(obj, hr[:2], ha)
        with self.assertRaises(ValueError):
            predict_object_poses(obj, hr, ha, grasp=np.ones((2, 2), bool))


class TestSplitBlocks(unittest.TestCase):
    def test_paper_horizon_is_20(self):
        self.assertEqual(H_DEFAULT, 20)

    def test_exact_multiple(self):
        self.assertEqual(split_blocks(40, 20), [(0, 20), (20, 40)])

    def test_tail_is_kept_by_default(self):
        """69 帧按 H=20 → 20/20/20/9，末尾那 9 帧不能丢（静默丢帧）。"""
        self.assertEqual(split_blocks(69, 20), [(0, 20), (20, 40), (40, 60), (60, 69)])

    def test_tail_can_be_dropped_explicitly(self):
        self.assertEqual(split_blocks(69, 20, keep_tail=False),
                         [(0, 20), (20, 40), (40, 60)])

    def test_coverage_is_complete_and_contiguous(self):
        for T in (1, 7, 20, 21, 69, 257):
            b = split_blocks(T, 20)
            self.assertEqual(b[0][0], 0)
            self.assertEqual(b[-1][1], T)
            for (_, s1), (s2, _) in zip(b, b[1:]):
                self.assertEqual(s1, s2)

    def test_empty_and_bad_args(self):
        self.assertEqual(split_blocks(0, 20), [])
        with self.assertRaises(ValueError):
            split_blocks(10, 0)
        with self.assertRaises(ValueError):
            split_blocks(-1, 20)


class TestPlan(unittest.TestCase):
    def _plan(self, e_per_frame, T=40, **kw):
        """用给定的逐帧误差直接造判决 —— 绕开位姿，只测判决逻辑。"""
        ref = np.tile(IDENT, (T, 1))
        ach = ref.copy()
        ach[:, 0] = e_per_frame          # 纯平移误差 ⇒ e_t 就是这个数
        return plan_blocks(ref, ach, **kw)

    def test_all_good_stays_replay(self):
        plans, s = self._plan(0.001, cfg=RefineConfig(horizon=20))
        self.assertEqual([p.mode for p in plans], ["replay", "replay"])
        self.assertFalse(s["needs_escalation"])
        self.assertEqual(s["status_counts"]["ok"], 2)

    def test_budget_scales_with_block_length(self):
        """余块用自己的长度算预算 —— 否则 9 帧的块几乎不可能被判坏。"""
        cfg = RefineConfig(horizon=20, per_frame_budget=0.05)
        plans, _ = self._plan(0.06, T=29, cfg=cfg)
        self.assertEqual([p.score.budget for p in plans], [1.0, 0.45])
        self.assertEqual([p.score.status for p in plans], ["over", "over"])

    def test_early_termination_frame_is_where_cumsum_crosses(self):
        """每帧 0.06、预算 1.0 → 第 17 帧（0-based 16）累计 1.02 首次越线。"""
        cfg = RefineConfig(horizon=20, per_frame_budget=0.05)
        plans, _ = self._plan(0.06, T=20, cfg=cfg)
        self.assertEqual(plans[0].score.terminated_at, 16)
        self.assertAlmostEqual(plans[0].score.cum, 0.06 * 17, places=6)

    def test_cum_stops_accumulating_after_early_stop(self):
        e = np.full(20, 0.0)
        e[:18] = 0.06
        e[18:] = 5.0                     # 早停之后的帧不该被记账
        plans, _ = self._plan(e, T=20, cfg=RefineConfig(horizon=20))
        self.assertLess(plans[0].score.cum, 1.1)

    def test_only_the_bad_block_escalates(self):
        e = np.full(40, 0.001)
        e[20:] = 0.5
        plans, s = self._plan(e, T=40, cfg=RefineConfig(horizon=20), requested="mpc")
        self.assertEqual([p.score.status for p in plans], ["ok", "over"])
        self.assertEqual([p.mode for p in plans], ["mpc", "mpc"])   # 第 0 块被下一块拖下来
        self.assertTrue(plans[0].blocked_by_next)
        self.assertFalse(plans[1].blocked_by_next)
        self.assertEqual(s["n_escalate"], 2)

    def test_two_chunk_window_only_reaches_one_block_back(self):
        """block0 好、block1 好、block2 坏 → 只有 1 和 2 升级，0 不受影响。"""
        e = np.full(60, 0.001)
        e[40:] = 0.5
        plans, s = self._plan(e, T=60, cfg=RefineConfig(horizon=20), requested="mpc")
        self.assertEqual([p.escalate for p in plans], [False, True, True])
        self.assertEqual(s["n_blocked_by_next"], 1)

    def test_last_block_has_no_next_and_is_judged_alone(self):
        e = np.full(40, 0.001)
        plans, _ = self._plan(e, T=40, cfg=RefineConfig(horizon=20))
        self.assertFalse(plans[-1].blocked_by_next)

    def test_unknown_is_not_ok(self):
        """量不到的块判 unknown 并要求升级 —— 当成 ok 是这条链最容易犯的错。"""
        ref = np.tile(IDENT, (20, 1))
        ach = ref.copy()
        valid = np.zeros(20, bool)
        valid[:5] = True                 # 只有 5/20 可信 < min_valid_frac=0.5
        plans, s = plan_blocks(ref, ach, valid, RefineConfig(horizon=20))
        self.assertEqual(plans[0].score.status, "unknown")
        self.assertFalse(plans[0].score.feasible)
        self.assertTrue(plans[0].escalate)
        self.assertEqual(s["status_counts"]["unknown"], 1)
        self.assertIn("判不了", plans[0].reason)

    def test_enough_valid_frames_can_still_be_ok(self):
        ref = np.tile(IDENT, (20, 1))
        valid = np.zeros(20, bool)
        valid[:15] = True
        plans, _ = plan_blocks(ref, ref.copy(), valid, RefineConfig(horizon=20))
        self.assertEqual(plans[0].score.status, "ok")
        self.assertEqual(plans[0].score.n_valid, 15)

    def test_none_marks_escalation_but_does_not_switch_mode(self):
        e = np.full(20, 0.5)
        plans, s = self._plan(e, T=20, cfg=RefineConfig(horizon=20), requested="none")
        self.assertTrue(plans[0].escalate)
        self.assertEqual(plans[0].mode, "replay")
        self.assertIn("只出判决不升级", plans[0].reason)
        self.assertTrue(s["needs_escalation"])

    def test_mpc_is_the_first_escalation_not_rl(self):
        e = np.full(20, 0.5)
        plans, _ = self._plan(e, T=20, cfg=RefineConfig(horizon=20), requested="rl")
        self.assertEqual(plans[0].mode, "mpc")   # 阶梯是一级一级上的

    def test_summary_records_the_knobs_that_have_no_paper_value(self):
        _, s = self._plan(0.001, cfg=RefineConfig(horizon=20))
        for k in ("horizon", "per_frame_budget", "lam_p", "lam_R"):
            self.assertIn(k, s)

    def test_json_has_no_nan(self):
        """NaN 不是合法 JSON —— 会让下游 json.load 直接炸或者拿到 float('nan')。"""
        ref = np.tile(IDENT, (20, 1))
        plans, _ = plan_blocks(ref, ref.copy(), np.zeros(20, bool),
                              RefineConfig(horizon=20))
        txt = json.dumps([p.as_dict() for p in plans])
        self.assertNotIn("NaN", txt)
        self.assertIsNone(plans[0].as_dict()["e_mean"])

    def test_bad_config_rejected(self):
        for kw in ({"horizon": 0}, {"per_frame_budget": 0.0}, {"min_valid_frac": 1.5}):
            with self.assertRaises(ValueError):
                RefineConfig(**kw)


class TestModesAndRun(unittest.TestCase):
    def test_ladder_order(self):
        self.assertEqual(MODES, ("replay", "mpc", "rl"))

    def test_replay_returns_a_copy(self):
        a = np.arange(30, dtype=float).reshape(10, 3)
        out = replay_solve(a, 2, 5)
        out[0, 0] = 999.0
        self.assertEqual(a[2, 0], 6.0)

    def test_replay_range_checked(self):
        with self.assertRaises(ValueError):
            replay_solve(np.zeros((5, 3)), 0, 9)

    def test_mpc_and_rl_refuse_loudly(self):
        """静默退回 replay 会让下游拿到一份"以为精修过"的数据。"""
        for fn in (mpc_solve, rl_solve):
            with self.assertRaises(NotImplementedError) as cm:
                fn(np.zeros((5, 3)), 0, 5)
            self.assertIn("还没实现", str(cm.exception))

    def _poses(self, T=40, states=None):
        p = straight_line(T)
        st = np.array(states if states is not None else ["grasped_r"] * T)
        tr = ObjectTrack(oid=0, poses=p.astype(np.float32),
                         valid=np.ones(T, bool), state=st)
        return ObjectPoseSet(tracks=[tr], n_frames=T, fps=30.0, clip="unit_test",
                             camera=CameraIntrinsics(500.0, 320.0, 240.0, 640, 480,
                                                     (0.0, -1.0, 0.0)),
                             task_object_id=0)

    def test_run_writes_all_three_files(self):
        T = 40
        poses = self._poses(T)
        hr, ha = hands(T)
        ha[:, 1, 0] += 0.2                    # 右手偏 20cm → 必然要升级
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                s = refine_run(out, poses, hr, ha,
                               cfg=RefineConfig(horizon=20), requested="mpc")
            for f in ("action_refine.json", "action_refine.npz", "hand_poses.npz"):
                self.assertTrue((out / f).exists(), f)
            doc = json.loads((out / "action_refine.json").read_text())
            self.assertEqual(len(doc["blocks"]), 2)
            self.assertTrue(doc["summary"]["needs_escalation"])
            self.assertFalse(doc["summary"]["refined"])       # 没真精修，标清楚
            self.assertIn("Replay", doc["summary"]["refined_note"])
            npz = np.load(out / "action_refine.npz")
            for k in ("e", "ep", "eR", "object_poses_ref", "object_poses_ach",
                      "object_valid", "grasp", "block_start", "block_stop",
                      "block_status", "block_mode", "block_escalate"):
                self.assertIn(k, npz.files, k)
            self.assertEqual(npz["e"].shape, (T,))
            self.assertEqual(np.load(out / "hand_poses.npz")["hand_ref"].shape, (T, 2, 7))
            self.assertTrue(s["needs_escalation"])
            self.assertIn("⚠", buf.getvalue())               # 得有那句不许当精修用的警告

    def test_run_with_no_hand_error_needs_nothing(self):
        T = 40
        hr, ha = hands(T)
        with tempfile.TemporaryDirectory() as td:
            with contextlib.redirect_stdout(io.StringIO()):
                s = refine_run(Path(td), self._poses(T), hr, ha,
                               cfg=RefineConfig(horizon=20))
            self.assertFalse(s["needs_escalation"])
            self.assertEqual(s["status_counts"]["ok"], 2)

    def test_run_rejects_frame_count_mismatch(self):
        """帧数对不上就是逐帧没对齐，此时误差是假的，必须报错而不是广播。"""
        hr, ha = hands(30)
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(ValueError) as cm:
                refine_run(Path(td), self._poses(40), hr, ha)
            self.assertIn("帧数", str(cm.exception))

    def test_run_reads_npz_path_as_well_as_object(self):
        T = 20
        with tempfile.TemporaryDirectory() as td:
            npz = save_object_poses(Path(td) / "object_poses.npz", self._poses(T))
            hr, ha = hands(T)
            with contextlib.redirect_stdout(io.StringIO()):
                s = refine_run(Path(td), npz, hr, ha, cfg=RefineConfig(horizon=20))
            self.assertEqual(s["n_frames"], T)

    def test_state_strings_drive_the_grasp_mask(self):
        """只有 grasped_* 的帧算受影响；static 帧的偏差不该记到物体上。"""
        T = 20
        states = ["static"] * 10 + ["grasped_l"] * 10
        hr, ha = hands(T)
        ha[:, 0, 0] += 0.5
        with tempfile.TemporaryDirectory() as td:
            with contextlib.redirect_stdout(io.StringIO()):
                refine_run(Path(td), self._poses(T, states), hr, ha,
                           cfg=RefineConfig(horizon=20))
            npz = np.load(Path(td) / "action_refine.npz")
        np.testing.assert_allclose(npz["e"][:10], 0.0, atol=1e-6)
        np.testing.assert_allclose(npz["e"][10:], 0.5, atol=1e-5)

    def test_cli_refuses_a_dir_without_hand_poses(self):
        """只开过 --object_tracking on 的目录判不了 —— 不许拿参考当执行（误差恒 0）。"""
        from web2robot.refine.cli import main
        with tempfile.TemporaryDirectory() as td:
            save_object_poses(Path(td) / "object_poses.npz", self._poses(20))
            with self.assertRaises(SystemExit) as cm:
                main(["--run", td])
            self.assertIn("hand_poses.npz", str(cm.exception))

    def test_cli_rejudges_from_disk(self):
        from web2robot.refine.cli import main
        T = 40
        hr, ha = hands(T)
        ha[:, 1, 0] += 0.2
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            save_object_poses(out / "object_poses.npz", self._poses(T))
            with contextlib.redirect_stdout(io.StringIO()):
                refine_run(out, self._poses(T), hr, ha, cfg=RefineConfig(horizon=20))
                # 把预算放大到 20cm/帧，同一份数据就不该再要求升级
                main(["--run", str(out), "--per_frame_budget", "0.5"])
            doc = json.loads((out / "action_refine.json").read_text())
        self.assertFalse(doc["summary"]["needs_escalation"])
        self.assertEqual(doc["summary"]["per_frame_budget"], 0.5)


if __name__ == "__main__":
    unittest.main()
