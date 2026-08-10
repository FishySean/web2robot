"""``retarget/`` 两个模块的隔离测试 —— 迁移前的内联实现当参照物。

这是迁移方法论里的第②条线（隔离比对）：不跑端到端、不要 GPU，直接拿**搬迁前写在
上游 ``scripts/test.py`` 里的那几段代码**当参照实现，喂同一份合成输入，逐位比对。

参照实现是从 2026-08-10 迁移前的 ``test.py`` 原样抄进来的（见每个 ``_old_*``
函数的注释），刻意不整理、不重命名 —— 它的价值就在于"和当时跑出所有基线的那份
代码一模一样"。整理它就等于失去参照价值。

留在测试里而不是删掉，是因为它以后还有用：``fallback.py`` 每次改动都能再对一次
"和 2026-08-10 的基线行为是否一致"，而这条比"端到端 md5 没变"更早、更便宜。
"""
import sys
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from web2robot.retarget.fallback import (  # noqa: E402
    apply_rest_fallback, clean_input_wrists, relax_fingers_on_rest,
    status_overlay_text,
)
from web2robot.retarget.root_anchor import sample_best_anchor  # noqa: E402
from web2robot.trajectory.traj_cleanup import (  # noqa: E402
    FILL_REST, STATUS_NAMES, blend_to_rest, clean_wrist_trajectory, relax_fingers,
)


# ── 迁移前的内联实现（原样抄自 test.py，勿整理） ───────────────────────────────

def _old_clean_input(seq_raw, seq_fps, max_interp_sec, max_hold_sec, no_detect):
    """test.py 迁移前的输入侧清洗块。"""
    _raw_l, _raw_r = seq_raw
    left_cam_np, st_l, ca_l, _rep_l = clean_wrist_trajectory(
        _raw_l, seq_fps, max_interp_sec=max_interp_sec,
        max_hold_sec=max_hold_sec, detect_bad=not no_detect, side="left")
    right_cam_np, st_r, ca_r, _rep_r = clean_wrist_trajectory(
        _raw_r, seq_fps, max_interp_sec=max_interp_sec,
        max_hold_sec=max_hold_sec, detect_bad=not no_detect, side="right")
    if np.isnan(left_cam_np[:, 0]).all() or np.isnan(right_cam_np[:, 0]).all():
        _miss = "left" if np.isnan(left_cam_np[:, 0]).all() else "right"
        raise RuntimeError(f"{_miss} hand is never detected")
    return left_cam_np, right_cam_np, st_l, st_r, ca_l, ca_r


def _old_rest_fallback(q_left, q_right, st_l, st_r, _rest, seq_fps, ramp, seq_len):
    """test.py 迁移前的输出侧静息位块。"""
    w_rest_l = np.zeros(seq_len)
    w_rest_r = np.zeros(seq_len)
    if (st_l == FILL_REST).any():
        q_left, w_rest_l = blend_to_rest(
            q_left, st_l, np.asarray(_rest["left"], np.float64),
            seq_fps, ramp_sec=ramp)
    if (st_r == FILL_REST).any():
        q_right, w_rest_r = blend_to_rest(
            q_right, st_r, np.asarray(_rest["right"], np.float64),
            seq_fps, ramp_sec=ramp)
    return q_left, q_right, w_rest_l, w_rest_r


def _old_relax(Q_lf, Q_rf, w_rest_l, w_rest_r):
    """test.py 迁移前的手指放松块。"""
    if w_rest_l.any():
        Q_lf = relax_fingers(Q_lf, w_rest_l)
    if w_rest_r.any():
        Q_rf = relax_fingers(Q_rf, w_rest_r)
    return Q_lf, Q_rf


def _old_overlay(st_l, st_r, t):
    """test.py 迁移前的来路标注块。"""
    return " ".join(f"{s}:{STATUS_NAMES[int(a[t])]}"
                    for s, a in (("L", st_l), ("R", st_r))
                    if int(a[t]) != 0)


def _old_best_of_n(estimate, select, n_samples, seed, seeder):
    """test.py 迁移前的 best-of-N 块（``torch.manual_seed`` 换成可注入的 seeder）。"""
    best = None
    for i in range(n_samples):
        if seed is not None:
            seeder(seed + i)
        kf_positions, Rs, ts = estimate()
        R_anchor, t_anchor, anchor_rate = select(Rs, ts)
        if best is None or anchor_rate > best[0]:
            best = (anchor_rate, kf_positions, Rs, ts, R_anchor, t_anchor)
    return best


# ── 合成输入 ──────────────────────────────────────────────────────────────────

def _synthetic_wrists(T=180, fps=15.0, seed=0):
    """两只手腕的 (T, 7) 位姿轨迹：平滑运动 + 三种坏帧，覆盖三条填补分支。

    左手中段挖一个 45 帧（3s）的长空洞 → 必须走 FILL_REST（这正是 fill_jar 那段
    11 秒左手崩坏的缩影）；右手挖一个 9 帧（0.6s）的内部小洞 → FILL_INTERP；
    再在右手放一个深度爆点 → 先被判坏、打成缺失、再填。开头挖 4 帧边界洞 →
    FILL_HOLD。
    """
    rng = np.random.default_rng(seed)
    t = np.arange(T) / fps
    def traj(phase):
        xyz = np.stack([0.30 + 0.05 * np.sin(2 * t + phase),
                        0.10 * np.cos(1.5 * t + phase),
                        0.45 + 0.04 * np.sin(t + phase)], axis=1)
        # 单位四元数，绕 z 缓慢转
        a = 0.3 * np.sin(0.7 * t + phase)
        quat = np.stack([np.cos(a / 2), np.zeros(T), np.zeros(T), np.sin(a / 2)], axis=1)
        return np.concatenate([xyz, quat], axis=1).astype(np.float32)
    L, R = traj(0.0), traj(1.1)
    L[60:105] = np.nan                    # 长空洞 → FILL_REST
    R[80:89] = np.nan                     # 短内部洞 → FILL_INTERP
    R[:4] = np.nan                        # 边界洞 → FILL_HOLD
    R[130, 2] += 3.5                      # 深度爆点 → 判坏
    L += rng.normal(0, 1e-5, L.shape)     # 一点噪声，避免退化成常量
    R += rng.normal(0, 1e-5, R.shape)
    return L.astype(np.float32), R.astype(np.float32)


class TestFallbackMatchesPreMigration(unittest.TestCase):
    """``fallback.py`` 与迁移前内联实现逐位一致。"""

    FPS = 15.0

    @classmethod
    def setUpClass(cls):
        cls.raw = _synthetic_wrists(fps=cls.FPS)
        cls.T = len(cls.raw[0])

    def _run_both(self):
        kw = dict(max_interp_sec=1.5, max_hold_sec=0.5)
        new = clean_input_wrists(*self.raw, self.FPS, detect_bad=True,
                                 log=lambda *_: None, **kw)
        old = _old_clean_input(self.raw, self.FPS, kw["max_interp_sec"],
                               kw["max_hold_sec"], no_detect=False)
        return new, old

    def test_input_cleanup_bitwise(self):
        new, old = self._run_both()
        o_l, o_r, o_stl, o_str, o_cal, o_car = old
        for name, a, b in (("left", new.left, o_l), ("right", new.right, o_r),
                           ("status_left", new.status_left, o_stl),
                           ("status_right", new.status_right, o_str),
                           ("cause_left", new.cause_left, o_cal),
                           ("cause_right", new.cause_right, o_car)):
            self.assertTrue(np.array_equal(a, b, equal_nan=True), f"{name} 不一致")
            self.assertEqual(a.dtype, b.dtype, f"{name} dtype 不一致")

    def test_the_synthetic_input_actually_exercises_all_three_branches(self):
        """参照物再准，输入没打到分支也证明不了什么 —— 所以先钉住覆盖面。"""
        new, _ = self._run_both()
        seen = set(new.status_left.tolist()) | set(new.status_right.tolist())
        self.assertEqual(seen, {0, 1, 2, 3},
                         f"合成输入没覆盖全四种状态，只看到 {sorted(seen)}")

    def test_rest_fallback_and_fingers_bitwise(self):
        new_cl, old = self._run_both()
        _, _, o_stl, o_str, _, _ = old
        rng = np.random.default_rng(7)
        q_l = rng.normal(0, 0.4, (self.T, 7)).astype(np.float32)
        q_r = rng.normal(0, 0.4, (self.T, 7)).astype(np.float32)
        Q_lf = rng.uniform(0, 1.5, (self.T, 12)).astype(np.float32)
        Q_rf = rng.uniform(0, 1.5, (self.T, 12)).astype(np.float32)
        rest = {"left": np.full(7, 0.11), "right": np.full(7, -0.11)}

        n_ql, n_qr, n_wl, n_wr = apply_rest_fallback(
            q_l.copy(), q_r.copy(), new_cl.status_left, new_cl.status_right,
            rest, self.FPS, ramp_sec=0.5, log=lambda *_: None)
        o_ql, o_qr, o_wl, o_wr = _old_rest_fallback(
            q_l.copy(), q_r.copy(), o_stl, o_str, rest, self.FPS, 0.5, self.T)

        for name, a, b in (("q_left", n_ql, o_ql), ("q_right", n_qr, o_qr),
                           ("w_left", n_wl, o_wl), ("w_right", n_wr, o_wr)):
            self.assertTrue(np.array_equal(a, b), f"{name} 不一致")
        self.assertGreater(n_wl.max(), 0.0, "左手长空洞没触发静息位，这个用例白跑了")

        n_lf, n_rf = relax_fingers_on_rest(Q_lf.copy(), Q_rf.copy(), n_wl, n_wr)
        o_lf, o_rf = _old_relax(Q_lf.copy(), Q_rf.copy(), o_wl, o_wr)
        self.assertTrue(np.array_equal(n_lf, o_lf), "Q_left_fingers 不一致")
        self.assertTrue(np.array_equal(n_rf, o_rf), "Q_right_fingers 不一致")

    def test_overlay_text_matches_every_frame(self):
        new_cl, old = self._run_both()
        _, _, o_stl, o_str, _, _ = old
        for t in range(self.T):
            self.assertEqual(status_overlay_text(new_cl.status_left,
                                                 new_cl.status_right, t),
                             _old_overlay(o_stl, o_str, t), f"第 {t} 帧标注不一致")

    def test_never_detected_hand_is_rejected(self):
        """整段没检测到 → 报错，不许编一只手出来（根估计器同时吃两只手腕）。"""
        L, R = self.raw
        dead = np.full_like(L, np.nan)
        with self.assertRaises(RuntimeError) as cm:
            clean_input_wrists(dead, R, self.FPS, log=lambda *_: None)
        self.assertIn("left hand is never detected", str(cm.exception))
        with self.assertRaises(RuntimeError) as cm:
            clean_input_wrists(L, dead, self.FPS, log=lambda *_: None)
        self.assertIn("right hand is never detected", str(cm.exception))

    def test_relax_fingers_on_rest_tolerates_no_fingers(self):
        """没做手指重定向的机器人 Q 是 None，不能因此炸。"""
        w = np.linspace(0, 1, 10)
        self.assertEqual(relax_fingers_on_rest(None, None, w, w), (None, None))


class TestRootAnchorMatchesPreMigration(unittest.TestCase):
    """``root_anchor.py`` 与迁移前内联实现一致，且 n_samples=1 就是原始单发路径。"""

    @staticmethod
    def _fakes(rates):
        """假的估计器/打分器：第 i 次采样给出可辨认的数组和预设的 ik_rate。"""
        box = {"i": 0}
        def estimate():
            i = box["i"]
            return (np.array([i, i]), np.full(3, i, np.float64), np.full(3, -i, np.float64))
        def select(Rs, ts):
            i = box["i"]; box["i"] += 1
            return np.full((3, 3), i, np.float64), np.full(3, i * 0.5), rates[i]
        return estimate, select

    def test_single_shot_is_identical_and_silent(self):
        est, sel = self._fakes([0.42])
        lines = []
        got = sample_best_anchor(est, sel, n_samples=1, log=lines.append)
        est2, sel2 = self._fakes([0.42])
        exp = _old_best_of_n(est2, sel2, 1, None, lambda _: None)
        self.assertEqual(lines, [], "n_samples=1 不该打印任何东西（基线都是单发跑的）")
        self.assertEqual(got.ik_rate, exp[0])
        for a, b in zip((got.kf_positions, got.Rs, got.ts, got.R_anchor, got.t_anchor),
                        exp[1:]):
            self.assertTrue(np.array_equal(a, b))

    def test_best_of_n_picks_highest_rate_and_keeps_that_samples_arrays(self):
        rates = [0.10, 0.90, 0.30]          # 最好的是第 2 次（下标 1）
        est, sel = self._fakes(rates)
        got = sample_best_anchor(est, sel, n_samples=3, log=lambda *_: None)
        self.assertEqual(got.ik_rate, 0.90)
        # 窗口估计和锚点必须来自同一次采样 —— 混用会让后面的逐帧混合跑偏
        self.assertTrue(np.array_equal(got.kf_positions, np.array([1, 1])))
        self.assertTrue(np.array_equal(got.Rs, np.full(3, 1.0)))
        self.assertTrue(np.array_equal(got.R_anchor, np.full((3, 3), 1.0)))
        est2, sel2 = self._fakes(rates)
        exp = _old_best_of_n(est2, sel2, 3, None, lambda _: None)
        self.assertEqual(got.ik_rate, exp[0])
        self.assertTrue(np.array_equal(got.kf_positions, exp[1]))

    def test_ties_keep_the_first_sample(self):
        """并列时留先抽到的那次 —— 否则 seed 固定了结果还会变。"""
        est, sel = self._fakes([0.5, 0.5, 0.5])
        got = sample_best_anchor(est, sel, n_samples=3, log=lambda *_: None)
        self.assertTrue(np.array_equal(got.kf_positions, np.array([0, 0])))

    def test_seed_is_offset_per_sample(self):
        seen = []
        est, sel = self._fakes([0.1, 0.2])
        sample_best_anchor(est, sel, n_samples=2, seed=1234,
                           log=lambda *_: None, seed_fn=seen.append)
        self.assertEqual(seen, [1234, 1235])

    def test_no_seeding_when_seed_is_none(self):
        seen = []
        est, sel = self._fakes([0.1])
        sample_best_anchor(est, sel, n_samples=1, seed=None,
                           log=lambda *_: None, seed_fn=seen.append)
        self.assertEqual(seen, [])

    def test_logs_once_per_sample_plus_summary(self):
        est, sel = self._fakes([0.1, 0.2])
        lines = []
        sample_best_anchor(est, sel, n_samples=2, log=lines.append)
        self.assertEqual(len(lines), 3)
        self.assertIn("[sample 1/2]", lines[0])
        self.assertIn("best-of-2", lines[2])

    def test_zero_samples_raises(self):
        est, sel = self._fakes([0.1])
        with self.assertRaises(ValueError):
            sample_best_anchor(est, sel, n_samples=0)


if __name__ == "__main__":
    unittest.main()
