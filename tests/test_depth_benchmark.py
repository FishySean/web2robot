"""把论文里要引的那张表钉死在测试里。

这份测试的作用和别的不一样：它不是"防止代码改坏"，是**防止论文里的数字和仓库里的
证据悄悄脱钩**。一旦有人动了统计口径（中位改均值、换对齐方式、改单位），这里当场变红，
而不是等到投稿前才发现表和代码算出来的不是一回事。

所以断言写的是**具体数值**，不是"大于/小于"这种软条件。数值来源是 2026-07-14 那次
评测的输出，也就是 `.claude` 记忆和 `MODEL_ROUTING_RESULTS.md` 里记的那一组。

秒级、纯 numpy，不需要 GPU、不需要 HO-3D 数据集 —— 因为读的是冻好的
`evidence/depth_benchmark_ho3d/data/bench_*.npz`。
"""
import sys
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from web2robot.eval import depth_benchmark as DB  # noqa: E402

# 序列 → 方法 → (深度err cm, 平面err cm, 深度相对 %, 深度跟随 r, GT 帧数)
PUBLISHED = {
    "ABF12": {"wilor": (11.0, 0.7, 22, -0.64, 74),
              "hawor": (0.6, 2.6, 1, +0.61, 74)},
    "SMu41": {"wilor": (9.5, 0.4, 17, +0.11, 46),
              "hawor": (3.5, 2.0, 6, +0.61, 46)},
    "MC4":   {"wilor": (0.7, 1.7, 1, +0.91, 66),
              "hawor": (2.6, 2.3, 5, +0.87, 66)},
}


class TestEvidenceIsPresent(unittest.TestCase):
    """证据文件本身必须在库里 —— 这条测试就是"别再被 gitignore 吃掉"的守卫。"""

    def test_all_three_sequences_are_frozen(self):
        for seq in DB.SEQUENCES:
            p = DB.EVIDENCE_DIR / f"bench_{seq}.npz"
            self.assertTrue(p.is_file(), f"{p} 不在了 —— 论文证据丢了")

    def test_each_file_has_all_three_parties(self):
        for seq in DB.SEQUENCES:
            b = DB.load_bench(seq)
            for k in ("gt_frames", "gt_wrist", "hawor_frames", "hawor_wrist",
                      "wilor_frames", "wilor_wrist"):
                self.assertIn(k, b, f"{seq} 缺 {k}")
            self.assertGreater(len(b["gt_frames"]), 0)
            self.assertGreater(len(b["hawor_frames"]), 0)
            self.assertGreater(len(b["wilor_frames"]), 0)

    def test_stored_points_are_metres_not_millimetres(self):
        """单位错了整张表就差三个数量级，而每个数看着都"像"合理值。"""
        for seq in DB.SEQUENCES:
            b = DB.load_bench(seq)
            z = np.abs(b["gt_wrist"][:, 2])
            self.assertTrue((z > 0.1).all() and (z < 3.0).all(),
                            f"{seq} GT 深度 {z.min():.3f}~{z.max():.3f}，不像米")

    def test_missing_file_says_how_to_regenerate(self):
        with self.assertRaises(FileNotFoundError) as cm:
            DB.load_bench("ABF12", evidence_dir="/nonexistent")
        self.assertIn("freeze_depth_benchmark.py", str(cm.exception))


class TestPublishedNumbers(unittest.TestCase):
    """逐个序列 × 逐个方法，和 2026-07-14 那次评测的数字对齐。"""

    @classmethod
    def setUpClass(cls):
        cls.results = DB.evaluate_all()

    def test_every_published_number_reproduces(self):
        for seq, per_method in PUBLISHED.items():
            for method, (dep, inp, rel, r, n) in per_method.items():
                got = self.results[seq][method]
                with self.subTest(seq=seq, method=method):
                    self.assertEqual(got["n_frames"], n)
                    self.assertAlmostEqual(got["depth_cm"], dep, delta=0.05)
                    self.assertAlmostEqual(got["inplane_cm"], inp, delta=0.05)
                    self.assertAlmostEqual(got["depth_rel_pct"], rel, delta=0.5)
                    self.assertAlmostEqual(got["depth_r"], r, delta=0.005)

    def test_the_headline_claim_holds(self):
        """"ABF12 上深度误差 11cm → 0.6cm" —— 论文摘要里那句话。"""
        w = self.results["ABF12"]["wilor"]["depth_cm"]
        h = self.results["ABF12"]["hawor"]["depth_cm"]
        self.assertGreater(w, 10.0)
        self.assertLess(h, 1.0)
        self.assertGreater(w / h, 15.0)

    def test_depth_following_flips_from_negative_to_positive(self):
        """比误差更要紧的一项：WiLoR+MoGe 在 ABF12 上深度是**反相关**的。

        r 为负意味着物体靠近时它认为在远离 —— 拿这种深度做重定向会得到反向的 reach，
        而只看绝对误差看不出这件事。
        """
        self.assertLess(self.results["ABF12"]["wilor"]["depth_r"], -0.5)
        self.assertGreater(self.results["ABF12"]["hawor"]["depth_r"], +0.5)

    def test_hawor_is_bounded_not_uniformly_better(self):
        """诚实版结论：HaWoR 是"稳定有界"，不是每条都碾压。MC4 上它反而输。"""
        hawor = [self.results[s]["hawor"]["depth_cm"] for s in DB.SEQUENCES]
        wilor = [self.results[s]["wilor"]["depth_cm"] for s in DB.SEQUENCES]
        self.assertLessEqual(max(hawor), 3.5 + 0.05, "HaWoR 三条都该 ≤3.5cm")
        self.assertGreater(max(wilor), 9.0, "WiLoR 该有灾难性失败的那条")
        self.assertLess(self.results["MC4"]["wilor"]["depth_cm"],
                        self.results["MC4"]["hawor"]["depth_cm"],
                        "MC4 上 WiLoR 更准 —— 这条输了要照实报，别让表变成单边宣传")

    def test_inplane_error_is_worse_for_hawor(self):
        """另一处如实报的地方：平面误差 HaWoR 略逊（自估焦距 600 + SLAM 系）。"""
        for seq in DB.SEQUENCES:
            self.assertGreater(self.results[seq]["hawor"]["inplane_cm"],
                               self.results[seq]["wilor"]["inplane_cm"], seq)


class TestAlignment(unittest.TestCase):
    """对齐逻辑：逐方法各自和 GT 求交，不是三方一起求交。"""

    def _bench(self):
        return {
            "gt_frames": np.array([0, 1, 2, 3]),
            "gt_wrist": np.arange(12, dtype=float).reshape(4, 3),
            "hawor_frames": np.array([1, 2, 3]),
            "hawor_wrist": np.arange(9, dtype=float).reshape(3, 3),
            "wilor_frames": np.array([0, 3]),
            "wilor_wrist": np.arange(6, dtype=float).reshape(2, 3),
        }

    def test_each_method_keeps_its_own_frames(self):
        b = self._bench()
        self.assertEqual(list(DB.align(b, "hawor")[0]), [1, 2, 3])
        self.assertEqual(list(DB.align(b, "wilor")[0]), [0, 3])

    def test_alignment_pairs_the_right_rows_not_just_the_right_count(self):
        """帧号对上了但取错了行，帧数一样、误差却全是垃圾 —— 这是静默失败。"""
        b = self._bench()
        _, gt, est = DB.align(b, "wilor")
        np.testing.assert_array_equal(gt, b["gt_wrist"][[0, 3]])
        np.testing.assert_array_equal(est, b["wilor_wrist"][[0, 1]])

    def test_unknown_method_raises(self):
        with self.assertRaises(ValueError):
            DB.align(self._bench(), "moge")

    def test_empty_method_yields_no_metrics(self):
        b = self._bench()
        b["wilor_frames"] = np.zeros(0, np.int64)
        b["wilor_wrist"] = np.zeros((0, 3))
        self.assertIsNone(DB.evaluate(b, "wilor"))


class TestTable(unittest.TestCase):

    def test_small_gt_depth_range_gets_an_explicit_caveat(self):
        """SMu41 的 GT 深度只变 1cm，表里必须自己带上"r 不可信"的告示。"""
        txt = DB.format_table(DB.evaluate_all())
        self.assertIn("SMu41", txt)
        self.assertIn("不可信", txt)
        self.assertRegex(txt, r"SMu41.*只变 1\.0cm")

    def test_sequences_with_real_depth_motion_get_no_caveat(self):
        txt = DB.format_table(DB.evaluate_all(sequences=("ABF12", "MC4")))
        self.assertNotIn("不可信", txt)


class TestProvenance(unittest.TestCase):
    """把 HaWoR 那三次运行的出处钉住 —— 尤其是它自己估的度量尺度。

    这三个数是 "0.6 cm" 的前提：HaWoR 的尺度逐段现估，同一份权重换一段视频就是另一个数
    （这里 0.19 / 2.34 / 3.92，差 20 倍）。重跑拿到别的尺度，深度误差表就一定跟着变。
    所以尺度必须和结论一起被冻住，否则表和它的前提会悄悄脱钩。
    """

    SCALES = {"abf12": "0.1902507320046425",
              "smu41": "3.923701286315918",
              "mc4": "2.3423545360565186"}
    PROV = (Path(__file__).resolve().parents[1]
            / "evidence" / "depth_benchmark_ho3d" / "provenance")

    def _log(self, seq):
        import gzip
        p = self.PROV / f"hawor_run_{seq}.log.gz"
        self.assertTrue(p.is_file(), f"{p} 不在了 —— 出处日志别再被清理掉")
        with gzip.open(p, "rt", errors="replace") as f:
            return f.read()

    def test_estimated_scale_matches_what_the_readme_claims(self):
        for seq, want in self.SCALES.items():
            with self.subTest(seq=seq):
                self.assertIn(f"estimated scale: {want}", self._log(seq))

    def test_readme_table_lists_the_same_scales(self):
        """README 里的表和日志不许各说各话。"""
        txt = (self.PROV / "README.md").read_text()
        for seq, want in self.SCALES.items():
            self.assertIn(want, txt, f"{seq} 的尺度在 README 表里对不上")

    def test_hawor_ran_on_the_default_focal_not_the_true_intrinsics(self):
        """这条是对比公平性的前提：HaWoR 用默认 600，WiLoR 用了 HO-3D 真 camMat。

        也就是说这份对比对 WiLoR 有利，而 WiLoR 仍差一个量级 —— 引用这张表时
        必须把这句话一起写上，否则会被合理地质疑"是不是给 HaWoR 喂了真内参"。
        """
        for seq in self.SCALES:
            with self.subTest(seq=seq):
                self.assertIn("use default 600", self._log(seq))

    def test_the_nfr_in_the_log_matches_the_frozen_npz(self):
        """日志里 SLAM 文件名带的帧数要和冻结数据的 nfr 对得上，否则冻的是另一次运行。"""
        for seq in self.SCALES:
            with self.subTest(seq=seq):
                nfr = int(DB.load_bench(seq.upper() if seq != "smu41" else "SMu41")["nfr"])
                self.assertIn(f"hawor_slam_w_scale_0_{nfr}.npz", self._log(seq))


if __name__ == "__main__":
    unittest.main()
