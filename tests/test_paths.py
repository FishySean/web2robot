"""paths.py 的回归测试。

用 stdlib unittest 而不是 pytest：三个 venv 都没装 pytest，而这是共享机器
不能随便装包。unittest 永远可用，测试也就永远跑得起来。

跑法（用 rt_env，它同时装了质检与重定向要用的东西）::

    envs/rt_env/bin/python -m unittest discover -s tests -v
    envs/rt_env/bin/python tests/test_paths.py          # 单跑这一个
"""
import subprocess
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from web2robot.paths import P  # noqa: E402

MARKERS = {
    "retarget":   "ultralytics",   # 质检的手部检测器
    "hawor":      "torch",
    "perception": "torch",
}


def _prefix_of(py: Path) -> str:
    return subprocess.run([str(py), "-c", "import sys; print(sys.prefix)"],
                          capture_output=True, text=True).stdout.strip()


class TestConfig(unittest.TestCase):
    def test_repo_root_has_config(self):
        self.assertTrue((P.repo_root / "configs" / "paths.yaml").exists())

    def test_unknown_key_error_lists_available(self):
        with self.assertRaises(KeyError) as cm:
            P.weights("no_such_key")
        self.assertIn("hand_detector", str(cm.exception))


class TestEnvInterpreters(unittest.TestCase):
    """venv 解释器的路径不能被 resolve —— 这是踩过的坑，不是洁癖。

    venv 的 bin/python 本身是指向基础环境的 symlink，隔离靠 pyvenv.cfg 所在
    位置。跟着 symlink 走会掉回 gs3dgs_env，包完全是另一套：实测
    envs/rt_env/bin/python 有 ultralytics，resolve 成 gs3dgs_env/bin/python3.10
    之后就没有。这种错以 ModuleNotFoundError 的形式出现，看起来像"环境装漏了"，
    很难查到真因。
    """

    def test_each_env_keeps_its_own_prefix(self):
        for key in sorted(MARKERS):
            with self.subTest(env=key):
                py = P.env(key)
                expect = str(P.repo_root / "envs" / py.parent.parent.name)
                self.assertEqual(
                    _prefix_of(py), expect,
                    f"{key} 的 sys.prefix 不对，说明解释器路径被 resolve 掉了")

    def test_env_prefixes_are_all_distinct(self):
        got = {k: _prefix_of(P.env(k)) for k in MARKERS}
        self.assertEqual(len(set(got.values())), len(MARKERS),
                         f"环境撞车了: {got}")

    def test_env_has_its_marker_package(self):
        for key, pkg in sorted(MARKERS.items()):
            with self.subTest(env=key, pkg=pkg):
                r = subprocess.run(
                    [str(P.env(key)), "-c",
                     f"import importlib.util as u; "
                     f"raise SystemExit(0 if u.find_spec('{pkg}') else 1)"])
                self.assertEqual(r.returncode, 0,
                                 f"{key} 环境里没有 {pkg}，环境接错了")


class TestWeights(unittest.TestCase):
    def test_hand_detector_found(self):
        w = P.weights("hand_detector")
        self.assertIsNotNone(
            w, "手部检测器权重找不到，找过：" +
            ", ".join(str(c) for c in P.weight_candidates("hand_detector")))
        self.assertGreater(w.stat().st_size, 1_000_000)

    def test_missing_weight_returns_none_not_raises(self):
        """缺失必须是 None（→ 报 unknown），不能抛异常也不能当成"没有手"。"""
        P._cfg.setdefault("weights", {})["_probe"] = "no/such/file.pt"
        try:
            self.assertIsNone(P.weights("_probe"))
        finally:
            del P._cfg["weights"]["_probe"]


if __name__ == "__main__":
    unittest.main(verbosity=2)
