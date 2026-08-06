"""src/ 里不许出现绝对路径字面量。

这条不变量是整次重构的核心：重构前 40 个 .py 文件里散着
``/mnt/vlm/fanshaoheng``，换机器或搬目录就全碎。现在唯一的来源是
``configs/paths.yaml``，代码一律走 ``web2robot.paths.P``。

写成测试而不是写在 README 里，因为"约定"会被下一次赶工时的一行
硬编码悄悄破掉，而这个测试 0.1 秒就能跑，会当场报出来。

跑法::

    envs/rt_env/bin/python -m unittest discover -s tests -v
"""
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# 允许出现绝对路径的地方：配置文件本身，以及 paths.py 里解释那个坑的注释。
ALLOWED = {
    "src/web2robot/paths.py",     # 文档字符串里举了 gs3dgs_env 的实例
}
NEEDLES = ("/mnt/vlm", "/home/", "/root/")


class TestNoHardcodedPaths(unittest.TestCase):
    def test_src_has_no_absolute_paths(self):
        offenders = []
        for py in sorted((REPO / "src").rglob("*.py")):
            rel = py.relative_to(REPO).as_posix()
            if rel in ALLOWED:
                continue
            for i, line in enumerate(py.read_text().splitlines(), 1):
                for needle in NEEDLES:
                    if needle in line:
                        offenders.append(f"{rel}:{i}  {line.strip()[:90]}")
        self.assertEqual(
            offenders, [],
            "src/ 里出现了绝对路径字面量，请改成走 web2robot.paths.P：\n  "
            + "\n  ".join(offenders))

    def test_scripts_only_hardcode_via_root(self):
        """scripts/ 里的薄壳可以拼路径，但必须相对自己的位置推出仓库根。"""
        for sh in sorted((REPO / "scripts").rglob("*.sh")):
            body = sh.read_text()
            with self.subTest(script=sh.name):
                self.assertNotIn("/mnt/vlm", body)
                self.assertIn("BASH_SOURCE", body,
                              "薄壳必须用 BASH_SOURCE 推仓库根，不能写死路径")


class TestQualityModuleWiring(unittest.TestCase):
    """迁移后的质检模块能在新位置 import，且权重路径是查出来的。"""

    def test_import_and_weights_resolved(self):
        import subprocess
        import sys
        code = (
            "from web2robot.quality import QCConfig, diagnose_clip, ViewClass;"
            "from web2robot.quality import hand_gate;"
            "from web2robot.routing import labels;"
            "assert QCConfig().hand_weights is None, '默认应当留空、由 paths.yaml 查';"
            "w = hand_gate.find_weights();"
            "assert w and w.endswith('.pt'), f'手部检测器没查到: {w}';"
            "print('ok')"
        )
        r = subprocess.run(
            [str(_interpreter()), "-c", code],
            capture_output=True, text=True,
            env={"PYTHONPATH": str(REPO / "src"), "PATH": "/usr/bin:/bin",
                 "HOME": str(Path.home())})
        self.assertEqual(r.returncode, 0, r.stderr[-2000:])
        self.assertIn("ok", r.stdout)


def _interpreter() -> Path:
    import sys
    sys.path.insert(0, str(REPO / "src"))
    from web2robot.paths import P
    return P.env("retarget")


if __name__ == "__main__":
    unittest.main(verbosity=2)
