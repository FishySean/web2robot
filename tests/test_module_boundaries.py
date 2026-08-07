"""模块边界 —— `src/` 不许 import 上游仓库的模块。

这条不变量服务的是"模块化、即插即用"这个架构目标：**未来新增一种技术方案，
应当只在对应模块里加一个实现，不需要改动其他模块。** 如果我们的代码反过来
import 上游的 ``models.*`` / ``utils.*`` / ``sim.*``，那就等于把自己焊死在
EgoInfinity 的目录结构上 —— 上游一动、或者想把这套碰撞检测用到别的重定向框架上，
就得回来改代码。

碰撞层现在的做法是：机器人构型以 ``robot_cfg`` 字典传进来（只用 ``env_cls`` /
``scene_path`` / ``start_config`` 三个键），所以换机器人、换上游框架都不用改这一层。
这个测试就是防止有人哪天"顺手"加一行 ``from sim.robots.m7.config import CONFIG``
把这个性质破掉 —— 那一行能跑通（薄壳把上游也放进 PYTHONPATH 了），所以只靠人看
是看不住的。

跑法::

    envs/rt_env/bin/python -m unittest discover -s tests -v
"""
import ast
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# 上游 EgoInfinity/retarget 的顶层包名。我们的代码不许直接 import 这些。
UPSTREAM_TOP = {"models", "utils", "sim", "configs", "scripts", "data_utils"}


def _imported_tops(py: Path):
    """这个文件 import 了哪些顶层模块名（只看绝对 import；相对 import 不算）。"""
    tree = ast.parse(py.read_text())
    tops = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                tops.add(a.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:      # level>0 是相对 import
                tops.add(node.module.split(".")[0])
    return tops


class TestNoUpstreamImports(unittest.TestCase):
    def test_src_does_not_import_upstream(self):
        offenders = []
        for py in sorted((REPO / "src").rglob("*.py")):
            hits = _imported_tops(py) & UPSTREAM_TOP
            if hits:
                offenders.append(f"{py.relative_to(REPO).as_posix()}  ← {sorted(hits)}")
        self.assertEqual(
            offenders, [],
            "src/ 里 import 了上游模块，会把我们焊死在 EgoInfinity 的目录结构上；"
            "需要的东西请以参数传入：\n  " + "\n  ".join(offenders))


class TestCollisionPackage(unittest.TestCase):
    """碰撞/轨迹两个包能在新位置 import，且公开符号齐全。"""

    def test_import_in_rt_env(self):
        import subprocess
        code = (
            "from web2robot.collision import (M7CapsuleModel, HandSphereModel,"
            " ArmTorsoFilter, DualHandFilter);"
            "from web2robot.trajectory import (clean_wrist_trajectory, blend_to_rest,"
            " relax_fingers, FILL_REST, STATUS_NAMES);"
            "import inspect;"
            # 构造签名是判据的一部分：保守门槛 4cm 不能被谁顺手调松
            "assert inspect.signature(ArmTorsoFilter.__init__)"
            ".parameters['enter_thresh'].default == 0.04;"
            "assert inspect.signature(DualHandFilter.__init__)"
            ".parameters['enter_thresh'].default == 0.04;"
            "assert inspect.signature(ArmTorsoFilter.__init__)"
            ".parameters['include_fingers'].default is True;"
            "print('ok')"
        )
        r = subprocess.run([str(_interpreter()), "-c", code],
                           capture_output=True, text=True,
                           env={"PYTHONPATH": str(REPO / "src"),
                                "PATH": "/usr/bin:/bin", "HOME": str(Path.home())})
        self.assertEqual(r.returncode, 0, r.stderr[-2000:])
        self.assertIn("ok", r.stdout)


def _interpreter() -> Path:
    import sys
    sys.path.insert(0, str(REPO / "src"))
    from web2robot.paths import P
    return P.env("retarget")


if __name__ == "__main__":
    unittest.main(verbosity=2)
