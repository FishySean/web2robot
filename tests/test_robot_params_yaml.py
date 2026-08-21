"""``configs/robots/*.yaml`` —— 机器人参数的**唯一**来源。

格式照 HandUMI（robonet-ai.github.io/handumi-sw）的机器人配置设计：一台机器人一个
yaml，装 IK 权重 / 关节限位 / 静息姿态，外加 ``verified`` 标志位。只借格式不借数值。

这个测试守三件事，每一件都是搬迁**本来会引入的**故障模式：

1. **不许留第二份。** 搬迁最典型的失败是"yaml 加了、代码里那行没删"，然后有人改
   yaml 以为生效了。所以这里用 AST 扫 ``src/`` 里的浮点字面量（AST 而不是 grep：
   注释和 docstring 里引用 MJCF 的原值是**应该**留着的，只有代码里的字面量才算第二份）。
2. **默认值真的是 yaml 里那个数。** 不是"yaml 有一份、签名有一份、两份碰巧相等"——
   断言 ``inspect.signature`` 拿到的默认值和 yaml 逐个相等，就把"从 yaml 读"这件事
   本身钉住了：谁把默认值改回硬编码，只要数值和 yaml 有一点不同就红。
   ``tests/test_module_boundaries.py`` 那边钉的是"这个数必须是 0.04"，两个测试合起来
   才是完整的：一个守数值，一个守来源。
3. **``verified: true`` 不许扩散。** 这个标志位的全部价值在于"看配置的人能分清哪些
   数字是真金实测的"。哪天有人给一组没标定过的参数标上 true，这个区分就废了 ——
   所以 true 的集合是**按名单钉死**的，加一组就必须来改这个名单（连带解释凭据）。

跑法::

    envs/rt_env/bin/python -m unittest tests.test_robot_params_yaml -v
"""
import ast
import inspect
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from web2robot.robots.params import (  # noqa: E402
    META_KEYS, robot_params, robot_params_path, values, verified_rows,
)

#: 搬进 yaml 的那些数字。谁在 ``src/`` 的**代码**里再写一遍就是第二份来源。
#: 刻意不含 1.57（那是 π/2，``sample_config.py`` 里当采样区间用，和 wrist_roll 限位
#: 撞值但没关系）以及 0.04/20.0/60.0/2.0 这类太常见的权重 —— 后者由第 2 条
#: （签名默认值 == yaml）守，不靠字面量黑名单。
MOVED_LITERALS = {
    0.105, 0.135, 0.215,          # collision.proxy.torso_half
    0.012,                        # collision.proxy.tip_radius
    0.0695, 0.119,                # collision.arm_torso.routes.grid.torso_half
    0.139, 0.17, 0.239,           # collision.mesh_aabb_half
    2.79, 2.53, 2.36, 4.36, 1.22, 0.56,   # ik.joint_limits
}

#: ``verified: true`` 的完整名单。改动它需要附上凭据（哪个脚本、哪批素材）。
VERIFIED_TRUE = {
    "m7": {
        "collision.arm_torso.routes.grid",   # sweep_arm_torso_params.py 2026-08-20
        "collision.mesh_aabb_half",          # 量自躯干网格 AABB
    },
    "l3_4": set(),
}


def _float_literals(py: Path):
    """这个文件**代码里**出现的浮点字面量。注释/docstring 不算（它们不是来源）。"""
    tree = ast.parse(py.read_text())
    return {n.value for n in ast.walk(tree)
            if isinstance(n, ast.Constant) and isinstance(n.value, float)}


class TestFilesWellFormed(unittest.TestCase):
    def test_each_yaml_names_itself(self):
        found = sorted(p.stem for p in (REPO / "configs" / "robots").glob("*.yaml"))
        self.assertEqual(found, ["l3_4", "m7"], found)
        for robot in found:
            self.assertEqual(robot_params(robot)["robot"], robot)

    def test_unknown_robot_raises_with_the_known_list(self):
        with self.assertRaises(FileNotFoundError) as cm:
            robot_params("nope")
        self.assertIn("m7", str(cm.exception))

    def test_returned_dict_is_a_copy(self):
        """调用方常把某个 group 当 kwargs 改一改；污染必须只发生在他自己那份上。"""
        a = robot_params("m7")
        a["ik"]["start_config"]["left"][0] = 99.0
        self.assertNotEqual(robot_params("m7")["ik"]["start_config"]["left"][0], 99.0)

    def test_values_strips_metadata(self):
        g = robot_params("m7")["collision"]["arm_torso"]["defaults"]
        self.assertTrue(META_KEYS & set(g), "这个 group 本来就该带 verified/source")
        self.assertFalse(META_KEYS & set(values(g)))
        self.assertIn("enter_thresh", values(g))

    def test_every_group_says_where_the_numbers_came_from(self):
        for robot in ("m7", "l3_4"):
            rows = verified_rows(robot)
            self.assertTrue(rows, robot)
            for path, _ok, source in rows:
                self.assertTrue(source.strip(),
                                f"{robot}:{path} 有 verified 但没写 source —— "
                                f"标志位不说明凭据就没有意义")


class TestVerifiedFlag(unittest.TestCase):
    """``verified: true`` 只能出现在真拿数据量过的那几组上。"""

    def test_true_set_is_pinned(self):
        for robot, expected in VERIFIED_TRUE.items():
            got = {p for p, ok, _ in verified_rows(robot) if ok is True}
            self.assertEqual(
                got, expected,
                f"{robot} 的 verified: true 名单变了。改这个名单请一起写清凭据"
                f"（哪个脚本、哪批素材、什么判据）—— 这个标志位的全部价值就是"
                f"让人分清'实测'和'暂时用着'。")

    def test_flag_is_a_real_bool(self):
        for robot in VERIFIED_TRUE:
            for path, ok, _ in verified_rows(robot):
                self.assertIsInstance(ok, bool, f"{robot}:{path} = {ok!r}")

    def test_neural_route_is_not_marked_verified(self):
        """空覆盖集谈不上"标定过"。它的凭据是字节级 md5，不是量出来的数字。"""
        neural = robot_params("m7")["collision"]["arm_torso"]["routes"]["neural"]
        self.assertFalse(neural["verified"])
        self.assertEqual(values(neural), {},
                         "neural 路线必须是空的（见 collision/presets.py 的说明）")


class TestSingleSource(unittest.TestCase):
    def test_no_moved_number_is_hardcoded_in_src(self):
        offenders = []
        for py in sorted((REPO / "src").rglob("*.py")):
            hits = sorted(_float_literals(py) & MOVED_LITERALS)
            if hits:
                offenders.append(f"{py.relative_to(REPO).as_posix()}  ← {hits}")
        self.assertEqual(
            offenders, [],
            "这些数字已经搬到 configs/robots/*.yaml 了，代码里不该再有一份 —— "
            "两份来源的下场是改了一处没改另一处，而且不报错：\n  "
            + "\n  ".join(offenders))

    def test_sweep_script_reads_the_same_mesh_aabb(self):
        """标定脚本的分母必须和被标定的代码同源，否则扫出来的比例会悄悄错位。"""
        hits = _float_literals(REPO / "scripts/dev/sweep_arm_torso_params.py")
        self.assertFalse(hits & {0.139, 0.17, 0.239},
                         "sweep 脚本又自己抄了一份 mesh AABB")

    def test_capsule_model_reads_yaml(self):
        from web2robot.collision.capsule_collision import M7CapsuleModel
        proxy = robot_params("m7")["collision"]["proxy"]
        self.assertEqual(M7CapsuleModel.TORSO_HALF.tolist(),
                         [float(v) for v in proxy["torso_half"]])
        self.assertEqual(M7CapsuleModel.TIP_RADIUS, float(proxy["tip_radius"]))

    def test_presets_read_yaml(self):
        from web2robot.collision.presets import GRID, MESH_HALF, NEURAL
        coll = robot_params("m7")["collision"]
        self.assertEqual(list(MESH_HALF), list(coll["mesh_aabb_half"]["value"]))
        self.assertEqual(NEURAL, {})
        want = values(coll["arm_torso"]["routes"]["grid"])
        self.assertEqual(set(GRID), set(want))
        self.assertEqual(list(GRID["torso_half"]), list(want["torso_half"]))
        self.assertEqual(GRID["enter_thresh"], want["enter_thresh"])
        self.assertEqual(GRID["margin"], want["margin"])
        # 元组而不是 list：调用方直接把它传给 M7CapsuleModel，不该能被就地改掉
        self.assertIsInstance(GRID["torso_half"], tuple)

    def test_filter_defaults_are_the_yaml_values(self):
        """构造签名的默认值**就是** yaml 里那个数，不是"碰巧相等的第二份"。"""
        from web2robot.collision.arm_torso_filter import ArmTorsoFilter
        from web2robot.collision.dual_hand_filter import DualHandFilter
        coll = robot_params("m7")["collision"]
        for cls, key in ((ArmTorsoFilter, "arm_torso"), (DualHandFilter, "dual_hand")):
            want = values(coll[key]["defaults"])
            params = inspect.signature(cls.__init__).parameters
            self.assertTrue(set(want) <= set(params),
                            f"{cls.__name__}: yaml 里有构造函数不认的键 "
                            f"{sorted(set(want) - set(params))}")
            for k, v in want.items():
                self.assertEqual(params[k].default, v, f"{cls.__name__}.{k}")

    def test_joint_limits_and_start_config_come_from_yaml(self):
        from web2robot.robots.l3_4 import ik_config as l34_ik
        from web2robot.robots.m7 import ik_config as m7_ik
        for robot, mod in (("m7", m7_ik), ("l3_4", l34_ik)):
            jl = robot_params(robot)["ik"]["joint_limits"]
            self.assertEqual(mod.JOINT_LIMITS,
                             {s: [[float(a), float(b)] for a, b in jl[s]]
                              for s in ("left", "right")})
            self.assertEqual(list(mod.JOINT_ORDER), list(jl["order"]))
            for side in ("left", "right"):
                self.assertEqual(len(jl[side]), len(jl["order"]))
                self.assertTrue(all(len(row) == 2 and row[0] < row[1]
                                    for row in jl[side]), f"{robot}/{side}")

    def test_start_config_matches_yaml(self):
        import numpy as np

        from web2robot.robots.l3_4.config import CONFIG as L34
        from web2robot.robots.m7.config import CONFIG as M7
        for robot, cfg in (("m7", M7), ("l3_4", L34)):
            want = robot_params(robot)["ik"]["start_config"]
            for side in ("left", "right"):
                arr = cfg["start_config"][side]
                # float32 比较：CONFIG 存的是 float32（0.20 → 0.20000000298…），
                # 逐位相等要在 float32 上比，不是在 yaml 的 float64 上比
                self.assertEqual(arr.dtype, np.float32)
                self.assertTrue(np.array_equal(
                    arr, np.asarray(want[side], np.float32)), f"{robot}/{side}")
                # home_config 和 start_config 同源，别哪天只搬了一个
                self.assertTrue(np.array_equal(cfg["home_config"][side], arr))


class TestL34HasNoCollisionSection(unittest.TestCase):
    """L3.4 刻意没有 collision: 一节 —— 这是主张，不是漏写。

    碰撞过滤目前是 M7 专用的（代理盒挂 waist_pitch_link、body 名写死成
    left/right_hand_frame），那套参数一次也没在 L3.4 上标定、也没在 L3.4 上跑过。
    写一节 ``verified: false`` 的复制品进来，只会让看配置的人以为"L3.4 也支持，
    参数就在这儿"。等真支持了再加，连标定一起加。
    """

    def test_absent(self):
        self.assertNotIn("collision", robot_params("l3_4"))

    def test_the_omission_is_explained_in_the_file(self):
        txt = robot_params_path("l3_4").read_text()
        self.assertIn("刻意", txt)
        self.assertIn("collision", txt)


if __name__ == "__main__":
    unittest.main(verbosity=2)
