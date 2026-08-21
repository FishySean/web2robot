"""L3.4（rel3_4）机器人定义的不变量。

L3.4 的上肢和 M7 逐位同构，于是 ``robots/l3_4/`` 里的限位表、start_config、采样参数
和 ``robots/m7/`` 数值相同。**两个包刻意不共享一行代码**（一台机器人的定义不该依赖
另一台的模块存在），代价就是两份表可能悄悄漂 —— 这个文件就是顶住这一条的地方。

关键取舍：**拿各自的 MJCF 当唯一真相，不拿 M7 的表当参照**。
如果这里写 ``assert L34.JOINT_LIMITS == M7.JOINT_LIMITS``，那么等哪天真拿到不同批次的
L3.4（限位变了），测试会红在"和 M7 不一样"上——而那时候不一样才是对的。所以断言的是
``表 == l3_4.xml 里那个关节的 range``：改了资产必须同步改表，换了机器人只要资产对就自动对。

六条：

1. **接口一致性** —— ``L34Env`` 不继承上游 ``sim.base_env.BaseEnv``（那会把机器人定义
   焊死在 EgoInfinity 的目录结构上），于是丢了"忘实现某个方法 → 实例化 TypeError"这个
   保护。这里补回来：import 上游 ABC（**测试可以**，``src/`` 不可以）逐个断言。
2. **表 vs MJCF** —— 7×2 臂限位逐位相等；12 个手指限位被 MJCF 的 range 包住
   （右手 ``index_bend`` 是 ±0.175 而表用较紧的 ±0.174，这是**故意**的，见
   ``hand_mapping.py``，所以这条是"包含且相差 <2e-3"而不是"相等"）。
3. **关节全覆盖** —— arm(14) ∪ finger(24) ∪ locked(17) 必须恰好是 MJCF 的全部 55 个
   关节。少一个就意味着有个自由度既没人控制、也没被锁住 —— 那种自由度会在
   ``mj_forward`` 里自己动起来（重力、或者上层直接写 qpos 的副作用）。
4. **锁死是真的锁住** —— 不是"没人去写它们"。``set_arm_joints`` / ``set_finger_joints``
   之后、以及外部直接篡改 ``data.qpos`` 再走一次之后，17 个自由度都必须还在原位。
5. **hand_frame 轴向约定** —— finger+y / thumb−x / palm+z（左手），右手镜像。
   吃过亏的地方（2026-07-24 在 M7 上：只验左手 → 右手翻了 180°），所以**两只手都验**
   并且单独断言镜像关系。
6. **不反向依赖 M7** —— ``robots/l3_4/`` 里不许出现 ``robots.m7``；上游资产路径必须真的
   存在（拼接出来的路径 grep 不到，M7 迁移时正是这么漏的，见 ``test_m7_robot.py``）。

跑法::

    envs/rt_env/bin/python -m unittest tests.test_l3_4_robot -v
"""
import ast
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PKG = REPO / "src" / "web2robot" / "robots" / "l3_4"


def _interpreter() -> Path:
    import sys
    sys.path.insert(0, str(REPO / "src"))
    from web2robot.paths import P
    return P.env("retarget")


def _upstream() -> Path:
    import sys
    sys.path.insert(0, str(REPO / "src"))
    from web2robot.paths import P
    return P.root("egoinfinity") / "retarget"


def _run(code: str):
    """在 rt_env 里跑一段代码；PYTHONPATH 同时含我方 src 和上游 retarget/。"""
    import subprocess
    env = {
        "PYTHONPATH": f"{REPO / 'src'}:{_upstream()}",
        "PATH": "/usr/bin:/bin",
        "HOME": str(Path.home()),
        "MUJOCO_GL": "osmesa",          # 无头机器：GLFW 起不来，egl 抛 EGLError
    }
    return subprocess.run([str(_interpreter()), "-c", code],
                          capture_output=True, text=True, env=env)


class TestBaseEnvConformance(unittest.TestCase):
    """L34Env 实现了上游 BaseEnv 的全部抽象方法（不继承，但接口相容）。"""

    def test_implements_every_abstractmethod(self):
        code = r"""
import inspect
from sim.base_env import BaseEnv
from web2robot.robots.l3_4 import L34Env

missing, mismatched = [], []
for name in sorted(BaseEnv.__abstractmethods__):
    impl = getattr(L34Env, name, None)
    if impl is None or not callable(impl):
        missing.append(name)
        continue
    want = list(inspect.signature(getattr(BaseEnv, name)).parameters)
    got  = list(inspect.signature(impl).parameters)
    if want != got:
        mismatched.append(f"{name}: BaseEnv{want} vs L34Env{got}")

assert not missing,    f"没实现的抽象方法: {missing}"
assert not mismatched, f"签名不一致: {mismatched}"
assert not issubclass(L34Env, BaseEnv), "刻意不继承上游 ABC，见 robots/l3_4/env.py 头部说明"
print("ok", len(BaseEnv.__abstractmethods__), "个抽象方法全部相容")
"""
        r = _run(code)
        self.assertEqual(r.returncode, 0, r.stderr[-2000:])
        self.assertIn("ok", r.stdout)

    def test_env_spec_builds_upstream_dataclass(self):
        """ENV_SPEC 是纯 dict，但必须刚好能喂进上游的 RobotConfig。"""
        code = r"""
from sim.robot_config import RobotConfig
from web2robot.robots.l3_4 import ENV_SPEC

cfg = RobotConfig(**ENV_SPEC)
assert cfg.mjcf_path.name == "l3_4_mjx.xml", cfg.mjcf_path
assert cfg.mjcf_path.is_file(), f"MJX 模型不存在 -> {cfg.mjcf_path}"
assert set(cfg.joint_groups) == {"left", "right"}
assert len(cfg.joint_groups["left"]) == 7
assert cfg.end_effectors == {"left": "left_hand_frame", "right": "right_hand_frame"}
print("ok")
"""
        r = _run(code)
        self.assertEqual(r.returncode, 0, r.stderr[-2000:])
        self.assertIn("ok", r.stdout)


class TestTablesMatchMJCF(unittest.TestCase):
    """限位表的真源是 l3_4.xml，不是 robots/m7/ 的同名表。"""

    def test_arm_limits_equal_mjcf_range(self):
        code = r"""
import mujoco, numpy as np
from web2robot.robots.l3_4 import MJCF_PATH, ENV_SPEC
from web2robot.robots.l3_4.ik_config import JOINT_LIMITS, ROOT_LINK_NAME, END_LINK_NAME

m = mujoco.MjModel.from_xml_path(str(MJCF_PATH))
bad = []
for side, names in ENV_SPEC["joint_groups"].items():
    assert len(JOINT_LIMITS[side]) == len(names) == 7, side
    for (lo, hi), jname in zip(JOINT_LIMITS[side], names):
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jname)
        assert jid >= 0, f"MJCF 里没有关节 {jname}"
        r_lo, r_hi = m.jnt_range[jid]
        # MJCF 是 6 位小数写出来的，表是 2 位 —— 允许 5e-3 的书写差，不允许方向或量级差
        if abs(lo - r_lo) > 5e-3 or abs(hi - r_hi) > 5e-3:
            bad.append(f"{jname}: 表({lo},{hi}) vs MJCF({r_lo:.4f},{r_hi:.4f})")
assert not bad, "限位表和 MJCF 不一致（真源是 MJCF，改资产必须同步改表）:\n" + "\n".join(bad)

# 串链根/末端必须是 MJCF 里真的 body
for name in [ROOT_LINK_NAME, *END_LINK_NAME.values()]:
    assert mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name) >= 0, f"没有 body {name}"
print("ok")
"""
        r = _run(code)
        self.assertEqual(r.returncode, 0, r.stderr[-2000:])
        self.assertIn("ok", r.stdout)

    def test_hand_limits_inside_mjcf_range(self):
        """手指限位要被 MJCF 的 range 包住 —— 命令值不能越限。

        不是"相等"：右手 ``hand_index_bend_joint`` 的 range 是 ±0.175 而表里写 ±0.174
        （两侧较紧的那个），这样一张表喂两只手都不越限。所以断言"包含 + 相差 <2e-3"，
        既允许这个 0.06° 的故意收紧，又不允许谁把某一行改成宽松 10 倍。
        """
        code = r"""
import mujoco
from web2robot.robots.l3_4 import MJCF_PATH, HAND_JOINT_SPEC
from web2robot.robots.l3_4.env import _FINGER_JOINT_NAMES

m = mujoco.MjModel.from_xml_path(str(MJCF_PATH))
assert len(HAND_JOINT_SPEC) == 12, len(HAND_JOINT_SPEC)
bad, checked = [], 0
for spec in HAND_JOINT_SPEC:
    short = spec["robot_name"]
    suffix = _FINGER_JOINT_NAMES[short]        # 缺了就 KeyError，正是我们要的
    lo, hi = spec["limit"]
    for side in ("left", "right"):
        jname = f"{side}_{suffix}"
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jname)
        assert jid >= 0, f"MJCF 里没有关节 {jname}"
        r_lo, r_hi = m.jnt_range[jid]
        checked += 1
        if lo < r_lo - 1e-9 or hi > r_hi + 1e-9:
            bad.append(f"{jname}: 表({lo},{hi}) 越出 MJCF({r_lo:.4f},{r_hi:.4f})")
        elif (r_lo - lo) > 2e-3 or (hi - r_hi) < -2e-3:
            bad.append(f"{jname}: 表({lo},{hi}) 比 MJCF({r_lo:.4f},{r_hi:.4f}) 保守过多")
assert not bad, "\n".join(bad)
assert checked == 24, checked
print("ok", checked, "个手指关节限位受检")
"""
        r = _run(code)
        self.assertEqual(r.returncode, 0, r.stderr[-2000:])
        self.assertIn("ok", r.stdout)

    def test_every_mjcf_joint_is_controlled_or_locked(self):
        """arm ∪ finger ∪ locked == MJCF 的全部关节。没有第四类。

        漏掉一个自由度不会报错 —— 它只会在 ``mj_forward`` 里自己动起来（重力，或上层
        直接写 qpos 的副作用），然后在渲出来的画面里表现为"某个零件莫名歪着"。
        """
        code = r"""
import mujoco
from web2robot.robots.l3_4 import MJCF_PATH, ENV_SPEC, LOCKED_JOINTS
from web2robot.robots.l3_4.env import _FINGER_JOINT_NAMES

m = mujoco.MjModel.from_xml_path(str(MJCF_PATH))
all_joints = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(m.njnt)}

arm    = {j for names in ENV_SPEC["joint_groups"].values() for j in names}
finger = {f"{side}_{suf}" for suf in _FINGER_JOINT_NAMES.values()
          for side in ("left", "right")}
locked = set(LOCKED_JOINTS)

assert len(arm) == 14,    len(arm)
assert len(finger) == 24, len(finger)
assert len(locked) == 17, len(locked)
overlap = (arm & finger) | (arm & locked) | (finger & locked)
assert not overlap, f"一个关节同时属于两类: {sorted(overlap)}"

covered = arm | finger | locked
missing = all_joints - covered      # 既不控制也不锁 → 会自己动
ghost   = covered - all_joints      # 表里有、MJCF 里没有 → 静默失效
assert not missing, f"没人管的自由度: {sorted(missing)}"
assert not ghost,   f"MJCF 里不存在的关节名: {sorted(ghost)}"
assert len(all_joints) == 55, f"MJCF 关节数变了: {len(all_joints)}（原 55）"
print("ok 14 arm + 24 finger + 17 locked =", len(all_joints))
"""
        r = _run(code)
        self.assertEqual(r.returncode, 0, r.stderr[-2000:])
        self.assertIn("ok", r.stdout)


class TestLockedDofs(unittest.TestCase):
    """腿/腰/颈是"锁死"不是"没人写" —— 被外部篡改后也必须回到锁定值。"""

    def test_held_through_writes_and_tampering(self):
        code = r"""
import numpy as np
from web2robot.robots.l3_4 import L34Env, CONFIG, LOCKED_JOINTS, HAND_JOINT_NAMES
import mujoco

env = L34Env(start_config=CONFIG["start_config"])
adrs = {}
for j, val in LOCKED_JOINTS.items():
    jid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, j)
    assert jid >= 0, f"MJCF 里没有被锁的关节 {j}"
    adrs[j] = (int(env.model.jnt_qposadr[jid]), float(val))
assert len(adrs) == 17, len(adrs)

def check(tag):
    off = [f"{j}={env.data.qpos[a]:+.4f}(want {v:+.2f})"
           for j, (a, v) in adrs.items() if abs(env.data.qpos[a] - v) > 1e-9]
    assert not off, f"{tag}: 锁定自由度跑掉了 -> {off}"

check("reset")
env.set_arm_joints("left",  np.array([0.3, 0.4, -0.2, -1.2, 0.1, 0.2, -0.3]))
env.set_arm_joints("right", np.array([-0.3, -0.4, 0.2, -1.2, -0.1, -0.2, 0.3]))
check("set_arm_joints")

names = [f"{s}_{n}" for s in ("left", "right") for n in HAND_JOINT_NAMES]
env.set_finger_joints(np.full(len(names), 0.5), names)
check("set_finger_joints")

# 模拟上层（碰撞过滤 / 渲染）直接篡改 qpos —— 这是"锁死"和"没人写"的区别所在
for a, _ in adrs.values():
    env.data.qpos[a] = 0.37
for side in ("left", "right"):
    env.set_arm_joints(side, np.asarray(CONFIG["start_config"][side], dtype=np.float64))
check("外部篡改后")

# 顺带：回到 start_config 后两侧手腕左右镜像（y 反号、x/z 相同）—— 锁定值被篡改成 0.37
# 又被按回 0 是这条能成立的前提（腰歪了手腕就不镜像了）
lp, _ = env.get_wrist_pose("left")
rp, _ = env.get_wrist_pose("right")
assert abs(lp[1] + rp[1]) < 1e-6, f"y 未镜像: {lp} / {rp}"
assert abs(lp[0] - rp[0]) < 1e-6 and abs(lp[2] - rp[2]) < 1e-6, f"x/z 不同: {lp} / {rp}"
assert lp[1] > 0.1, f"左手腕应在 +y 侧: {lp}"
print("ok 17 个锁定自由度全程按住")
"""
        r = _run(code)
        self.assertEqual(r.returncode, 0, r.stderr[-2000:])
        self.assertIn("ok", r.stdout)


class TestHandFrameConvention(unittest.TestCase):
    """finger+y / thumb−x / palm+z，且两只手镜像。两只手都验，永远不只验一侧。"""

    def test_both_hands(self):
        code = r"""
import numpy as np, mujoco
from web2robot.paths import P

m = mujoco.MjModel.from_xml_path(str(P.asset("l3_4_scene")))
d = mujoco.MjData(m)
kid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "home")
if kid >= 0:
    mujoco.mj_resetDataKeyframe(m, d, kid)
mujoco.mj_forward(m, d)
bid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)

def axis(v):
    i = int(np.argmax(np.abs(v)))
    return ("+" if v[i] > 0 else "-") + "xyz"[i]

got = {}
for side in ("left", "right"):
    hf = bid(f"{side}_hand_frame")
    assert hf >= 0, f"没有 body {side}_hand_frame"
    R, p = d.xmat[hf].reshape(3, 3), d.xpos[hf]
    fd = R.T @ (d.xpos[bid(f"{side}_hand_mid_link2")] - p)
    td = R.T @ (d.xpos[bid(f"{side}_hand_thumb_rota_link2")] - p)
    got[side] = (axis(fd), axis(td), axis(np.cross(fd, td)))

assert got["left"]  == ("+y", "-x", "+z"), f"左手 {got['left']} != ('+y','-x','+z')"
assert got["right"] == ("+y", "+x", "-z"), f"右手 {got['right']} != ('+y','+x','-z')"
lp, rp = got["left"][2], got["right"][2]
assert lp[1] == rp[1] and lp[0] != rp[0], f"palm normal 未镜像: {lp} / {rp}"
print("ok", got)
"""
        r = _run(code)
        self.assertEqual(r.returncode, 0, r.stderr[-2000:])
        self.assertIn("ok", r.stdout)


class TestUpstreamWiring(unittest.TestCase):
    """框架侧的五处接线都真的通了（注册表 / IK / 手部重定向器）。"""

    def test_registry_has_l3_4_alongside_m7(self):
        code = r"""
from sim.robots import ROBOT_CONFIGS, SAMPLE_CONFIGS, ENV_CONFIGS

for name, table in (("ROBOT_CONFIGS", ROBOT_CONFIGS),
                    ("SAMPLE_CONFIGS", SAMPLE_CONFIGS),
                    ("ENV_CONFIGS", ENV_CONFIGS)):
    assert "l3_4" in table, f"{name} 里没有 l3_4"
    assert "m7" in table,   f"{name} 里 m7 不见了"        # 并列，不是替换
# 两台机器人是两个独立对象（不是同一个 dict 的别名）
assert ROBOT_CONFIGS["l3_4"] is not ROBOT_CONFIGS["m7"]
assert ROBOT_CONFIGS["l3_4"]["ik_robot"] == "l3_4"
assert ROBOT_CONFIGS["m7"]["ik_robot"] == "m7"
print("ok")
"""
        r = _run(code)
        self.assertEqual(r.returncode, 0, r.stderr[-2000:])
        self.assertIn("ok", r.stdout)

    def test_ik_config_and_hand_retargeter(self):
        """``RobotIKConfig.l3_4`` 必须存在（dispatch 是 getattr，名字错了才在跑到一半炸），
        且指向我方资产、限位和纯数据表一致。"""
        code = r"""
import mujoco
from kinematics.wrist_ik import RobotIKConfig
from kinematics.wilor_retargeter import get_wilor_hand_retargeter
from web2robot.robots.l3_4 import MJCF_PATH, HAND_JOINT_NAMES
from web2robot.robots.l3_4.ik_config import JOINT_LIMITS

for side in ("left", "right"):
    cfg = RobotIKConfig.l3_4(side)
    assert cfg.mjcf_path.is_file(), f"{side}: IK 用的 MJCF 不存在 -> {cfg.mjcf_path}"
    assert cfg.mjcf_path == MJCF_PATH, \
        f"{side}: IK 走的不是我方资产 {cfg.mjcf_path} != {MJCF_PATH}"
    m = mujoco.MjModel.from_xml_path(str(cfg.mjcf_path))
    for link in (cfg.root_link_name, cfg.end_link_name):
        assert mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, link) >= 0, \
            f"{side}: MJCF 里没有 body {link}"
    lim = cfg.joint_limits.cpu().numpy() if hasattr(cfg.joint_limits, "cpu") else cfg.joint_limits
    assert lim.shape == (7, 2), lim.shape
    for (a, b), (c, dd) in zip(lim.tolist(), JOINT_LIMITS[side]):
        assert abs(a - c) < 1e-6 and abs(b - dd) < 1e-6, f"{side}: {(a,b)} != {(c,dd)}"

    rt = get_wilor_hand_retargeter("l3_4", side)
    assert rt is not None, f"{side}: 没建出手部重定向器"
    # 短键（不带 left_/right_ 前缀），顺序即 q_fingers 的列顺序；贴前缀是调用方的事
    assert len(rt.joint_names) == 12, f"{side}: {len(rt.joint_names)} 个手指关节（应 12）"
    assert rt.joint_names == HAND_JOINT_NAMES, \
        f"{side}: 重定向器的关节顺序和纯数据表不一致\n{rt.joint_names}\n{HAND_JOINT_NAMES}"
print("ok")
"""
        r = _run(code)
        self.assertEqual(r.returncode, 0, r.stderr[-2000:])
        self.assertIn("ok", r.stdout)

    def test_cli_accepts_l3_4(self):
        """``--robot l3_4`` 在 test.py / train.py 的 choices 里；且默认（不传）仍是 m7 之外
        的行为不变 —— choices 只增不改。"""
        up = _upstream()
        for rel in ("scripts/test.py", "scripts/train.py"):
            src = (up / rel).read_text()
            self.assertIn('"l3_4"', src, f"{rel}: --robot choices 里没有 l3_4")
            self.assertIn('"m7"', src, f"{rel}: m7 不见了")


class TestNoReverseDependency(unittest.TestCase):
    """``robots/l3_4/`` 不许 import ``robots/m7/``，也不许 import 上游。

    同构不等于可以互相依赖：一旦 l3_4 从 m7 拿表，"改 M7 的关节名"就会隔着一台机器人
    炸，而且两台机器人谁是真源变得说不清。上游那条由
    ``test_module_boundaries.py`` 全局管，这里只管反向依赖这一条。
    """

    def test_no_import_of_m7(self):
        offenders = []
        for py in sorted(PKG.glob("*.py")):
            tree = ast.parse(py.read_text())
            for node in ast.walk(tree):
                mods = []
                if isinstance(node, ast.Import):
                    mods = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    mods = [node.module]
                for mod in mods:
                    if "robots.m7" in mod or mod.endswith(".m7"):
                        offenders.append(f"{py.name}:{node.lineno} -> {mod}")
        self.assertEqual(offenders, [],
                         "l3_4 反向依赖了 m7（真源应是各自的 MJCF）:\n" + "\n".join(offenders))

    def test_assets_are_generated_not_handwritten(self):
        """资产目录必须有生成脚本，且 MJCF 带"别手改"的生成头。"""
        script = REPO / "scripts" / "dev" / "build_l3_4_assets.py"
        self.assertTrue(script.is_file(), f"生成脚本不见了: {script}")
        import sys
        sys.path.insert(0, str(REPO / "src"))
        from web2robot.paths import P
        for key in ("l3_4_mjcf", "l3_4_mjx", "l3_4_urdf", "l3_4_scene"):
            self.assertTrue(P.asset(key).is_file(), f"{key} 不存在 -> {P.asset(key)}")
        head = P.asset("l3_4_urdf").read_text()[:2000]
        self.assertIn("build_l3_4_assets.py", head,
                      "URDF 头里没写生成脚本 —— 手改过的资产会在下次重建时被静默覆盖")


if __name__ == "__main__":
    unittest.main(verbosity=2)
