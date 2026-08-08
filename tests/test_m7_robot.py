"""M7 机器人定义的四条不变量。

这四条都是"只靠人看看不住"的性质，所以写成测试：

1. **接口一致性** —— ``M7Env`` 不再继承上游 ``sim.base_env.BaseEnv``（那会把机器人
   定义焊死在 EgoInfinity 的目录结构上），于是丢了"忘实现某个方法 → 实例化时
   TypeError"这个保护。这里把它补回来：import 上游 ABC（**测试可以**，``src/``
   不可以），逐个断言我们实现了每个 abstractmethod、且参数名一致。
   上游哪天给 BaseEnv 加一个方法，这里当场红，而不是跑到一半 AttributeError。

2. **hand_frame 轴向约定** —— finger+y / thumb−x / palm+z（左手），右手镜像。
   这是吃过亏的地方（2026-07-24：只验左手 + 用了退化的 r2 body，错误地得出"两只手
   同一套约定"，结果右手翻了 180°）。所以这里**两只手都验**，并且断言镜像关系。
   完整版（带 g1/r2 参照组）是 ``scripts/dev/check_handframe_convention.py``；
   这里是秒级的回归版，改 MJCF 就会当场报。

3. **sample_config 的比例参数**是按 M7 实际比例重算过的，不是抄 R2 的。抄来的
   肘 jitter (-1.6,-0.3) 会把 M7 的肘压死在 [-2.36,-1.30]（永不伸展）。钉住数值，
   免得谁"统一一下各机器人的配置"时又抄回去。

4. **上游指向 M7 资产的路径必须真的存在** —— 拼接出来的路径 grep 不到，删旧目录时
   会漏。见 ``TestUpstreamAssetPaths`` 里的说明。

跑法::

    envs/rt_env/bin/python -m unittest discover -s tests -v
"""
import subprocess
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


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
    env = {
        "PYTHONPATH": f"{REPO / 'src'}:{_upstream()}",
        "PATH": "/usr/bin:/bin",
        "HOME": str(Path.home()),
        "MUJOCO_GL": "osmesa",          # 无头机器：GLFW 起不来，egl 抛 EGLError
    }
    return subprocess.run([str(_interpreter()), "-c", code],
                          capture_output=True, text=True, env=env)


class TestBaseEnvConformance(unittest.TestCase):
    """M7Env 实现了上游 BaseEnv 的全部抽象方法（不继承，但接口相容）。"""

    def test_implements_every_abstractmethod(self):
        code = r"""
import inspect
from sim.base_env import BaseEnv
from web2robot.robots.m7 import M7Env

missing, mismatched = [], []
for name in sorted(BaseEnv.__abstractmethods__):
    impl = getattr(M7Env, name, None)
    if impl is None or not callable(impl):
        missing.append(name)
        continue
    want = list(inspect.signature(getattr(BaseEnv, name)).parameters)
    got  = list(inspect.signature(impl).parameters)
    if want != got:
        mismatched.append(f"{name}: BaseEnv{want} vs M7Env{got}")

assert not missing,     f"没实现的抽象方法: {missing}"
assert not mismatched,  f"签名不一致: {mismatched}"
assert not issubclass(M7Env, BaseEnv), "刻意不继承上游 ABC，见 robots/m7/env.py 头部说明"
print("ok", len(BaseEnv.__abstractmethods__), "个抽象方法全部相容")
"""
        r = _run(code)
        self.assertEqual(r.returncode, 0, r.stderr[-2000:])
        self.assertIn("ok", r.stdout)

    def test_env_spec_builds_upstream_dataclass(self):
        """ENV_SPEC 是纯 dict，但必须刚好能喂进上游的 RobotConfig。

        这是"只出数据不出类型"这个设计的落地检查：字段名一旦对不上，
        上游注册表 import 时就会 TypeError，这里提前把它捕住。
        """
        code = r"""
from sim.robot_config import RobotConfig
from web2robot.robots.m7 import ENV_SPEC
cfg = RobotConfig(**ENV_SPEC)
assert cfg.mjcf_path.name == "m7_mjx.xml", cfg.mjcf_path
assert set(cfg.joint_groups) == {"left", "right"}
assert len(cfg.joint_groups["left"]) == 7
assert cfg.end_effectors == {"left": "left_hand_frame", "right": "right_hand_frame"}
print("ok")
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

m = mujoco.MjModel.from_xml_path(str(P.asset("m7_scene")))
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
    R, p = d.xmat[hf].reshape(3, 3), d.xpos[hf]
    fd = R.T @ (d.xpos[bid(f"{side}_hand_mid_link2")] - p)
    td = R.T @ (d.xpos[bid(f"{side}_hand_thumb_rota_link2")] - p)
    pn = np.cross(fd, td)
    got[side] = (axis(fd), axis(td), axis(pn))

# 左手是基准约定；右手镜像 thumb 与 palm，finger 不变
assert got["left"]  == ("+y", "-x", "+z"), f"左手 {got['left']} != ('+y','-x','+z')"
assert got["right"] == ("+y", "+x", "-z"), f"右手 {got['right']} != ('+y','+x','-z')"
# 镜像关系单独断一遍：palm normal 同轴、反号
lp, rp = got["left"][2], got["right"][2]
assert lp[1] == rp[1] and lp[0] != rp[0], f"palm normal 未镜像: {lp} / {rp}"
print("ok", got)
"""
        r = _run(code)
        self.assertEqual(r.returncode, 0, r.stderr[-2000:])
        self.assertIn("ok", r.stdout)


class TestSampleConfigProportions(unittest.TestCase):
    """采样参数是按 M7 比例重算的，不是抄 R2 的。"""

    def test_elbow_jitter_and_ou_step(self):
        code = r"""
import numpy as np
from web2robot.robots.m7 import SAMPLE_CONFIG

# j3 = elbow_pitch。R2 抄来的 (-1.6,-0.3) 会把 M7 的肘压死在 [-2.36,-1.30]
assert SAMPLE_CONFIG["proximal_jitter"][3] == (-0.5, 0.5), \
    SAMPLE_CONFIG["proximal_jitter"][3]
# ou_step = R2 的向量按臂展比 1.00/1.28 缩放（保留 R2 的各向异性）
assert np.allclose(SAMPLE_CONFIG["ou_step"], [0.031, 0.027, 0.023]), \
    SAMPLE_CONFIG["ou_step"]
print("ok")
"""
        r = _run(code)
        self.assertEqual(r.returncode, 0, r.stderr[-2000:])
        self.assertIn("ok", r.stdout)


class TestUpstreamAssetPaths(unittest.TestCase):
    """上游那边每一处指向 M7 资产的路径，都必须真的存在。

    这条是拿代价换来的（2026-08-09）：迁移时我用 ``grep -rn "robots\\.m7\\|robots/m7"``
    找上游的引用，漏掉了 ``kinematics/wrist_ik.py`` 里的
    ``_ROBOTS_DIR / "m7" / "m7.xml"`` —— 路径是**拼出来的**，源码里根本没有
    "robots/m7" 这个字样。删掉上游旧目录后，端到端在建 IK 串链时当场
    ``FileNotFoundError``，而迁移完成前跑的那次端到端是绿的（那会儿旧目录还在，
    它悄悄用的是旧文件）。

    所以这里不靠 grep，靠**把 config 造出来看路径存不存在** —— 拼接的路径也躲不掉。
    """

    def test_ik_config_mjcf_exists(self):
        code = r"""
from kinematics.wrist_ik import RobotIKConfig
from web2robot.robots.m7 import MJCF_PATH

for side in ("left", "right"):
    cfg = RobotIKConfig.m7(side)
    assert cfg.mjcf_path.is_file(), f"{side}: IK 用的 MJCF 不存在 -> {cfg.mjcf_path}"
    assert cfg.mjcf_path == MJCF_PATH, \
        f"{side}: IK 走的不是我方资产 {cfg.mjcf_path} != {MJCF_PATH}"
    # IK 串链的根/末端必须在这个 MJCF 里真的有
    import mujoco
    m = mujoco.MjModel.from_xml_path(str(cfg.mjcf_path))
    for link in (cfg.root_link_name, cfg.end_link_name):
        assert mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, link) >= 0, \
            f"{side}: MJCF 里没有 body {link}"
print("ok")
"""
        r = _run(code)
        self.assertEqual(r.returncode, 0, r.stderr[-2000:])
        self.assertIn("ok", r.stdout)

    def test_no_stale_m7_paths_left_upstream(self):
        """上游不该再有指向已删目录的 m7 资产路径，也不该留重复的 m7 脚本副本。"""
        up = _upstream()
        stale = [p for p in ((up / "robots" / "m7"), (up / "sim" / "robots" / "m7"),
                             (up / "scripts" / "generate_m7_mjx.py"))
                 if p.exists()]
        self.assertEqual(stale, [], f"上游残留（应已迁到 web2robot）: {stale}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
