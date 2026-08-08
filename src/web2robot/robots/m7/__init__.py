"""M7 机器人定义 —— MJCF/关节名/末端帧/IK 种子姿态，**不 import 任何重定向框架**。

这一层回答的是"这台机器人长什么样"，跟用哪个重定向框架无关，所以也不该知道框架的
存在。**加一台新机器人 = 在 ``robots/`` 下新建一个子包，别的模块一行不用改。**
框架侧怎么认识它是框架侧的事（EgoInfinity 的注册在它自己的 ``sim/robots/__init__.py``，
见 ``external/patches/``）。这一条由 ``tests/test_module_boundaries.py`` 钉住。

为此这一层只出**数据和自己的类**，不出上游类型：

- ``CONFIG`` —— 纯 dict（``env_cls`` 是我们自己的类，其余是字符串/数组/数字）
- ``ENV_SPEC`` —— 纯 dict，而不是上游的 ``RobotConfig`` 实例；要 dataclass 的人
  自己 ``RobotConfig(**ENV_SPEC)``
- ``M7Env`` —— 实现上游 ``BaseEnv`` 的接口但不继承它（``BaseEnv`` 是纯抽象类，
  全仓库无 ``isinstance`` 检查；继承换成了 ``tests/test_m7_robot.py`` 里的一致性
  断言，报错比继承更早）

## M7 的两条硬约定，改这一层前先看

**1. hand_frame 的轴向必须是 finger+y / thumb−x / palm+z（左手），右手镜像。**
这是吃过亏的地方：2026-07-24 之前的检查脚本只验左手、还用了退化的 r2 body，
错误地得出"两只手同一套约定"，结果 M7 两只手被建成完全一样 —— 右手手掌/拇指
翻了 180°。g1 和 r2 这两台已知正确的机器人都是**左右镜像 palm normal** 的。
``scripts/dev/check_handframe_convention.py`` 现在两只手都验，并拿 g1/r2 当参照
断言镜像关系。**永远不要只验一侧。**

**2. ``sample_config.py`` 的比例参数不能照抄别的机器人。**
它的方案抄自 robonaut2，但其中两个参数按 M7 的实际比例重算过（肘 jitter 和
ou_step）—— R2 的 (-1.6,-0.3) 会把 M7 的肘压到 [-2.36,-1.30]（死弯、永不伸展），
躯干→手腕距离只有臂展的 50%；重算后 60%，重训后手臂折叠和穿模都改善。
``SAMPLE_CONFIG`` 只有训练期的合成轨迹采样器（``train.py``）用，推理/验证路径
（``test.py``）不碰它。

## 验证脚本

```bash
scripts/dev/verify_m7_mjx_fk.py            # m7_mjx.xml 的 FK 与 m7.xml 是否一致（go/no-go）
scripts/dev/check_handframe_convention.py  # 上面那条 hand_frame 约定，两只手都验
```
"""
from web2robot.robots.m7.config import CONFIG, ENV_SPEC
from web2robot.robots.m7.env import M7Env, _MJCF_PATH, _SCENE_PATH
from web2robot.robots.m7.sample_config import SAMPLE_CONFIG

#: 全身 MJCF（``assets/robots/m7/m7.xml``）。导出它是因为框架侧还有第二处需要它：
#: 上游 ``kinematics/wrist_ik.py`` 的 ``RobotIKConfig.m7`` 要拿它建 IK 串链
#: （pytorch_kinematics）。别让框架侧自己拼路径 —— 迁移时正是那一处 ``_ROBOTS_DIR /
#: "m7" / "m7.xml"`` 因为不含 "robots/m7" 字样躲过了 grep，删掉旧目录后端到端当场
#: FileNotFoundError。由 ``tests/test_m7_robot.py::TestUpstreamAssetPaths`` 钉住。
MJCF_PATH  = _MJCF_PATH
#: 带地面/光照的场景（``m7_scene.xml``），渲染与 hand_frame 检查用。
SCENE_PATH = _SCENE_PATH

__all__ = ["CONFIG", "ENV_SPEC", "M7Env", "SAMPLE_CONFIG", "MJCF_PATH", "SCENE_PATH"]
