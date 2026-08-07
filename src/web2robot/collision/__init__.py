"""第⑤步 —— 自碰撞检测与纠正（代理几何 + 后处理）。

为什么是"后处理 + 代理几何"而不是用 MuJoCo 现成的碰撞设施：M7 的碰撞 geom
全部 disabled（``contype``/``conaffinity`` = 0），上游 ``models/collision.py``
的 ``CollisionFilter`` 靠 MuJoCo contacts、而且在构造上就把躯干排除了
（``shared = left ∩ right`` 被剔除），对 M7 查不到任何东西。所以这里另写一套
代理几何 + 有符号距离的版本，不改机器人模型。

- ``capsule_collision`` —— 几何代理：躯干**各向异性盒** + 手臂骨段胶囊
  （``M7CapsuleModel``）、每手 11 个球（``HandSphereModel``）。
- ``arm_torso_filter`` —— 臂/指-躯干纠正，只动犯规那一侧的手臂。
- ``dual_hand_filter`` —— 手-手纠正，比躯干那条**更保守**。

保守策略是量出来的，不是口味问题：双手相触多数是有意的双手抓握（实测交叠最深
仅 −2.5 cm，画面确认全是合抱罐子、胸前双手操作），强行推开会毁掉抓握；而手臂
穿进躯干永远非法。手指关节**从不**被改写 —— 指尖只作为触发条件，纠正由手臂
承担、整只手随手腕带出。

两个滤波器都不 import 上游任何东西：机器人构型是以 ``robot_cfg`` 字典传进来的
（只用到 ``env_cls`` / ``scene_path`` / ``start_config`` 三个键），所以这一层
换机器人不用改代码。
"""
from .capsule_collision import M7CapsuleModel, HandSphereModel
from .arm_torso_filter import ArmTorsoFilter
from .dual_hand_filter import DualHandFilter

__all__ = ["M7CapsuleModel", "HandSphereModel", "ArmTorsoFilter", "DualHandFilter"]
