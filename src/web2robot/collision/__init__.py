"""第⑤步 —— 自碰撞检测与纠正（代理几何 + 后处理）。

为什么是"后处理 + 代理几何"而不是用 MuJoCo 现成的碰撞设施 —— 注意**不是**因为
M7 没有碰撞 geom（2026-08-11 更正了这个说法）。实测：``m7.xml``（IK / 渲染 /
本模块加载的那个模型）196 个 geom 里有 98 个是开碰撞的，每个 link 一个可视 geom
（``contype=0``）+ 一个碰撞 mesh geom。全关的是 ``m7_mjx.xml`` —— 那是我们自己
用 ``scripts/dev/generate_m7_mjx.py`` 生成的 arms-only FK 训练模型，它不需要
contacts，两者别搞混。

真正的原因是上游 ``models/collision.py::CollisionFilter._build_geom_sets`` 圈错了
范围，它在 M7 上**结构性地什么也看不见**（实测 216 帧跨臂 contact = 0）：

1. ``shared = left_chain ∩ right_chain`` 被剔除 → 三个 waist body（含 IK 链根
   ``waist_pitch_link``）不在任何一侧的集合里，臂-躯永远不会被统计；
2. ``chain_to_root(hand_frame)`` 只走末端到根的那**一条链** → 手掌和手指的 108 个
   geom 一个都不在集合里。而 fill_jar 里真实的跨臂接触**全部**是指-指接触。

代理几何这条路仍然是主路，但理由是"梯度"而不是"没得用"：MuJoCo 只在进入 margin
之后才产生 contact，没有远场的有符号距离，梯度下降在还没碰上时拿不到方向；代理
的 SDF 处处连续，还能按部位给不同的保守度（见下）。开销不是理由 —— 实测 mesh
``mj_forward`` 0.91 ms vs 代理 SDF 0.50 ms，只差 1.8 倍。

官方 contacts 现在的定位是**独立复核**（``scripts/dev/audit_mujoco_contacts.py``，
只报告不改轨迹）。要用它必须先排掉 6 组结构性自重叠 —— URDF 转出来的凸包在静息位
就互相插着（q=0 时 ncon=10、最深 3.1 cm），否则 "ncon > 0" 恒为真。

- ``capsule_collision`` —— 几何代理：躯干**各向异性盒** + 手臂骨段胶囊
  （``M7CapsuleModel``）、每手 11 个球（``HandSphereModel``）。
- ``arm_torso_filter`` —— 臂/指-躯干纠正，只动犯规那一侧的手臂。
- ``dual_hand_filter`` —— 手-手纠正，比躯干那条**更保守**。
- ``presets`` —— 按底座求解路线（``--root_solver neural`` / ``grid``）分开的
  标定参数。两条路线底座落点不同、手臂贴身的方式也不同，一套余量伺候不了两边；
  ``neural`` 那组是**空的**，即保持历史行为逐位不变。

保守策略是量出来的，不是口味问题：双手相触多数是有意的双手抓握（实测交叠最深
仅 −2.5 cm，画面确认全是合抱罐子、胸前双手操作），强行推开会毁掉抓握；而手臂
穿进躯干永远非法。手指关节**从不**被改写 —— 指尖只作为触发条件，纠正由手臂
承担、整只手随手腕带出。

已知残留（2026-08-11 审计出来的，写在这里免得下次又当成新问题查）：
``ArmTorsoFilter`` 的 ``enter_thresh=0.04`` 意味着它**只管深过 4 cm 的**，浅擦不动；
且深帧不保证收敛 —— fill_jar 左臂 71 个超阈帧里 fixed 53 / remaining 18，
纠正后 MuJoCo 仍在 50/216 帧看到真实网格穿透、最深 8.07 cm（纠正前 99 帧 / 14.19 cm）。
右臂最深 2.36 cm 全在阈值下，所以整段没被动过。这是"保真项 ``w_ee=60`` 压着
推出项 ``w_pen=20`` + 60 步预算"的直接后果，不是检测漏了。

两个滤波器都不 import 上游任何东西：机器人构型是以 ``robot_cfg`` 字典传进来的
（只用到 ``env_cls`` / ``scene_path`` / ``start_config`` 三个键），所以这一层
换机器人不用改代码。
"""
from .capsule_collision import M7CapsuleModel, HandSphereModel
from .arm_torso_filter import ArmTorsoFilter
from .dual_hand_filter import DualHandFilter
from .presets import arm_torso_preset

__all__ = ["M7CapsuleModel", "HandSphereModel", "ArmTorsoFilter", "DualHandFilter",
           "arm_torso_preset"]
