"""L3.4 五指手 12 DoF ← MANO 关节的映射表。**纯数据，不 import 任何重定向框架。**

框架侧自己把下面的 dict 包成它的 dataclass（EgoInfinity 那边是
``RobotHandConfig(joints=[JointMapping(**d) for d in HAND_JOINT_SPEC])``，
见 ``external/patches/`` 对 ``kinematics/wilor_retargeter.py`` 的改动）。出 dict 而不是
上游类型顺带当检查：上游哪天给 ``JointMapping`` 加了必填字段，会在 import 时就
TypeError，而不是悄悄用默认值跑出错的手型。

## 先说清一件事：这只手**不是** 11 自由度的另一款

厂家 URDF 里 mesh 路径长这样::

    package://robot_control/description/l3.4/xhand/meshes/left_hand_link.STL
                                             ^^^^^

那个 ``xhand`` 是**目录名**，不是型号。实际量下来（``mujoco`` 读 ``l3_4.xml``）：

* 每只手 12 个 hinge：``hand_thumb_bend`` + ``hand_thumb_rota_joint1/2``，
  食指 ``hand_index_bend`` + ``hand_index_joint1/2``，中/无名/小指各
  ``joint1/joint2`` —— 关节名和 M7 那只手**逐字相同**；
* 12 个限位也逐位相同（唯一差别：右手 ``index_bend`` 是 ±0.175 而左手 ±0.174，
  厂家 URDF 里的四舍五入差，0.06°）；
* 95 个同名 link 里 92 个 mass/inertia/COM 逐位相同 —— 是同一批零件。

所以映射表不需要重新设计，MANO→12 DoF 这套原样成立。**如果哪天真换成 11 自由度的
手，那 12 行会在 import 时报 KeyError**（``env.py::_FINGER_JOINT_NAMES`` 找不到关节），
不会静默跑错。

## 映射依据（每一行为什么这么连）

MANO 那侧的特征数组是 (15, 2)，顺序 index[mcp,pip,dip] middle[...] pinky[...]
ring[...] thumb[cmc,mcp,ip]，每项给 (flex, abd)。下面 ``mano_joint`` 用的是这个
**0..14** 编号，*不是* 21 个关键点的编号（上游 ``wilor_retargeter.py`` 顶部有说明）。

机器人这侧每指只有 2 个 curl（``joint1`` = 近节，``joint2`` = 中节），人手每指有
3 段（mcp/pip/dip）。所以：

* ``*_mcp`` ← MANO 的 mcp（0/3/6/9），``*_pip`` ← MANO 的 pip（1/4/7/10）；
  **dip 丢掉** —— 机械上没有对应的自由度，硬塞进 pip 会让指尖过弯。
* 拇指 3 个自由度对上 MANO 拇指的 cmc/mcp/ip（12/13/14）：``thumb_bend`` 是对掌
  （不是屈曲），拿 cmc 的 flex 驱动，因为人手的对掌主要发生在 cmc。
* 食指 ``index_abd`` 是唯一的外展自由度，取 MANO 食指的 abd，``scale=0.5``：
  机器人只有 ±0.174 rad（±10°）而人手食指外展远超这个数，1:1 会常年顶限位。

## 和 G1 不一样的那一点（别照抄符号翻转）

G1 的左右手指屈曲关节范围是**反号**的，所以上游给 G1 写了 per-side 的 flex 符号翻转。
**L3.4 两只手的屈曲范围完全相同且都是正的**（curl 都是 0..1.919），而 MANO 的 bend
角本来就 ≥0（伸直=0，攥紧≈π），方向天然对上 —— 所以 flex 的 ``scale`` 两侧都是
``+1.0``。只有 ``index_abd``（食指外展）左右镜像，用 ``left_sign=-1.0`` 表达。

## 关节名是短键

``robot_name`` 是短键（``thumb_bend`` / ``index_mcp`` ...），不带 ``left_``/``right_``
前缀，也不是 MJCF 里的真名。重新贴前缀、映射到真名（``hand_thumb_bend_joint`` /
``hand_index_joint1`` / ``hand_mid_joint1`` ...）是 ``robots/l3_4/env.py`` 的事。
"""

#: 12 个关节，顺序即 ``q_fingers`` 的列顺序（``env.py`` 依赖这个顺序）。
#: 每项是能直接喂给上游 ``JointMapping(**d)`` 的 kwargs。
#: ``limit`` 取自 ``l3_4.xml`` 的 ``range``；``index_abd`` 用两侧较紧的那个
#: （左 ±0.174 / 右 ±0.175），这样两只手都不会命令到限位之外。
HAND_JOINT_SPEC = [
    # ── 拇指：bend（对掌，← MANO cmc）+ 两个 curl（← mcp / ip） ───────────────
    dict(robot_name="thumb_bend",  mano_joint=12, dof="flex", scale=1.0,
         limit=(0.0,    1.832)),
    dict(robot_name="thumb_rota1", mano_joint=13, dof="flex", scale=1.0,
         limit=(-0.698, 1.745)),
    dict(robot_name="thumb_rota2", mano_joint=14, dof="flex", scale=1.0,
         limit=(0.0,    1.919)),
    # ── 食指：abduction（← MANO abd，缩一半）+ MCP + PIP（dip 丢掉） ──────────
    dict(robot_name="index_abd",   mano_joint=0,  dof="abd",  scale=0.5,
         limit=(-0.174, 0.174), left_sign=-1.0),
    dict(robot_name="index_mcp",   mano_joint=0,  dof="flex", scale=1.0,
         limit=(0.0,    1.919)),
    dict(robot_name="index_pip",   mano_joint=1,  dof="flex", scale=1.0,
         limit=(0.0,    1.919)),
    # ── 中指：MCP + PIP ───────────────────────────────────────────────────────
    dict(robot_name="middle_mcp",  mano_joint=3,  dof="flex", scale=1.0,
         limit=(0.0,    1.919)),
    dict(robot_name="middle_pip",  mano_joint=4,  dof="flex", scale=1.0,
         limit=(0.0,    1.919)),
    # ── 无名指：MCP + PIP ─────────────────────────────────────────────────────
    dict(robot_name="ring_mcp",    mano_joint=9,  dof="flex", scale=1.0,
         limit=(0.0,    1.919)),
    dict(robot_name="ring_pip",    mano_joint=10, dof="flex", scale=1.0,
         limit=(0.0,    1.919)),
    # ── 小指：MCP + PIP ───────────────────────────────────────────────────────
    dict(robot_name="pinky_mcp",   mano_joint=6,  dof="flex", scale=1.0,
         limit=(0.0,    1.919)),
    dict(robot_name="pinky_pip",   mano_joint=7,  dof="flex", scale=1.0,
         limit=(0.0,    1.919)),
]

#: 12 个短键，顺序同上。``env.py`` 拿它拼真实 MJCF 关节名。
HAND_JOINT_NAMES = [d["robot_name"] for d in HAND_JOINT_SPEC]

__all__ = ["HAND_JOINT_SPEC", "HAND_JOINT_NAMES"]
