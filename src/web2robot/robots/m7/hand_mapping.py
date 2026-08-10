"""M7 五指手 12 DoF ← MANO 关节的映射表。**纯数据，不 import 任何重定向框架。**

和 ``config.ENV_SPEC`` 是同一个套路：这一层回答"M7 的手长什么样、哪个 MANO 关节
驱动哪个机器人关节"，这是机器人属性，跟用哪个重定向框架无关；框架侧自己把
``HAND_JOINT_SPEC`` 里的 dict 包成它的 dataclass（EgoInfinity 那边是
``RobotHandConfig(joints=[JointMapping(**d) for d in HAND_JOINT_SPEC])``，
见 ``external/patches/`` 对 ``kinematics/wilor_retargeter.py`` 的改动）。

出 dict 而不是上游类型，除了解耦还顺带当检查：上游哪天给 ``JointMapping`` 加了
必填字段，``JointMapping(**d)`` 会在 import 时就 TypeError，而不是悄悄用默认值跑出
错的手型。

## M7 和 G1 不一样的那一点

G1 的左右手指屈曲关节范围是**反号**的，所以上游给 G1 写了 per-side 的 flex 符号翻转。
**M7 两只手的屈曲范围完全相同且都是正的**（curl 都是 0..1.919），而 MANO 的 bend
角本来就 ≥0（伸直=0，攥紧≈π），方向天然对上 —— 所以 M7 的 flex ``scale`` 两侧都是
``+1.0``，不要照抄 G1 去加符号翻转。只有 ``index_abd``（食指外展）是左右镜像的，
用 ``left_sign=-1.0`` 表达。

## 关节名是短键

``robot_name`` 是短键（``thumb_bend`` / ``index_mcp`` ...），不带 ``left_``/``right_``
前缀，也不是 MJCF 里的真名。重新贴前缀、映射到真实 MJCF 关节名
（``hand_thumb_bend_joint`` / ``hand_index_joint1`` / ``hand_mid_joint1`` ...）是
``web2robot/robots/m7/env.py`` 的事。

MANO-21 的关节编号约定见上游 ``kinematics/wilor_retargeter.py`` 顶部的说明：
特征数组是 (15, 2)，顺序 index[mcp,pip,dip] middle[...] pinky[...] ring[...]
thumb[cmc,mcp,ip]，每项给 (flex, abd)。下面 ``mano_joint`` 用的就是这个 0..14 编号，
**不是** 21 个关键点的编号。
"""

#: 12 个关节，顺序即 ``q_fingers`` 的列顺序（``env.py`` 依赖这个顺序）。
#: 每项是能直接喂给上游 ``JointMapping(**d)`` 的 kwargs。
HAND_JOINT_SPEC = [
    # ── 拇指：bend（对掌）+ 两个 curl ──────────────────────────────────────────
    dict(robot_name="thumb_bend",  mano_joint=12, dof="flex", scale=1.0,
         limit=(0.0,    1.832)),
    dict(robot_name="thumb_rota1", mano_joint=13, dof="flex", scale=1.0,
         limit=(-0.698, 1.745)),
    dict(robot_name="thumb_rota2", mano_joint=14, dof="flex", scale=1.0,
         limit=(0.0,    1.919)),
    # ── 食指：abduction（张开）+ MCP + PIP ────────────────────────────────────
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
