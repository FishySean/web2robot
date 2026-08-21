"""L3.4 单臂 7-DoF 的 IK 串链参数。**纯数据，不 import 任何重定向框架、不 import torch。**

和 ``config.ENV_SPEC`` / ``hand_mapping.HAND_JOINT_SPEC`` 同一个套路：链根、末端帧、
关节限位都是这台机器人的属性，跟用哪个 IK 求解器无关。框架侧自己包成它的类型
（EgoInfinity 那边是 ``kinematics/wrist_ik.py::RobotIKConfig.l3_4``，把
``joint_limits`` 转成 ``torch.tensor`` 再塞进 dataclass，见 ``external/patches/``）。

刻意**不 import torch**：这个包会被 hand_frame 检查、FK 校验这类只需要 mujoco 的
脚本 import，不该为了一个限位表拉进整个 torch。

## 链根为什么是 waist_pitch_link

它是两条手臂在 MJCF 树里的共同祖先。``torso_frame`` 和它位置重合，但 ``torso_frame``
是**叶子** body —— 拿叶子当串链根，pytorch_kinematics 会建出一条空链（IK 全帧失败
但不抛异常，很难查）。

盆骨 ``base_link`` 不是链根：本阶段腰/腿锁死，把它们纳进串链只会给 IK 多 15 个
永远不动的自由度。以后要做下肢，把 ``config.LOCKED_JOINTS`` 里对应的行删掉，
再决定链根往下挪到哪。

## 7 个关节的顺序

``waist_pitch_link`` → ``<side>_hand_frame`` 之间依次是：shoulder_pitch、shoulder_roll、
arm_yaw、elbow_pitch、elbow_yaw、wrist_pitch、wrist_roll。``JOINT_LIMITS`` 每行一个，
顺序必须和这个一致，也必须和 ``config.py`` 里 ``start_config`` 的 7 个数对齐。

左右不是简单镜像：shoulder_roll 和 arm_yaw 的区间是镜像的（外展方向相反），
其余五个两侧相同。

## 这些数字和 M7 的一样，但不共用

L3.4 的上肢和 M7 逐位同构（``scripts/dev/build_l3_4_assets.py`` 的 docstring 里列了
量到的证据），所以这张表和 ``robots/m7/ik_config.py`` 的数值相同。**没有 import
过去**是刻意的（见 ``env.py``）；防漂的办法是拿 MJCF 当唯一真相：
``tests/test_l3_4_robot.py`` 断言每一行都等于 ``l3_4.xml`` 里那个关节的 ``range``。

## 数字本身在 ``configs/robots/l3_4.yaml``

2026-08-21 搬过去的（``ik.joint_limits``）。**这里不留第二份。** 搬家没有让两台机器人
开始共用一张表 —— 是 ``m7.yaml`` 和 ``l3_4.yaml`` 各存一份、没用 yaml 锚点，理由和这两个
``ik_config.py`` 当初不互相 import 完全一样：真相是各自的 MJCF，哪天 L3.4 换一版 URDF
改了某个区间，各自一份能自然分叉，共享一份则会把 M7 的数悄悄按到 L3.4 头上还不报错。

## 链根/末端的名字不在这里写死

它们已经在 ``config.CONFIG`` 里了（``torso_body`` / ``wrist_body``，碰撞代理和
MuJoCo env 都从那儿取）。同一个 body 名存两份，改 MJCF 时必然漏一处，而漏了不会
报错 —— 所以这里直接引用，不复制。
"""
from web2robot.robots.l3_4.config import CONFIG as _CONFIG
from web2robot.robots.params import robot_params as _robot_params

_P = _robot_params("l3_4")

#: 串链根 body 名（两条手臂共用），来自 ``config.CONFIG["torso_body"]``。
ROOT_LINK_NAME = _CONFIG["torso_body"]

#: 末端 body 名，来自 ``config.CONFIG["wrist_body"]``。这两个 body 的轴向约定
#: （finger+y / thumb∓x / palm±z）是硬约定，见 ``robots/l3_4/__init__.py``。
END_LINK_NAME = dict(_CONFIG["wrist_body"])

#: (7, 2) 的关节限位，单位 rad，顺序见模块 docstring。纯 list，不是 tensor。
#: 数值来自 ``configs/robots/l3_4.yaml`` 的 ``ik.joint_limits``（唯一来源）。
JOINT_LIMITS = {
    side: [[float(lo), float(hi)] for lo, hi in _P["ik"]["joint_limits"][side]]
    for side in ("left", "right")
}

#: yaml 里记的关节顺序，和模块 docstring 那一串一致。
JOINT_ORDER = tuple(_P["ik"]["joint_limits"]["order"])


def ik_spec(side: str) -> dict:
    """一侧手臂的 IK 串链参数。

    返回的 dict 的键刻意和上游 ``RobotIKConfig`` 的字段名对齐，这样框架侧只要
    ``RobotIKConfig(mjcf_path=..., **spec)`` 级别的接线，不用再做名字翻译。
    ``mjcf_path`` 不在这里，由 ``robots.l3_4.MJCF_PATH`` 出。
    """
    if side not in END_LINK_NAME:
        raise ValueError(f"side 只能是 left/right，收到 {side!r}")
    return {
        "end_link_name":  END_LINK_NAME[side],
        "root_link_name": ROOT_LINK_NAME,
        "joint_limits":   JOINT_LIMITS[side],
    }


__all__ = ["ROOT_LINK_NAME", "END_LINK_NAME", "JOINT_LIMITS", "JOINT_ORDER",
           "ik_spec"]
