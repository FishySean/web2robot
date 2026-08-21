"""M7 单臂 7-DoF 的 IK 串链参数。**纯数据，不 import 任何重定向框架、不 import torch。**

和 ``config.ENV_SPEC`` / ``hand_mapping.HAND_JOINT_SPEC`` 是同一个套路：链根、末端帧、
关节限位都是 M7 这台机器人的属性，跟用哪个 IK 求解器无关。框架侧自己包成它的类型
（EgoInfinity 那边是 ``kinematics/wrist_ik.py::RobotIKConfig.m7``，把 ``joint_limits``
转成 ``torch.tensor`` 再塞进 dataclass，见 ``external/patches/``）。

这里刻意**不 import torch**：``robots/m7/`` 会被 hand_frame 检查、FK 校验这类只需要
mujoco 的脚本 import，不该为了一个限位表拉进整个 torch。

## 链根为什么是 waist_pitch_link

它是两条手臂在 MJCF 树里的共同祖先。``torso_frame`` 和它位置重合，但 ``torso_frame``
是**叶子** body —— 拿叶子当串链根，pytorch_kinematics 会建出一条空链。

## 7 个关节的顺序

``waist_pitch_link`` → ``<side>_hand_frame`` 之间依次是：shoulder_pitch、shoulder_roll、
arm_yaw、elbow_pitch、elbow_yaw、wrist_pitch、wrist_roll。``JOINT_LIMITS`` 每行一个，
顺序必须和这个一致，也必须和 ``config.py`` 里 ``start_config`` 的 7 个数对齐。

左右不是简单镜像：shoulder_roll 和 arm_yaw 的区间是镜像的（外展方向相反），
其余五个两侧相同。

## 数字本身在 ``configs/robots/m7.yaml``

2026-08-21 搬过去的（``ik.joint_limits``）。**这里不留第二份** —— 原来这张表是写死在
本文件里的，翻参数得进代码看，而且没地方记"这个数是抄 MJCF 的、还没人标定过"。
真相依然是 ``m7.xml``：``tests/test_m7_robot.py`` 断言 yaml 里每一行等于对应关节的
``range``，所以改 MJCF 忘了改 yaml 会红，而不是悄悄用旧限位跑。

## 链根/末端的名字不在这里写死

它们已经在 ``config.CONFIG`` 里了（``torso_body`` / ``wrist_body``，碰撞代理和
MuJoCo env 都从那儿取）。同一个 body 名在仓库里存两份，改 MJCF 时必然漏一处，
而漏了不会报错 —— pytorch_kinematics 会拿旧名字建一条**空链**，IK 全帧失败但
不抛异常。所以这里直接引用，不复制。
"""
from web2robot.robots.m7.config import CONFIG as _CONFIG
from web2robot.robots.params import robot_params as _robot_params

_P = _robot_params("m7")

#: 串链根 body 名（两条手臂共用），来自 ``config.CONFIG["torso_body"]``。
ROOT_LINK_NAME = _CONFIG["torso_body"]

#: 末端 body 名，来自 ``config.CONFIG["wrist_body"]``。这两个 body 的轴向约定
#: （finger+y / thumb∓x / palm±z）是硬约定，见 ``robots/m7/__init__.py`` 的说明和
#: ``scripts/dev/check_handframe_convention.py``。
END_LINK_NAME = dict(_CONFIG["wrist_body"])

#: (7, 2) 的关节限位，单位 rad，顺序见模块 docstring。纯 list，不是 tensor。
#: 数值来自 ``configs/robots/m7.yaml`` 的 ``ik.joint_limits``（唯一来源）。
JOINT_LIMITS = {
    side: [[float(lo), float(hi)] for lo, hi in _P["ik"]["joint_limits"][side]]
    for side in ("left", "right")
}

#: yaml 里记的关节顺序，和模块 docstring 那一串一致（``tests/test_robot_params_yaml.py``
#: 断言两侧限位都是 7 行、和这个顺序等长）。
JOINT_ORDER = tuple(_P["ik"]["joint_limits"]["order"])


def ik_spec(side: str) -> dict:
    """一侧手臂的 IK 串链参数。

    返回的 dict 的键刻意和上游 ``RobotIKConfig`` 的字段名对齐，这样框架侧只要
    ``RobotIKConfig(mjcf_path=..., **{k: ... for k in spec})`` 级别的接线，
    不用再做名字翻译。``mjcf_path`` 不在这里，由 ``robots.m7.MJCF_PATH`` 出。
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
