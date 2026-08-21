import numpy as np

from web2robot.robots.l3_4.env import (
    L34Env, _SCENE_PATH, _MJCF_MJX_PATH, _ARM_JOINTS, _EE_BODY, _LOCKED_JOINTS,
)

# IK seed poses (arm order: shoulder_pitch, shoulder_roll, arm_yaw,
# elbow_pitch, elbow_yaw, wrist_pitch, wrist_roll).
# elbow_pitch bends negative (range -2.36..0.7); shoulder_roll abducts +left/-right.
# 数值和 M7 相同，因为这条手臂的运动学和限位与 M7 逐位相同（见 env.py 的说明）——
# 是量出来的巧合，不是共用了代码。
_START_L = np.array([0.0,  0.20, 0.0, -1.0, 0.0, 0.0, 0.0], dtype=np.float32)
_START_R = np.array([0.0, -0.20, 0.0, -1.0, 0.0, 0.0, 0.0], dtype=np.float32)

CONFIG = {
    # ── environment ───────────────────────────────────────────────────────────
    "env_cls":              L34Env,
    "scene_path":           _SCENE_PATH,
    # ── IK / retargeting ─────────────────────────────────────────────────────
    "ik_robot":             "l3_4",
    "retargeter":           "l3_4",
    # ── trajectory sampling ───────────────────────────────────────────────────
    "apply_workspace_bias": False,
    # ── MuJoCo body names ─────────────────────────────────────────────────────
    "torso_body":           "waist_pitch_link",
    "wrist_body":           {"left": "left_hand_frame", "right": "right_hand_frame"},
    "extra_bodies":         [],
    # ── joint configurations ──────────────────────────────────────────────────
    "zero_config": {
        "left":  np.zeros(7, dtype=np.float32),
        "right": np.zeros(7, dtype=np.float32),
    },
    "home_config": {
        "left":  _START_L.copy(),
        "right": _START_R.copy(),
    },
    "start_config": {
        "left":  _START_L.copy(),
        "right": _START_R.copy(),
    },
    # ── viewer camera ─────────────────────────────────────────────────────────
    "cam_azimuth":   180,          # front view（躯干朝 -x，同 M7）
    "cam_elevation": -15,
    "cam_distance":  1.7,
    "cam_lookat":    [0.0, 0.0, 0.15],
}

#: 锁死的 17 个自由度（腰 3 + 颈 2 + 腿 12）→ 各自被按住的角度 [rad]。
#: 从 ``env`` 转出来是为了让"哪些自由度不动"这件事在配置层可见、可改：
#: **从这张表里删掉一行就解锁一个自由度**，不用碰资产、不用改 env 的逻辑。
#: 本阶段全是 0.0（URDF 零位：腿直立、腰颈正中）。
#: ``tests/test_l3_4_robot.py`` 钉住"arm ∪ finger ∪ locked = MJCF 的全部 55 个关节"——
#: 少一个都算漏，漏了就意味着有个自由度既没人控制也没被锁住。
LOCKED_JOINTS = dict(_LOCKED_JOINTS)

# 和 robots/m7/config.py 同一个套路：这里只出**数据**，不出上游**类型**。
# 谁要 dataclass 谁自己包（EgoInfinity 那侧是 ``RobotConfig(**ENV_SPEC)``，见
# external/patches/ 对 sim/robots/__init__.py 的改动）。这样 ①这个模块不 import
# 上游任何东西；②上游给 RobotConfig 加了必填字段会在 import 时就 TypeError，
# 而不是悄悄用默认值跑出错的 FK。
ENV_SPEC = {
    "mjcf_path":     _MJCF_MJX_PATH,   # arms-only MJX model：FK / grid 根位姿搜索
    "joint_groups":  _ARM_JOINTS,
    "end_effectors": _EE_BODY,
}
