import numpy as np

from web2robot.robots.m7.env import (
    M7Env, _SCENE_PATH, _MJCF_MJX_PATH, _ARM_JOINTS, _EE_BODY,
)

# IK seed poses (arm order: shoulder_pitch, shoulder_roll, arm_yaw,
# elbow_pitch, elbow_yaw, wrist_pitch, wrist_roll).
# elbow_pitch bends negative (range -2.36..0.7); shoulder_roll abducts +left/-right.
_START_L = np.array([0.0,  0.20, 0.0, -1.0, 0.0, 0.0, 0.0], dtype=np.float32)
_START_R = np.array([0.0, -0.20, 0.0, -1.0, 0.0, 0.0, 0.0], dtype=np.float32)

CONFIG = {
    # ── environment ───────────────────────────────────────────────────────────
    "env_cls":              M7Env,
    "scene_path":           _SCENE_PATH,
    # ── IK / retargeting ─────────────────────────────────────────────────────
    "ik_robot":             "m7",
    "retargeter":           "m7",
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
    "cam_azimuth":   180,          # front view (empirically; m7 torso faces -x)
    "cam_elevation": -15,
    "cam_distance":  1.7,
    "cam_lookat":    [0.0, 0.0, 0.15],
}

# 迁移前这里是 ``ENV_CONFIG = RobotConfig(mjcf_path=..., joint_groups=..., end_effectors=...)``，
# 即直接构造上游 ``sim.robot_config.RobotConfig``。现在只出**数据**，不出上游**类型**：
# 谁要用谁自己包（EgoInfinity 那侧是 ``RobotConfig(**ENV_SPEC)``，见
# external/patches/ 里对 sim/robots/__init__.py 的改动）。
#
# 好处有两个：①这个模块不 import 上游任何东西，换重定向框架不用改它；
# ②上游哪天给 RobotConfig 加了必填字段，``RobotConfig(**ENV_SPEC)`` 会在
# import 时就 TypeError 报出来，而不是悄悄用默认值跑出错的 FK。
ENV_SPEC = {
    "mjcf_path":     _MJCF_MJX_PATH,   # arms-only MJX model for training FK
    "joint_groups":  _ARM_JOINTS,
    "end_effectors": _EE_BODY,
}
