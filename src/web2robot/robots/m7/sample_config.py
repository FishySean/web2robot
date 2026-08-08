import numpy as np

# NOTE: SAMPLE_CONFIG is only consumed by the training-time synthetic trajectory
# sampler (scripts/train.py).  It is NOT used by the inference/verification path
# (test.py).  Base scheme mirrors robonaut2; two params were re-derived for M7's
# actual proportions (elbow jitter + ou_step, see inline notes below), everything
# else is retained as general (non-proportion) design experience from G1/R2.
#
# M7 arm order: j0 shoulder_pitch, j1 shoulder_roll, j2 arm_yaw, j3 elbow_pitch,
#               j4 elbow_yaw, j5 wrist_pitch, j6 wrist_roll.

SAMPLE_CONFIG = {
    # lateral_joint: per-side jitter on j1 (shoulder_roll) keeps each arm on its own side.
    "lateral_joint": {"index": 1, "left": (0.10, 0.90), "right": (-0.90, -0.10)},

    # proximal_jitter: j0=shoulder_pitch, j2=arm_yaw, j3=elbow_pitch (bend variation).
    # j3 re-derived for M7: start_config elbow = -1.0 (range -2.36..0.70, 0=straight).
    # The R2-copied (-1.6,-0.3) drove the elbow to [-2.36,-1.30] (dead-bent, never
    # extends) -> torso->wrist mean 0.50m (50% of M7's 1.00m reach) vs R2's ~60%.
    # (-0.5, 0.5) puts the elbow in [-1.50,-0.50]: never straight (no singularity),
    # never fully folded, and lifts torso->wrist mean to 0.60m / p90 0.73m (~60%).
    "proximal_jitter": {0: (-0.6, 0.6), 2: (-0.7, 0.7), 3: (-0.5, 0.5)},

    # OU walk for workspace coverage.
    # ou_step scales with arm reach (R2 0.040/1.28m, G1 0.026/0.72m -> ratio ~0.031).
    # M7 reach = 1.00m, so R2's vector is scaled by 1.00/1.28: [0.040,0.035,0.030]
    # -> [0.031,0.027,0.023] (keeps R2's anisotropy, shrinks magnitude to M7's arm).
    "ou_step":   np.array([0.031, 0.027, 0.023], dtype=np.float32),
    "ou_spring": 0.04,

    # Wrist slice [4:7]: j4 elbow_yaw, j5 wrist_pitch, j6 wrist_roll.
    "wrist_joints":   (4, 7),
    "wrist_limits": {
        "left":  np.array([[-2.5, 2.5], [-0.79, 0.79], [-1.57, 1.57]], dtype=np.float32),
        "right": np.array([[-2.5, 2.5], [-0.79, 0.79], [-1.57, 1.57]], dtype=np.float32),
    },
    "wrist_relative": False,

    "workspace_bias": {},
}
