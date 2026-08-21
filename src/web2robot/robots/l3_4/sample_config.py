import numpy as np

# NOTE: SAMPLE_CONFIG 只被**训练期**的合成轨迹采样器（上游 scripts/train.py）用，
# 推理/验证路径（test.py）不碰它。本阶段 L3.4 借 M7 的根模型 checkpoint（双臂链逐位
# 相同，``build_l3_4_assets.py`` 验过），所以这张表现在还没被真正跑过 —— 留在这里是
# 为了"以后真要给 L3.4 单独训一个根模型"时不用从零推，**不是**已验证的配置。
#
# 数值和 robots/m7/sample_config.py 相同，理由是那两个按比例重算过的参数
# （elbow jitter + ou_step）依赖的比例在 L3.4 上完全一样：同一条手臂、同一套限位、
# 实测臂展同为 1.007 m。别以为"新机器人就该重调"——先确认比例是否真的变了。
#
# L3.4 arm order: j0 shoulder_pitch, j1 shoulder_roll, j2 arm_yaw, j3 elbow_pitch,
#                 j4 elbow_yaw, j5 wrist_pitch, j6 wrist_roll.

SAMPLE_CONFIG = {
    # lateral_joint: per-side jitter on j1 (shoulder_roll) keeps each arm on its own side.
    "lateral_joint": {"index": 1, "left": (0.10, 0.90), "right": (-0.90, -0.10)},

    # proximal_jitter: j0=shoulder_pitch, j2=arm_yaw, j3=elbow_pitch (bend variation).
    # j3 的 (-0.5, 0.5) 是按这条手臂的实际比例定的：start_config 的肘是 -1.0
    # （range -2.36..0.70，0 = 伸直），所以肘落在 [-1.50,-0.50] —— 既不到奇异点
    # （全伸直）也不死弯。照抄 R2 的 (-1.6,-0.3) 会把肘压到 [-2.36,-1.30]（永不伸展），
    # 躯干→手腕距离只有臂展的 50%。见 robots/m7 那边的实测记录。
    "proximal_jitter": {0: (-0.6, 0.6), 2: (-0.7, 0.7), 3: (-0.5, 0.5)},

    # OU walk for workspace coverage。ou_step 随臂展缩放
    # （R2 0.040/1.28m、G1 0.026/0.72m → 比值 ~0.031）；本臂实测 1.007 m，
    # 所以取 R2 的向量 × (1.00/1.28)：[0.040,0.035,0.030] → [0.031,0.027,0.023]
    # （保留 R2 的各向异性，只把幅度缩到这条手臂）。
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
