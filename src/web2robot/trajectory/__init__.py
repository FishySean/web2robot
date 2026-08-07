"""轨迹清洗 —— 坏帧检测、填补、静息混合。

放在 ``collision/`` 之外单独一个包，是因为它管的是**另一类问题**：碰撞纠正
处理的是"姿态在物理上不合法"，这里处理的是"这一帧的数据根本不可信"
（手没检测到、手腕深度爆点、四元数跳变）。两者的输入输出都是关节/位姿轨迹，
但判据、时机、可逆性都不一样 —— 清洗在重定向**之前**（喂给 IK 的源数据），
碰撞纠正在**之后**（改的是解出来的关节角）。

三级坏帧检测 + 长度感知填补：短的内部空洞插值（``FILL_INTERP``）、短的边界
空洞保持（``FILL_HOLD``）、长空洞标记为 ``FILL_REST`` 交给调用方渐入静息位 ——
**长空洞不是可用数据**，填出来也不能当训练数据用，所以状态要带出去而不是
悄悄补平。这套机制修掉了 ``fill_jar`` 里左手崩坏 11 秒那一段。
"""
from .traj_cleanup import (
    clean_wrist_trajectory, detect_bad_frames, canonicalize_quats,
    blend_to_rest, relax_fingers,
    OK, FILL_INTERP, FILL_HOLD, FILL_REST, STATUS_NAMES,
    C_OK, C_MISSING, C_BAD, CAUSE_NAMES,
)

__all__ = ["clean_wrist_trajectory", "detect_bad_frames", "canonicalize_quats",
           "blend_to_rest", "relax_fingers",
           "OK", "FILL_INTERP", "FILL_HOLD", "FILL_REST", "STATUS_NAMES",
           "C_OK", "C_MISSING", "C_BAD", "CAUSE_NAMES"]
