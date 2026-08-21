"""轨迹清洗 —— 坏帧检测、填补、静息混合。

放在 ``collision/`` 之外单独一个包，是因为它管的是**另一类问题**：碰撞纠正
处理的是"姿态在物理上不合法"，这里处理的是"这一帧的数据根本不可信"
（手没检测到、手腕深度爆点、四元数跳变）。两者的输入输出都是关节/位姿轨迹，
但判据、时机、可逆性都不一样 —— 清洗在重定向**之前**（喂给 IK 的源数据），
碰撞纠正在**之后**（改的是解出来的关节角）。

三级坏帧检测 + **位置感知**的填补。``FILL_REST``（渐入静息位）是最后兜底，
**能不触发就不触发**，所以空洞落在哪里和它有多长一样重要：

- **中间的短空洞** → 两端插值（``FILL_INTERP``，≤ 2.5 s）；更长才 ``FILL_REST``，
  因为几秒的直线插值是编的、不是插的。
- **开头的空洞** → 短的保持第一帧（``FILL_HOLD``，≤ 0.5 s），长的 ``FILL_REST``：
  空洞前面没有帧可以沿袭，手臂总得从某个地方起手，给静息位比编一个动作诚实。
- **结尾的空洞** → **一律沿袭最后一帧**（``FILL_HOLD``，不论多长）。后面没有帧，
  保持不会造成任何跳变；而在最没有信息的地方渐入静息位反而是编一段大幅运动。
  帧仍标 ``FILL_HOLD`` 而不是 ``OK``，长度记在 ``report["tail_hold"]`` 里，
  所以"这段是保持出来的、不是测到的"依然传得下去。

开头/结尾的不对称是 2026-08-11 定的设计决策，不是疏漏。**空洞填出来的帧都不是
可用数据**，所以状态必须带出去而不是悄悄补平。这套机制修掉了 ``fill_jar`` 里
左手崩坏 11 秒那一段。

## 上面那一整套是**逐帧**这一层

2026-08-21 按 EgoSmith（EgoSteer，arXiv 2607.09701）补了另外两个粒度 ——
**整段**（相机运动分布离群）和**轨迹段**（一小段的手腕/手指坐标空间离群），在
``tiers.py``。三层是**独立并列**的判据，新增的两层不覆盖、不替换上面这套
"跳变/鼓包/四元数翻转"，而且**都不改数据**（整段只警告、轨迹段只打标记）。
``--bad_frame_tiers`` 默认只有 ``frame``，即现状不变。
"""
from .traj_cleanup import (
    clean_wrist_trajectory, detect_bad_frames, canonicalize_quats,
    blend_to_rest, relax_fingers,
    OK, FILL_INTERP, FILL_HOLD, FILL_REST, STATUS_NAMES,
    C_OK, C_MISSING, C_BAD, CAUSE_NAMES,
)
from .tiers import (
    TIER_NAMES, DEFAULT_TIERS, parse_tiers,
    EpisodeReport, episode_camera_check,
    SegmentFinding, segment_spatial_check,
)

__all__ = ["clean_wrist_trajectory", "detect_bad_frames", "canonicalize_quats",
           "blend_to_rest", "relax_fingers",
           "OK", "FILL_INTERP", "FILL_HOLD", "FILL_REST", "STATUS_NAMES",
           "C_OK", "C_MISSING", "C_BAD", "CAUSE_NAMES",
           "TIER_NAMES", "DEFAULT_TIERS", "parse_tiers",
           "EpisodeReport", "episode_camera_check",
           "SegmentFinding", "segment_spatial_check"]
