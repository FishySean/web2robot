"""坏帧过滤的**三个粒度**（EgoSmith 的做法），判据本身。

出处
----
EgoSmith 数据清洗管线，出自 EgoSteer（arXiv 2607.09701，"post-filtering" 那一节）。
原文按三个粒度分别设判据，**每一层是独立的判据，不是同一套标准套三次**：

  * episode 级 —— "compute camera translation distributions to discard outliers,
    while applying hard rotation thresholds to drop episodes with excessive head
    motions"
  * chunk 级 —— "transform wrist poses into its middle camera frame, project
    finger keypoints into frame-wise wrist frames, and discard spatial outliers
    across wrist and finger coordinates"
  * frame 级 —— "compute frame-to-frame deltas of camera, wrist, and finger
    motions, filtering out abrupt jumps with hard thresholds"

我们和它的对应关系
------------------
frame 级**早就有了**，在 ``traj_cleanup.py``（速度门 + 跑合并 + 鼓包判据 + 四元数
翻转），而且比原文的"hard thresholds on frame-to-frame deltas"多做了一步（鼓包判据
把"真快速伸手"和"感知爆点"分开）。这个模块**不动它**，只补上面两层。

episode / segment 两层是这次（2026-08-21）新增的。

为什么这两层判的是 §V2/§V3
--------------------------
见 ``docs/VIDEO_SELECTION_GUIDE.md``：

  * **§V2**（人体不能大范围走位）和 **§V3**（人体朝向不能变）—— 交给 IK 的手部目标
    是**相机系**的（``utils/pose_utils.py::cam_to_root_targets`` 算
    ``p_root = R_rootᵀ(p_hand_cam − t_root)``），而 grid 路线的躯干位姿是整段
    **一个相机系里的常量**。所以相机/身体一动，假的手部位移就 1:1 注入。
  * 那两条是**准入**判据（挑素材的时候看），这两层是**事后核查**：素材已经跑到重定向
    这一步了，回头问一句"这一段/这一小段，是不是正好踩了 §V2/§V3 说的那种
    '手的目标位置被镜头运动带偏'"。同一个机理，两个时机。

和 §V1（时间连续性）的关系：§V1 管的是**阶跃**（切镜），那是 frame 级那一层的活；
这两层管的是**斜坡**（缓慢漂移），§V1 那节的表格里写了为什么斜坡更难发现。

检出之后做什么（**和原文不同，是我们自己定的**）
------------------------------------------------
原文对三层都是 **discard**（"systematically discards problematic data"）。我们**不
照抄**：

  * **episode 级 → 只警告，不阻断。** 这段视频照样跑完整条流水线，弃不弃用由人看了
    警告自己决定。理由：这一层的信号最粗（见下面"我们做不到原文哪一步"），拿最粗的
    信号做最重的决定（整段扔掉）不划算，而且我们的素材量远不到 9.6K 小时，扔错的
    代价比原文大得多。
  * **segment 级 → 只打标记，什么都不做。** 不插值、不丢弃、不填补。标记留给下游
    （``refine/``）决定"这一小段该不该升级精修"。理由：这一层指出的是"这一小段的
    手部/手指坐标在空间上离群"，而离群的原因可能是感知坏了，**也可能是动作本身就
    快**——在这里自动处理等于替下游做了它自己能做得更好的判断。

所以这个模块**不返回被改过的轨迹**，只返回 report。这是刻意的：它一个数都改不了。

我们做不到原文哪一步（别把这层当成原文那层）
--------------------------------------------
1. **episode 级只能做"段内"离群，做不了"跨语料"离群。** 原文是在 9.6K 小时的语料上
   看相机平移的**分布**，把分布尾巴上的 episode 扔掉。我们的流水线一次只处理一段
   clip，手里没有语料分布，所以这里做的是**段内**离群：这一段自己的哪些帧对比这一段
   自己的典型运动明显更大。跨语料那种要在第 1 步（质检扫全库的时候）做，见
   ``docs/BACKLOG.md``。
2. **没有相机位姿，只有光流代理。** 官方 clip 契约里没有逐帧相机位姿（``camera.json``
   只有内参 + 一个重力方向），所以"相机平移"是用 ``quality/motion.py::bg_flow_score``
   的背景光流代理的 —— 那是从官方 ``action100m_filter::_camera_motion_score_flow``
   移植过来的同一套逻辑，这里**直接复用，不另造一套**。
3. **光流分不开平移和旋转。** 所以 episode 级的警告写的是"§V2/§V3"，不是二者之一。
   原文那条"hard rotation thresholds"我们没有对应实现（没有相机旋转量可用）。
4. **绝对阈值不可迁移。** 官方那套 ``max_bg_flow=2.0`` 是在 **RGB** 上量的；重定向
   这一步手里只有 ``depth.mp4``（深度可视化），灰度含义完全不同。所以这里**只用相对
   判据**（中位数 + MAD 的稳健 z），一个绝对阈值都不设。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

#: 三个粒度的合法名字，顺序 = 粒度从粗到细。
TIER_NAMES = ("episode", "segment", "frame")

#: 默认只有 frame —— 等于现状不变（新增的两层是可选启用的）。
DEFAULT_TIERS = ("frame",)


# ── 两层共用的稳健离群统计 ────────────────────────────────────────────────────

def _robust_z(v: np.ndarray) -> np.ndarray:
    """一维稳健 z：中位数 + MAD（Iglewicz–Hoaglin），配 ``z_thresh=3.5`` 用。

    ``0.6745`` 是正态下 MAD 到标准差的换算常数。

    **MAD=0 时退回"绕中位数的平均绝对偏差"**（``1.253314`` 是它对应的换算常数）——
    这是 Iglewicz–Hoaglin 自己给 MAD=0 的处方，不是我们编的。为什么非要有这个退路：
    上游 ``get_window()`` 的零阶保持会填出**逐位相同**的一段，MAD 恰好是 0，此时
    "一根手指飞出去一帧"在 MAD 口径下是 0/0，会被判成"无离群" —— 而那正是最该抓的
    情形。两个尺度都是 0 才是真退化（全部相等），那时返回全 0。
    """
    v = np.asarray(v, dtype=np.float64)
    med = np.median(v)
    dev = np.abs(v - med)
    mad = float(np.median(dev))
    if mad > 1e-9:
        return 0.6745 * (v - med) / mad
    mean_ad = float(np.mean(dev))
    if mean_ad > 1e-12:
        return (v - med) / (1.253314 * mean_ad)
    return np.zeros_like(v)


def parse_tiers(spec: str) -> Tuple[str, ...]:
    """``"episode,segment,frame"`` → ``("episode", "segment", "frame")``。

    顺序一律规范化成 :data:`TIER_NAMES` 的顺序（粗到细），这样 report 里的键序和
    日志顺序不随命令行里写的顺序变 —— 否则同样的一次跑，参数写法不同产物就不同。

    空串 → 空元组（三层全关）。未知名字直接报错，不静默忽略：``--bad_frame_tiers
    epsiode`` 打错一个字母就变成"什么都没开"，而人以为开了。
    """
    parts = [p.strip() for p in str(spec).split(",") if p.strip()]
    bad = [p for p in parts if p not in TIER_NAMES]
    if bad:
        raise ValueError(
            f"未知的坏帧过滤粒度 {bad}；只能是 {list(TIER_NAMES)} 的逗号分隔组合")
    return tuple(t for t in TIER_NAMES if t in parts)


# ── episode 级：整段的相机运动分布 ─────────────────────────────────────────────

@dataclass
class EpisodeReport:
    """整段级核查结果。``warn=True`` 时只打警告，**不阻断流水线**（见模块 docstring）。"""
    warn:            bool
    reason:          str
    n_pairs:         int
    flow_med:        Optional[float]      # 背景光流中位数 [px/frame]
    flow_mad:        Optional[float]
    flow_p90:        Optional[float]
    outlier_frames:  List[int] = field(default_factory=list)
    outlier_frac:    float = 0.0
    max_robust_z:    Optional[float] = None
    clause:          str = "§V2/§V3"      # 引用的判据编号，见 VIDEO_SELECTION_GUIDE.md

    def to_dict(self) -> Dict[str, Any]:
        return dict(warn=self.warn, reason=self.reason, n_pairs=self.n_pairs,
                    flow_med=self.flow_med, flow_mad=self.flow_mad,
                    flow_p90=self.flow_p90, outlier_frames=self.outlier_frames,
                    outlier_frac=self.outlier_frac,
                    max_robust_z=self.max_robust_z, clause=self.clause)


def episode_camera_check(
    frames:       Sequence[np.ndarray],   # 逐帧 BGR（重定向这一步手里是 depth.mp4）
    stride:       int   = 1,              # 每隔几帧取一对（长片降采样用）
    z_thresh:     float = 3.5,            # 稳健 z（MAD 口径）超过它算离群帧对
    frac_thresh:  float = 0.05,           # 离群帧对占比超过它就警告
    grid:         Optional[Tuple[int, int]] = None,
    percentile:   Optional[int] = None,
) -> EpisodeReport:
    """整段级：相机运动分布里有没有离群 —— **依据 VIDEO_SELECTION_GUIDE.md §V2/§V3**。

    做法：相邻帧背景光流（``quality/motion.py::bg_flow_score``，即官方
    ``action100m_filter::_camera_motion_score_flow`` 那一套，直接复用）→ 中位数 +
    MAD → 稳健 z。z 超过 ``z_thresh`` 的帧对记为离群。

    ``z_thresh=3.5`` 是 MAD 离群检测的常规取法（Iglewicz–Hoaglin），**不是我们量出来
    的**；``frac_thresh=0.05`` 同理，是个占比意义上的"不止一两帧"。两个数都还没有用
    标注过的素材定过，见 ``docs/BACKLOG.md``。

    为什么用相对判据而不是绝对阈值、为什么分不开平移和旋转、为什么做不到原文的跨语料
    分布 —— 见模块 docstring 的"我们做不到原文哪一步"。
    """
    import cv2
    from web2robot.quality.config import QCConfig
    from web2robot.quality.motion import bg_flow_score

    _cfg = QCConfig()
    grid = _cfg.flow_grid if grid is None else grid
    percentile = _cfg.flow_percentile if percentile is None else percentile

    stride = max(1, int(stride))
    idx = list(range(0, max(0, len(frames) - stride), stride))
    if len(idx) < 3:
        # 3 对以下算不出分布，MAD 会退化成 0 → 任何一点差异都成"无穷大 z"。
        return EpisodeReport(warn=False, reason="帧数太少，算不出分布（<3 对）",
                             n_pairs=len(idx), flow_med=None, flow_mad=None,
                             flow_p90=None)

    def _gray(im):
        a = np.asarray(im)
        return cv2.cvtColor(a, cv2.COLOR_BGR2GRAY) if a.ndim == 3 else a

    scores = np.array(
        [bg_flow_score(_gray(frames[i]), _gray(frames[i + stride]), grid, percentile)
         for i in idx], dtype=np.float64)

    med = float(np.median(scores))
    mad = float(np.median(np.abs(scores - med)))
    z = _robust_z(scores)          # 稳健 z 的口径（含 MAD=0 的退路）见 _robust_z

    out = [int(idx[i]) for i in np.where(z > z_thresh)[0]]
    frac = len(out) / len(idx)
    warn = frac > frac_thresh
    reason = (f"相机运动分布里 {len(out)}/{len(idx)} 对帧离群（占 {frac:.1%} "
              f"> {frac_thresh:.0%}）—— 可能踩了 §V2（走位）或 §V3（转身/转头）；"
              f"光流分不开平移和旋转，需要人看画面确认"
              if warn else
              f"相机运动分布正常（{len(out)}/{len(idx)} 对帧离群，占 {frac:.1%}）")
    return EpisodeReport(
        warn=warn, reason=reason, n_pairs=len(idx),
        flow_med=round(med, 4), flow_mad=round(mad, 4),
        flow_p90=round(float(np.percentile(scores, 90)), 4),
        outlier_frames=out, outlier_frac=round(frac, 4),
        max_robust_z=round(float(z.max()), 3) if len(z) else None)


# ── segment 级：一小段轨迹在空间上是否离群 ────────────────────────────────────

@dataclass
class SegmentFinding:
    """一条 segment 级标记。**没有任何自动处理**，只是"这几帧值得怀疑"。"""
    side:    str            # left / right
    start:   int            # 帧号（含）
    end:     int            # 帧号（含）
    kind:    str            # wrist_outlier / finger_outlier
    n_flag:  int            # 这一段里被标记的帧数
    score:   float          # 最大稳健 z
    clause:  str = "§V2/§V3"

    def to_dict(self) -> Dict[str, Any]:
        return dict(side=self.side, start=self.start, end=self.end,
                    kind=self.kind, n_flag=self.n_flag, score=self.score,
                    clause=self.clause)


def _quat_wxyz_to_rotmat(q: np.ndarray) -> np.ndarray:
    """(4,) wxyz → (3,3)。手写而不是拉 scipy：这里一次要转几千个，且要容 NaN。"""
    w, x, y, z = (float(v) for v in q)
    n = w * w + x * x + y * y + z * z
    if not np.isfinite(n) or n < 1e-12:
        return np.full((3, 3), np.nan)
    s = 2.0 / n
    return np.array([
        [1 - s * (y * y + z * z), s * (x * y - w * z),     s * (x * z + w * y)],
        [s * (x * y + w * z),     1 - s * (x * x + z * z), s * (y * z - w * x)],
        [s * (x * z - w * y),     s * (y * z + w * x),     1 - s * (x * x + y * y)],
    ], dtype=np.float64)


def segment_spatial_check(
    traj:      np.ndarray,                  # (T, 7) 相机系手腕位姿，wxyz；NaN = 未检测
    joints:    Optional[np.ndarray],        # (T, 21, 3) 相机系手部关键点；可以是 None
    fps:       float,
    side:      str = "",
    seg_sec:   float = 2.0,                 # 一小段多长
    z_thresh:  float = 3.5,
    min_valid: int   = 5,                   # 一小段里至少这么多有效帧才判
) -> List[SegmentFinding]:
    """轨迹段级：一小段的手腕/手指坐标在空间上是否离群 —— **依据 §V2/§V3**。

    照 EgoSmith 的做法（见模块 docstring 的原文引文）：

    1. 把这一小段的手腕位姿**转到该段"中间帧"的坐标系**里
       （``p̃_t = R_midᵀ (p_t − p_mid)``）。为什么要转：相机系的绝对位置本身没有可比
       性（整段可能整体偏一边），转到段内中间帧之后，剩下的就是"这一小段自己内部"的
       相对位移 —— 而 §V2/§V3 说的正是这个量被镜头运动污染。
    2. 手指关键点**转到逐帧的手腕系**里（``R_tᵀ (j_{t,k} − p_t)``）。逐帧、不是中间帧：
       手指相对手腕的形状不该随时间大幅跳，跳了就是感知抖了。
    3. 两组坐标各自算稳健 z（中位数 + MAD），超过 ``z_thresh`` 的帧记为空间离群点。

    ``seg_sec=2.0`` 和 ``--window_secs`` 的默认值一致（同一个"一小段"的量级），
    ``z_thresh=3.5`` 同 episode 级，都还没有用标注素材定过 —— 见 ``docs/BACKLOG.md``。

    Returns
    -------
    list[SegmentFinding]
        **只有标记，没有修改。** 轨迹一个数都没动。
    """
    T = len(traj)
    out: List[SegmentFinding] = []
    if T == 0:
        return out
    seg = max(2, int(round(seg_sec * fps)))
    valid_all = np.isfinite(traj[:, 0])

    for s in range(0, T, seg):
        e = min(T, s + seg)
        vi = np.where(valid_all[s:e])[0] + s
        if len(vi) < min_valid:
            continue

        # ── 1. 手腕 → 该段中间帧的坐标系 ──────────────────────────────────────
        mid = int(vi[len(vi) // 2])
        R_mid = _quat_wxyz_to_rotmat(traj[mid, 3:])
        if not np.isfinite(R_mid).all():
            continue
        rel = (traj[vi, :3] - traj[mid, :3]) @ R_mid          # = R_midᵀ · Δp，逐行
        z_w = _robust_z(np.linalg.norm(rel, axis=1))
        flag_w = z_w > z_thresh
        if flag_w.any():
            out.append(SegmentFinding(
                side=side, start=int(vi[0]), end=int(vi[-1]), kind="wrist_outlier",
                n_flag=int(flag_w.sum()), score=round(float(z_w.max()), 3)))

        # ── 2. 手指关键点 → 逐帧手腕系 ────────────────────────────────────────
        if joints is None:
            continue
        loc = np.full((len(vi), joints.shape[1], 3), np.nan)
        for k, t in enumerate(vi):
            R_t = _quat_wxyz_to_rotmat(traj[t, 3:])
            if np.isfinite(R_t).all():
                loc[k] = (joints[t] - traj[t, :3]) @ R_t
        ok = np.isfinite(loc).all(axis=(1, 2))
        if ok.sum() < min_valid:
            continue
        # 每个关键点单独算离群（某一根手指抖了不该被 20 个稳定关键点平均掉），
        # 再对关键点取最大值当这一帧的分数。
        d = np.linalg.norm(loc[ok] - np.median(loc[ok], axis=0)[None], axis=2)
        z_f = np.max(np.stack([_robust_z(d[:, j]) for j in range(d.shape[1])], 1), axis=1)
        flag_f = z_f > z_thresh
        if flag_f.any():
            fi = vi[ok]
            out.append(SegmentFinding(
                side=side, start=int(fi[0]), end=int(fi[-1]), kind="finger_outlier",
                n_flag=int(flag_f.sum()), score=round(float(z_f.max()), 3)))
    return out


__all__ = ["TIER_NAMES", "DEFAULT_TIERS", "parse_tiers",
           "EpisodeReport", "episode_camera_check",
           "SegmentFinding", "segment_spatial_check"]
