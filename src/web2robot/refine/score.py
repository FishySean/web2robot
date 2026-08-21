"""逐帧物体位姿跟踪误差 —— EgoEngine §3.2.2 的 e_t 和 r_obj_t，照论文的形式写。

论文原式（符号照抄）::

    e_t      = √( λp · dp(trans(T̂ot), trans(Tot))²  +  λR · dR(rot(T̂ot), rot(Tot))² )
    r_obj_t  = C − √( λR (eR_t)² + λp (ep_t)² )

其中 ``dp`` 是平移的欧氏距离，``dR`` 是 SO(3) 上的测地距离（两个旋转之间的夹角，
弧度）。``T̂ot`` 是执行动作之后的物体位姿，``Tot`` 是参考（模块一的数字孪生）。

**论文没给 λp / λR 的数值**，所以这里做成参数，默认都是 1.0 —— 即"1 米的平移误差
和 1 弧度的旋转误差一样严重"。这个默认值是个约定不是结论：1 rad ≈ 57°，而 1 m 的
物体位移是灾难性的，所以默认值实际上偏向宽容旋转。要调就调 :class:`ErrorWeights`，
别去改公式。

r_obj_t 里那个 C 和"早停阈值 C"在论文里是同一个字母，但一个是逐步奖励的偏移常数、
一个是累计误差的上限，量纲都不一样。这里把两者拆成两个参数（``c_reward`` 和
:class:`~web2robot.refine.blocks.RefineConfig` 的 ``per_frame_budget``），拆的理由
写在 :mod:`~web2robot.refine.blocks` 里。

纯 numpy，无 IO，无上游 import。坏帧（NaN 位姿、或者孪生标了不可信）产出 NaN 而不是
抛异常 —— 逐帧数据里出坏帧是常态，整段中断会把其余可用帧一起丢掉。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from web2robot.twin.object_pose import posquat_to_mats


@dataclass(frozen=True)
class ErrorWeights:
    """e_t 里的 λp / λR。**论文没给数值**，默认 1.0 / 1.0，单位是 m 和 rad。"""

    lam_p: float = 1.0
    lam_R: float = 1.0

    def __post_init__(self):
        if self.lam_p < 0 or self.lam_R < 0:
            raise ValueError(f"λ 不能是负的：lam_p={self.lam_p} lam_R={self.lam_R}")


def _quat_geodesic(qa: np.ndarray, qb: np.ndarray) -> np.ndarray:
    """两串四元数（wxyz）之间的 SO(3) 测地距离 (T,)，弧度，值域 [0, π]。

    用 ``|<qa,qb>|`` 而不是 ``<qa,qb>`` —— q 和 −q 是同一个旋转，不取绝对值的话
    符号一翻角度就跳成 2π−θ。旋转矩阵那条路（``arccos((tr−1)/2)``）在小角度上数值
    更差，这里不用。
    """
    qa = np.asarray(qa, dtype=np.float64)
    qb = np.asarray(qb, dtype=np.float64)
    na = np.linalg.norm(qa, axis=-1)
    nb = np.linalg.norm(qb, axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        dot = np.abs(np.sum(qa * qb, axis=-1) / (na * nb))
    dot = np.clip(dot, -1.0, 1.0)
    ang = 2.0 * np.arccos(dot)
    bad = ~np.isfinite(qa).all(-1) | ~np.isfinite(qb).all(-1) | (na < 1e-8) | (nb < 1e-8)
    return np.where(bad, np.nan, ang)


def step_errors(ref: np.ndarray, ach: np.ndarray,
                valid: Optional[np.ndarray] = None,
                weights: ErrorWeights = ErrorWeights()) -> Dict[str, np.ndarray]:
    """参考位姿 vs 执行后位姿 → 逐帧误差。

    Parameters
    ----------
    ref, ach
        ``(T, 7)``，``[x, y, z, qw, qx, qy, qz]``（和模块一、和 ``root_frames.npz``
        同一套口径）。必须在**同一个坐标系**里 —— 这个函数没法检查这一点，
        所以换系的活在 :mod:`~web2robot.refine.attach` 里做完再进来。
    valid
        ``(T,)`` bool。孪生标了不可信的帧、或者手没检出的帧传 False，对应的误差
        直接是 NaN。不传就只按数值是否有限判。

    Returns
    -------
    dict
        ``ep`` 平移误差 (T,) 米；``eR`` 旋转误差 (T,) 弧度；``e`` 加权合成 (T,)。
        无效帧是 NaN —— 用 0 填是错的，那会被下游当成"这帧完美"。
    """
    ref = np.asarray(ref, dtype=np.float64)
    ach = np.asarray(ach, dtype=np.float64)
    if ref.shape != ach.shape or ref.ndim != 2 or ref.shape[1] != 7:
        raise ValueError(f"ref/ach 都要 (T,7) 且形状相同，收到 {ref.shape} / {ach.shape}")
    ep = np.linalg.norm(ach[:, :3] - ref[:, :3], axis=1)
    eR = _quat_geodesic(ach[:, 3:], ref[:, 3:])
    finite = np.isfinite(ep) & np.isfinite(eR)
    if valid is not None:
        valid = np.asarray(valid, dtype=bool)
        if valid.shape != (len(ref),):
            raise ValueError(f"valid 要 ({len(ref)},)，收到 {valid.shape}")
        finite = finite & valid
    e = np.sqrt(weights.lam_p * ep ** 2 + weights.lam_R * eR ** 2)
    nan = np.full(len(ref), np.nan)
    return {"ep": np.where(finite, ep, nan),
            "eR": np.where(finite, eR, nan),
            "e": np.where(finite, e, nan)}


def reward(e: np.ndarray, c_reward: float) -> np.ndarray:
    """r_obj_t = C − e_t，论文 §3.2.2 的逐步奖励。

    这里不做 clip —— 论文没写下界，人为截断会把"差得离谱"和"差一点"抹成一样，
    而升级判决恰恰要靠那个差别。
    """
    return float(c_reward) - np.asarray(e, dtype=np.float64)


def pose_matrices(poses: np.ndarray) -> np.ndarray:
    """``(T,7)`` → ``(T,4,4)``，直接复用模块一的实现，不另写一份。"""
    return posquat_to_mats(poses)
