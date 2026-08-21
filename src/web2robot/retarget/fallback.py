"""坏帧/丢帧兜底在重定向流水线里的**编排**（不是判据本身）。

判据和填补算法在 ``web2robot/trajectory/traj_cleanup.py``；这里只负责"在流水线的
哪一步、拿什么调它、结果怎么往下传"。分成两个文件是因为它们的变更理由不同：
判据会因为感知前端换代而调（HaWoR/WiLoR 的爆点长相不一样），编排会因为重定向框架
换代而调。

## 兜底分两处，不是一处

上游 ``SamplesSequence.get_window()`` 用**无上限的零阶保持**填检测空洞，而且
"检测到了但离谱"的位姿原样放过。所以：

1. **输入侧**（``clean_input_wrists``，IK 之前）—— 从带 NaN 的原始轨迹重做填补：
   感知爆点先打成缺失，再按空洞长度决定策略，每一帧的来路都记下来。
2. **输出侧**（``apply_rest_fallback``，IK 之后、关节空间）—— 长空洞（``FILL_REST``）
   没有可信目标，与其把手臂冻在最后一次（往往已经在退化的）检测上，不如渐入机器人
   的自然静息位，重新捕获时再渐出。手指同步放松（``relax_fingers_on_rest``）：
   手臂垂在身侧却还攥着上一次检测到的抓握手型，看起来像握着一个不存在的东西。

静息位是**最后兜底**，能不触发就不触发（2026-08-11 定的策略）：``FILL_REST`` 现在只
出现在"中间的长空洞"和"开头的长空洞"两种情况；**片尾的空洞一律沿袭最后一帧**
（``hold_tail=True``），因为后面没有帧了、保持不会造成任何跳变，而渐入静息位反而是在
最没有信息的地方编一段大幅运动。片尾保持的长度会记在 ``report["tail_hold"]`` 里
（帧仍标 ``FILL_HOLD``，不会冒充 ``OK``），所以"这段是保持出来的"依然传得下去。

顺序不能换：输入侧要在建 IK 之前（它改的是喂给 IK 的目标），输出侧要在 IK 之后
（它改的是解出来的关节角），而 ``FILL_REST`` 这个标记是前者产生、后者消费的。

2026-08-10 从上游 ``scripts/test.py`` 里搬出来（原来是 run() 中间的三段内联代码）。

## 2026-08-21：多了两个**只看不动**的粒度

``run_extra_tiers`` 跑 EgoSmith 的整段/轨迹段两层（判据在
``trajectory/tiers.py``）。它和上面两处的性质**完全不同**：那两处会改喂给 IK 的目标
和解出来的关节角，这一处**一个数都不改** —— 整段级只打警告（视频照样跑完），
轨迹段级只写标记（留给 ``refine/`` 自己决定要不要升级精修）。所以它在流水线里放哪
都不影响产物，选在"原始轨迹拿到手、视频帧也读进来了"之后，是为了让警告尽早打出来。
"""
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Sequence

import numpy as np

from web2robot.trajectory.traj_cleanup import (
    FILL_REST, STATUS_NAMES, blend_to_rest, clean_wrist_trajectory, relax_fingers,
)
from web2robot.trajectory.tiers import (
    episode_camera_check, segment_spatial_check,
)


@dataclass
class InputCleanup:
    """输入侧清洗的结果。``status`` / ``cause`` 是每帧来路，会写进 trajectory.npz。"""
    left:         np.ndarray
    right:        np.ndarray
    status_left:  np.ndarray
    status_right: np.ndarray
    cause_left:   np.ndarray
    cause_right:  np.ndarray
    report_left:  dict
    report_right: dict


def clean_input_wrists(
    raw_left:  np.ndarray,
    raw_right: np.ndarray,
    fps:       float,
    max_interp_sec: float = 2.5,
    max_hold_sec:   float = 0.5,   # 只管开头空洞
    hold_tail:      bool  = True,  # 结尾空洞：沿袭最后一帧，不论多长
    detect_bad:     bool  = True,
    log: Callable[[str], None] = print,
) -> InputCleanup:
    """两只手腕轨迹一起清洗。输入必须是**带 NaN 的原始轨迹**。

    拿 ``SamplesSequence.raw_wrist_trajectories()``，**不要**拿 ``get_window()`` ——
    后者已经把空洞用零阶保持填平了，填过的帧和真检测到的帧再也分不出来。

    Raises
    ------
    RuntimeError
        有一只手整段都没检测到。根位姿估计器同时吃两只手腕，凭空造一只会把躯干锚点
        带偏，所以这种片段直接拒掉，而不是编一只出来。整段单手支持不在这个兜底的
        范围里 —— 那是质检该筛掉的东西。
    """
    log("Bad/missing frame fallback:")
    kw = dict(max_interp_sec=max_interp_sec, max_hold_sec=max_hold_sec,
              hold_tail=hold_tail, detect_bad=detect_bad)
    left,  st_l, ca_l, rep_l = clean_wrist_trajectory(raw_left,  fps, side="left",  **kw)
    right, st_r, ca_r, rep_r = clean_wrist_trajectory(raw_right, fps, side="right", **kw)
    for name, traj in (("left", left), ("right", right)):
        if np.isnan(traj[:, 0]).all():
            raise RuntimeError(
                f"{name} hand is never detected in this clip — the root-frame "
                f"estimator needs both wrists. Whole-clip single-hand support is "
                f"out of scope for the gap fallback; screen this clip out instead.")
    return InputCleanup(left, right, st_l, st_r, ca_l, ca_r, rep_l, rep_r)


def apply_rest_fallback(
    q_left:  np.ndarray,
    q_right: np.ndarray,
    status_left:  np.ndarray,
    status_right: np.ndarray,
    rest_config: dict,
    fps:      float,
    ramp_sec: float = 0.5,
    log: Callable[[str], None] = print,
):
    """长空洞上把手臂渐入静息位。返回 ``(q_left, q_right, w_left, w_right)``。

    ``w_*`` 是每帧的静息权重（0=完全信原轨迹，1=完全静息位），余弦从两侧渐入，
    所以两个边界都不跳。手指要用**同一套权重**才能跟手臂同步张开/收回，
    所以权重是返回值而不是内部变量 —— 交给 :func:`relax_fingers_on_rest`。

    ``rest_config`` 是 ``CONFIG["start_config"]``，形如 ``{"left": (n,), "right": (n,)}``。
    """
    T = len(q_left)
    w_left, w_right = np.zeros(T), np.zeros(T)
    any_rest = (status_left == FILL_REST).any() or (status_right == FILL_REST).any()
    if not any_rest:
        return q_left, q_right, w_left, w_right

    log("Rest-pose fallback:")
    for side, q, st in (("left", q_left, status_left), ("right", q_right, status_right)):
        if not (st == FILL_REST).any():
            continue
        q_new, w = blend_to_rest(q, st, np.asarray(rest_config[side], np.float64),
                                 fps, ramp_sec=ramp_sec)
        if side == "left":
            q_left, w_left = q_new, w
        else:
            q_right, w_right = q_new, w
        log(f"  {side:<5}: {int((st == FILL_REST).sum())} frames → rest "
            f"(max weight {w.max():.2f})")
    return q_left, q_right, w_left, w_right


def relax_fingers_on_rest(Q_left, Q_right, w_left, w_right):
    """静息帧上把手指放松成中立张开手型，用和手臂**同一套**余弦权重。"""
    if w_left is not None and np.any(w_left):
        Q_left = relax_fingers(Q_left, w_left)
    if w_right is not None and np.any(w_right):
        Q_right = relax_fingers(Q_right, w_right)
    return Q_left, Q_right


def status_overlay_text(status_left, status_right, t: int) -> str:
    """输入可视化上第 t 帧的来路标注，例如 ``"L:rest R:interp"``。

    正常帧（两只手都是 ``ok``）返回空串 —— 逐帧都标"ok"会把画面糊住，而且看片的人
    要找的正是**不正常**的那几段。
    """
    tags = [f"{tag}:{STATUS_NAMES[int(arr[t])]}"
            for tag, arr in (("L", status_left), ("R", status_right))
            if arr is not None and int(arr[t]) != 0]
    return " ".join(tags)


# ── EgoSmith 的另外两个粒度（整段 / 轨迹段）：只看不动 ─────────────────────────

def run_extra_tiers(
    tiers:      Sequence[str],
    raw_left:   np.ndarray,           # (T,7) 带 NaN 的原始左手腕轨迹（相机系）
    raw_right:  np.ndarray,
    fps:        float,
    frames:     Optional[Sequence[np.ndarray]] = None,   # 逐帧图像，整段级要
    joints_left:  Optional[np.ndarray] = None,           # (T,21,3) 相机系关键点
    joints_right: Optional[np.ndarray] = None,
    seg_sec:    float = 2.0,
    log: Callable[[str], None] = print,
) -> Dict[str, Any]:
    """跑 ``tiers`` 里除 ``frame`` 以外的粒度，返回可直接写 json 的 report。

    ``frame`` 那一层**不在这里** —— 它是 :func:`clean_input_wrists` 的一部分
    （``detect_bad=True``），因为只有它会改数据、要在 IK 之前就位。这里只跑两个
    "只看不动"的粒度，所以调用点放哪都不影响产物。

    返回 ``{}`` 表示两层都没开 —— 调用方据此决定**不写**报告文件，默认跑法的产物
    因此和以前逐字节一样（多一个文件也是"变了"）。
    """
    want = set(tiers)
    report: Dict[str, Any] = {}
    if not (want & {"episode", "segment"}):
        return report

    report["tiers"] = [t for t in ("episode", "segment", "frame") if t in want]

    if "episode" in want:
        if not frames:
            report["episode"] = {"warn": False, "reason": "没有视频帧可用，整段级跳过"}
            log("  [tier:episode] 没有视频帧可用 → 跳过")
        else:
            ep = episode_camera_check(frames)
            report["episode"] = ep.to_dict()
            # 只警告，不阻断（策略见 trajectory/tiers.py 的 docstring）
            log(f"  [tier:episode] {'⚠ ' if ep.warn else ''}{ep.reason}")
            if ep.warn:
                log("      —— 依据 VIDEO_SELECTION_GUIDE.md §V2/§V3。"
                    "这段视频**照样跑完**，弃不弃用请人看画面决定")

    if "segment" in want:
        marks = []
        for side, traj, jts in (("left", raw_left, joints_left),
                                ("right", raw_right, joints_right)):
            marks += [f.to_dict() for f in segment_spatial_check(
                traj, jts, fps, side=side, seg_sec=seg_sec)]
        report["segment"] = {"seg_sec": seg_sec, "findings": marks,
                             "clause": "§V2/§V3"}
        n_w = sum(1 for m in marks if m["kind"] == "wrist_outlier")
        n_f = sum(1 for m in marks if m["kind"] == "finger_outlier")
        log(f"  [tier:segment] {len(marks)} 处空间离群标记"
            f"（手腕 {n_w} / 手指 {n_f}）—— **不做任何自动处理**，"
            f"留给 refine 决定要不要升级精修")
    return report


def save_tier_report(path, report: Dict[str, Any], clip: str = "",
                     robot: str = "") -> None:
    """把 :func:`run_extra_tiers` 的 report 写成 json。

    json 而不是 npz：这份东西是给人看的警告 + 给下游读的帧区间，字段是嵌套字典，
    npz 存不下（会退化成 0-d object 数组，读的人得 ``.item()``）。
    """
    import json
    from pathlib import Path

    out = dict(report)
    out["clip"] = clip
    out["robot"] = robot
    out["source"] = ("EgoSmith / EgoSteer arXiv 2607.09701 post-filtering；"
                     "判据依据 docs/VIDEO_SELECTION_GUIDE.md §V2/§V3；"
                     "处置策略（只警告/只标记）是我方决定，与原文的 discard 不同")
    Path(path).write_text(json.dumps(out, ensure_ascii=False, indent=1))


__all__ = ["InputCleanup", "clean_input_wrists", "apply_rest_fallback",
           "relax_fingers_on_rest", "status_overlay_text",
           "run_extra_tiers", "save_tier_report"]
