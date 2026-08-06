"""Hand presence gate -- the instrument that can actually count hands.

WHY THIS EXISTS (measured, 2026-08-05)
--------------------------------------
The body-pose gate's wrist keypoints cannot count hands on hands-only footage.
Two paired controls were built BY CONSTRUCTION from one top-down shirt-folding
shot -- `neg_onehand` blacks out the right half of the frame so exactly one hand
can be visible, `pos_twohands` is the same footage untouched:

    detector thresh   pos_twohands (2 hands)   neg_onehand (1 hand)
    0.7               both_wrist 0.21          both_wrist 0.25
    0.3               both_wrist 0.29          both_wrist 0.33
    0.1               both_wrist 0.29          both_wrist 0.38
    0.05              both_wrist 0.29          both_wrist 0.38

The one-hand clip scores HIGHER at every threshold: the signal is inverted, not
merely noisy, so no threshold can fix it. KeypointRCNN is a person detector and
a pair of disembodied hands is outside its design range -- it both misses hands
that are plainly visible and invents a second wrist that is not there.

WiLoR's YOLO hand detector separates the same controls cleanly:

    clip            ground truth   body both_wrist   THIS both_hand_rate
    pos_twohands    2 hands        0.21              0.42
    neg_onehand     1 hand         0.25              0.00
    neg_noperson    none           0.00              0.00
    cand_knit       2 hands        0.42              0.72
    cup_SUDRM       2 hands        0.96              1.00

(this gate's column is AFTER duplicate merging -- see merge_duplicates; before
it neg_onehand read 0.04 and cand_knit 0.83, inflated by the same hand counted
twice)

So: the body gate decides THIRD_PERSON_BODY (its home turf -- a full body is
what it was trained on), and this gate decides hands-only vs unusable.

Zero install: `ultralytics` is already in rt_env and the weights are already on
disk (HaWoR/weights/external/detector.pt, same file as
EgoInfinity/pretrained_models/detector.pt -- the official prefilter's detector).
路径由 configs/paths.yaml 的 weights.hand_detector 给出，不写死在代码里。
"""
from dataclasses import dataclass
from typing import List, Optional, Tuple
import os

import numpy as np

from .pose_gate import longest_run, pick_device
from ..paths import P

_HAND_MODEL = None
_HAND_DEV = None

# 候选路径来自 configs/paths.yaml 的 weights.hand_detector（两处是同一个文件）。
# 这里保留"候选列表"而不是单一路径：找不到时要能说清找过哪里。
WEIGHT_CANDIDATES = tuple(str(p) for p in P.weight_candidates("hand_detector"))


def find_weights(explicit: Optional[str] = None) -> Optional[str]:
    for p in ([explicit] if explicit else []) + list(WEIGHT_CANDIDATES):
        if p and os.path.exists(p):
            return p
    return None


def get_hand_model(weights: Optional[str] = None, device: str = "auto"):
    """Lazily load the YOLO hand detector, once per process.

    Returns (model, device) or (None, None) when the weights or ultralytics are
    missing -- the caller must then report the hand signals as None rather than
    as zero. An absent detector is 'unknown', not 'no hands' (decision 3).
    """
    global _HAND_MODEL, _HAND_DEV
    dev = pick_device(device)
    if _HAND_MODEL is not None and _HAND_DEV == dev:
        return _HAND_MODEL, dev
    w = find_weights(weights)
    if w is None:
        return None, None
    try:
        os.environ.setdefault("YOLO_VERBOSE", "False")
        from ultralytics import YOLO
        m = YOLO(w)
        m.to(dev)
    except Exception:
        return None, None
    _HAND_MODEL, _HAND_DEV = m, dev
    return m, dev


@dataclass
class HandFrame:
    frame_idx: int
    n_hands: int = 0
    boxes: Optional[np.ndarray] = None     # (n,4) xyxy, after duplicate merging
    sizes: Tuple[float, ...] = ()          # bbox area / frame area
    truncated: bool = False                # any box within edge_px of a border
    n_merged: int = 0                      # duplicate boxes dropped on this frame


def _truncated(xyxy: np.ndarray, h: int, w: int, edge_px: int = 5) -> bool:
    """Port of the official _is_truncated (detect.py)."""
    if len(xyxy) == 0:
        return False
    return bool(np.any((xyxy[:, 0] <= edge_px) | (xyxy[:, 1] <= edge_px)
                       | (xyxy[:, 2] >= w - edge_px) | (xyxy[:, 3] >= h - edge_px)))


def merge_duplicates(xyxy: np.ndarray, conf: np.ndarray, iou_thr: float = 0.10
                     ) -> np.ndarray:
    """Drop boxes that are a second detection of a hand already counted.

    The detector double-fires: it returned 3 boxes for 2 hands on one control
    frame, and 2 boxes for 1 hand on the blacked-out version of that same frame.
    Since `both_hand_rate` counts boxes, an uncorrected duplicate is a phantom
    hand and can push one-hand footage over the threshold.

    Overlap separates the two cases with nothing in between: genuine hand pairs
    measured IoU exactly 0.00 in 10/10 pairs (two hands cannot occupy the same
    pixels), while the duplicates measured 0.20 and 0.24. Keep the
    higher-confidence box of any overlapping group.

    Deliberately NOT capping at 2: a third-person clip may legitimately show a
    bystander's hands. Miscounting across people is the separate, recorded
    bystander limitation (pose_gate.select_actor), not this one.
    """
    if len(xyxy) < 2:
        return np.arange(len(xyxy))
    order = np.argsort(-conf)
    keep: List[int] = []
    for i in order:
        a = xyxy[i]
        dup = False
        for j in keep:
            b = xyxy[j]
            x1, y1 = max(a[0], b[0]), max(a[1], b[1])
            x2, y2 = min(a[2], b[2]), min(a[3], b[3])
            inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            if inter <= 0:
                continue
            ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
            if inter / max(ua, 1e-9) > iou_thr:
                dup = True
                break
        if not dup:
            keep.append(int(i))
    return np.array(sorted(keep))


def detect_hands(frames, cfg, batch: int = 8) -> Optional[List[HandFrame]]:
    """frames: [(frame_index, BGR)] -> one HandFrame each, or None if unavailable."""
    model, _ = get_hand_model(getattr(cfg, "hand_weights", None), cfg.device)
    if model is None:
        return None
    iou_thr = getattr(cfg, "hand_iou_merge", 0.10)
    out: List[HandFrame] = []
    for i in range(0, len(frames), batch):
        chunk = frames[i:i + batch]
        res = model([img for _, img in chunk], conf=cfg.hand_conf, verbose=False)
        for (fi, img), r in zip(chunk, res):
            h, w = img.shape[:2]
            xyxy = r.boxes.xyxy.cpu().numpy()
            conf = r.boxes.conf.cpu().numpy()
            n_raw = len(xyxy)
            if n_raw:
                sel = merge_duplicates(xyxy, conf, iou_thr)
                xyxy, conf = xyxy[sel], conf[sel]
            areas = ((xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1]) / (h * w)
                     if len(xyxy) else np.zeros(0))
            out.append(HandFrame(frame_idx=fi, n_hands=len(xyxy), boxes=xyxy,
                                 sizes=tuple(float(a) for a in areas),
                                 n_merged=n_raw - len(xyxy),
                                 truncated=_truncated(xyxy, h, w,
                                                      cfg.trunc_edge_px)))
    return out


def aggregate_hands(hfs: Optional[List[HandFrame]], fps: float, cfg) -> dict:
    """Clip-level hand statistics.

    `any_hand_rate` is the official `hand_ratio` (frames with >=1 hand), kept
    with that definition so our numbers stay comparable with theirs.
    `both_hand_rate` is ours: the official statistic cannot tell one hand from
    two (measured 0.54 vs 0.58 on the paired controls) and both wrists are what
    the root-frame estimator needs.
    """
    if hfs is None:
        return dict(available=False, any_hand_rate=None, both_hand_rate=None,
                    avg_hand_size=None, trunc_ratio=None, n_merged=None,
                    hands_span_est_sec=None, hands_span_est=None, n_sampled=0)
    n = max(1, len(hfs))
    sizes = [s for h in hfs for s in h.sizes]
    tol = getattr(cfg, "span_gap_tol", 1)
    dens = getattr(cfg, "min_run_density", 0.6)
    span_sec, span = longest_run(hfs, lambda h: h.n_hands >= 2, tol, fps, dens)
    return dict(
        available=True,
        n_sampled=len(hfs),
        any_hand_rate=round(sum(h.n_hands >= 1 for h in hfs) / n, 4),
        both_hand_rate=round(sum(h.n_hands >= 2 for h in hfs) / n, 4),
        mean_n_hands=round(float(np.mean([h.n_hands for h in hfs])), 3),
        avg_hand_size=round(float(np.mean(sizes)), 5) if sizes else 0.0,
        trunc_ratio=round(sum(h.truncated for h in hfs) / n, 4),
        n_merged=int(sum(h.n_merged for h in hfs)),
        hands_span_est_sec=span_sec,
        hands_span_est=span,
    )
