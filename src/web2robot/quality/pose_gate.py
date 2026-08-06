"""Framing-completeness gate: a coarse body-pose pass over sampled RGB frames.

Model: torchvision's KeypointRCNN (ResNet50-FPN, COCO-17 person keypoints).
Chosen because torchvision is already installed in every env here and its
weights come from download.pytorch.org, so this stage needs no new packages on
a shared machine. It is a 2017 architecture and roughly 5-10x slower than
YOLO11-pose, which is an acceptable trade for zero install; swapping in a
faster backend only requires reproducing `PoseFrame`.

The criterion is deliberately loose (user, 2026-08-05): both wrists + both
elbows + at least one torso keypoint. Head keypoints are NOT required -- footage
where the camera missed the head is still usable, and requiring the head
false-rejects it. Measured counter-example: a frame with head 5/5 and torso 4/4
but no wrists is useless, so head presence carries no usability information.
"""
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import os

import numpy as np

from ..paths import P

COCO = ["nose", "eyeL", "eyeR", "earL", "earR", "shoL", "shoR", "elbL", "elbR",
        "wriL", "wriR", "hipL", "hipR", "kneL", "kneR", "ankL", "ankR"]
WRIST = (9, 10)
ELBOW = (7, 8)
TORSO = (5, 6, 11, 12)      # shoulders + hips; "chest region" is between the shoulders
HEAD = (0, 1, 2, 3, 4)
LIMBS = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12),
         (0, 5), (0, 6)]

_MODEL = None
_MODEL_DEV = None


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #
def pick_device(spec: str = "auto") -> str:
    """'auto' -> the idlest visible GPU, else cpu."""
    if spec != "auto":
        return spec
    try:
        import torch
        if not torch.cuda.is_available():
            return "cpu"
        free = []
        for i in range(torch.cuda.device_count()):
            f, _ = torch.cuda.mem_get_info(i)
            free.append((f, i))
        return f"cuda:{max(free)[1]}"
    except Exception:
        return "cpu"


def get_model(device: str = "auto", det_score_thresh: float = 0.7):
    """Lazily build the pose model, once per process."""
    global _MODEL, _MODEL_DEV
    os.environ.setdefault("TORCH_HOME", str(P.torch_home()))
    dev = pick_device(device)
    if _MODEL is not None and _MODEL_DEV == dev:
        return _MODEL, dev
    import torch
    from torchvision.models.detection import (
        keypointrcnn_resnet50_fpn, KeypointRCNN_ResNet50_FPN_Weights)
    m = keypointrcnn_resnet50_fpn(
        weights=KeypointRCNN_ResNet50_FPN_Weights.DEFAULT,
        box_score_thresh=det_score_thresh)
    _MODEL, _MODEL_DEV = m.eval().to(dev), dev
    return _MODEL, dev


# --------------------------------------------------------------------------- #
# per-frame result
# --------------------------------------------------------------------------- #
@dataclass
class PoseFrame:
    frame_idx: int
    found: bool = False
    kps: Optional[np.ndarray] = None        # (17,3) x,y,logit-ish
    kscore: Optional[np.ndarray] = None     # (17,)
    box: Optional[np.ndarray] = None        # (4,)
    det: float = 0.0

    # derived under a visibility threshold
    n_wrist: int = 0
    n_elbow: int = 0
    n_torso: int = 0
    n_head: int = 0
    framing_ok: bool = False
    box_frac: float = 0.0
    forearm_frac: float = 0.0       # |wrist-elbow| / frame height
    elbow_at_bottom: bool = False   # egocentric signature: own arms enter from below

    def why(self) -> List[str]:
        r = []
        if not self.found:
            return ["no_person"]
        if self.n_wrist < 2:
            r.append("no_stable_hands")
        if self.n_elbow < 2:
            r.append("no_forearm")
        if self.n_torso < 1:
            r.append("no_torso")
        return r


def _derive(pf: PoseFrame, thr: float, W: int, H: int) -> PoseFrame:
    vis = pf.kscore >= thr
    pf.n_wrist = int(sum(vis[i] for i in WRIST))
    pf.n_elbow = int(sum(vis[i] for i in ELBOW))
    pf.n_torso = int(sum(vis[i] for i in TORSO))
    pf.n_head = int(sum(vis[i] for i in HEAD))
    pf.framing_ok = bool(pf.n_wrist == 2 and pf.n_elbow == 2 and pf.n_torso >= 1)

    b = pf.box
    pf.box_frac = float((b[2] - b[0]) * (b[3] - b[1]) / (W * H))

    # forearm length in image units: a close-up / egocentric proxy that needs no depth
    lens = []
    for e, w in ((7, 9), (8, 10)):
        if vis[e] and vis[w]:
            lens.append(float(np.linalg.norm(pf.kps[e, :2] - pf.kps[w, :2])))
    pf.forearm_frac = float(np.median(lens) / H) if lens else 0.0

    ys = [pf.kps[e, 1] for e in ELBOW if vis[e]]
    pf.elbow_at_bottom = bool(ys and min(H - y for y in ys) < 0.05 * H)
    return pf


def select_actor(boxes, scores, keypoints, kscores) -> int:
    """Pick which detected person is the one manipulating.

    KNOWN LIMITATION (accepted 2026-08-05, fix when a real case appears):
    this takes the highest-confidence detection. In a scene with a bystander
    fully in frame while the actual manipulator is a disembodied hand at the
    edge, it will pick the bystander and the clip can be falsely accepted.
    None of the 7 clips measured so far contains a bystander, so the risk is
    real but UNTESTED. The fix is to prefer the person whose wrists are nearest
    the detected hands -- and since 2026-08-05 that costs nothing new: the hand
    detector is wired in (hand_gate) and runs on the same frames. Still deferred
    per the user's call: change it when a real bystander case shows up, not on
    speculation.
    """
    return int(np.argmax(scores))


# --------------------------------------------------------------------------- #
# main entry
# --------------------------------------------------------------------------- #
def run_pose(frames: List[Tuple[int, "np.ndarray"]], cfg, batch: int = 4
             ) -> List[PoseFrame]:
    """frames: [(frame_index, BGR uint8 HxWx3)] -> one PoseFrame each."""
    import torch
    model, dev = get_model(cfg.device, cfg.det_score_thresh)
    out: List[PoseFrame] = []
    with torch.no_grad():
        for i in range(0, len(frames), batch):
            chunk = frames[i:i + batch]
            tens = [torch.from_numpy(img[:, :, ::-1].copy())
                    .permute(2, 0, 1).float().div_(255.).to(dev)
                    for _, img in chunk]
            res = model(tens)
            for (fi, img), r in zip(chunk, res):
                pf = PoseFrame(frame_idx=fi)
                if len(r["boxes"]):
                    j = select_actor(r["boxes"].cpu().numpy(),
                                     r["scores"].cpu().numpy(),
                                     r["keypoints"].cpu().numpy(),
                                     r["keypoints_scores"].cpu().numpy())
                    pf.found = True
                    pf.det = float(r["scores"][j])
                    pf.box = r["boxes"][j].cpu().numpy()
                    pf.kps = r["keypoints"][j].cpu().numpy()
                    pf.kscore = r["keypoints_scores"][j].cpu().numpy()
                    H, W = img.shape[:2]
                    _derive(pf, cfg.kpt_score_thresh, W, H)
                out.append(pf)
    return out


def longest_run(pfs: List[PoseFrame], pred, tol: int = 1, fps: float = 0.0,
                min_density: float = 0.6) -> Tuple[float, Optional[List[float]]]:
    """Longest contiguous stretch of frames satisfying `pred`.

    Returns (estimated seconds, [t_start, t_end] or None).

    ESTIMATED, and the name says so on purpose: the samples are spread out, so a
    run of k consecutive passing SAMPLES only implies the footage between them
    passes -- it is an interpolation, not a measurement. It is still far better
    than a whole-file average, which measures the editing rather than the
    footage (see QCConfig.min_usable_sec).

    `tol` failing samples inside a run do not break it, but do not extend it
    either: a span always starts and ends on a passing sample. `min_density`
    then rejects a span that only survived on tolerance -- with tol=1 alone an
    alternating `h.h.h.h` would be reported as one fully usable stretch, which
    it is not.
    """
    if not pfs or fps <= 0:
        return 0.0, None
    runs = []                        # (i0, i1, n_pass)
    i0 = last = None
    npass = miss = 0
    for i, p in enumerate(pfs):
        if pred(p):
            if i0 is None:
                i0, npass = i, 0
            last, miss = i, 0
            npass += 1
        elif i0 is not None:
            miss += 1
            if miss > tol:
                runs.append((i0, last, npass))
                i0, miss = None, 0
    if i0 is not None:
        runs.append((i0, last, npass))
    ok = [r for r in runs if r[2] / max(1, r[1] - r[0] + 1) >= min_density]
    if not ok:
        return 0.0, None
    a, b, _ = max(ok, key=lambda r: r[1] - r[0])
    t0, t1 = pfs[a].frame_idx / fps, pfs[b].frame_idx / fps
    return round(t1 - t0, 2), [round(t0, 2), round(t1, 2)]


def aggregate(pfs: List[PoseFrame], fps: float = 0.0, cfg=None) -> dict:
    """Clip-level framing statistics, plus the longest contiguous usable span.

    Both are reported. The rates describe the whole file; the spans describe
    where the usable footage actually is. For edited web video the two say very
    different things and the span is the one that matters downstream.
    """
    n = max(1, len(pfs))
    tol = getattr(cfg, "span_gap_tol", 1)
    dens = getattr(cfg, "min_run_density", 0.6)
    both_wrist = sum(p.n_wrist == 2 for p in pfs)
    any_wrist = sum(p.n_wrist >= 1 for p in pfs)
    body_sec, body_span = longest_run(pfs, lambda p: p.framing_ok, tol, fps, dens)
    wrist_sec, wrist_span = longest_run(pfs, lambda p: p.n_wrist == 2, tol, fps, dens)
    idxs = [p.frame_idx for p in pfs]
    step = ((idxs[-1] - idxs[0]) / max(1, len(idxs) - 1) / fps) if fps > 0 and len(idxs) > 1 else None
    return dict(
        n_sampled=len(pfs),
        sample_step_sec=round(step, 2) if step else None,
        person_rate=round(sum(p.found for p in pfs) / n, 4),
        body_frame_rate=round(sum(p.framing_ok for p in pfs) / n, 4),
        both_wrist_rate=round(both_wrist / n, 4),
        any_wrist_rate=round(any_wrist / n, 4),
        both_elbow_rate=round(sum(p.n_elbow == 2 for p in pfs) / n, 4),
        torso_rate=round(sum(p.n_torso >= 1 for p in pfs) / n, 4),
        head_rate=round(sum(p.n_head >= 1 for p in pfs) / n, 4),
        # temporal structure: where the usable footage is (interpolated, see longest_run)
        body_span_est_sec=body_sec,
        body_span_est=body_span,
        # reported only. Named wrist_*, not hands_*, so it cannot be mistaken for
        # the hand detector's span: the both-wrist statistic is FALSIFIED on
        # hands-only footage (hand_gate docstring) and decides nothing.
        wrist_span_est_sec=wrist_sec,
        wrist_span_est=wrist_span,
        # egocentric-prior evidence, recorded for step 2; NOT used in the verdict
        forearm_frac_med=round(float(np.median(
            [p.forearm_frac for p in pfs if p.forearm_frac > 0] or [0.0])), 4),
        elbow_at_bottom_rate=round(sum(p.elbow_at_bottom for p in pfs) / n, 4),
        box_frac_med=round(float(np.median(
            [p.box_frac for p in pfs if p.found] or [0.0])), 4),
    )


def classify_framing(agg: dict, cfg, hands: Optional[dict] = None
                     ) -> Tuple[str, List[str]]:
    """Three-way framing verdict. Returns (ViewClass value, reason codes).

    The three-way split matters: a binary pass/fail here would reject every
    first-person clip, and first-person is a VALID route in pipeline step 2 --
    a binary gate would starve that branch of input.

    TWO DIFFERENT INSTRUMENTS, one per boundary, because measurement showed each
    is only trustworthy on one of them:

      THIRD_PERSON_BODY   decided by THIS body-pose gate. A full body is what
                          KeypointRCNN was trained on; measured 0.83 on
                          cup_SUDRM with the frames visually confirmed.
      HANDS_ONLY vs
      NO_STABLE_HANDS     decided by the HAND DETECTOR (hand_gate). The body
                          gate's wrist keypoints are INVERTED on hands-only
                          footage -- see hand_gate's module docstring for the
                          paired-control measurement. Wrist counts are recorded
                          but must not decide this boundary.

    Each class can also be reached by a long enough contiguous span rather than
    a whole-file rate, because scraped video is edited and a whole-file average
    measures the editing, not the footage (see QCConfig.min_usable_sec).
    """
    from .schema import ViewClass
    hands = hands or {}
    h_avail = bool(hands.get("available"))

    if agg["person_rate"] == 0.0 and (not h_avail or hands.get("any_hand_rate") == 0.0):
        return ViewClass.NO_STABLE_HANDS.value, ["no_person"]

    if (agg["body_frame_rate"] >= cfg.body_frame_rate_min
            or agg["body_span_est_sec"] >= cfg.min_usable_sec):
        return ViewClass.THIRD_PERSON_BODY.value, []

    if not h_avail:
        # No hand detector: the body gate cannot settle this boundary, and
        # guessing with an inverted signal is worse than admitting it.
        reasons = ["hand_gate_unavailable"]
        if agg["torso_rate"] == 0.0:
            reasons.append("no_torso")
        return ViewClass.UNKNOWN.value, reasons

    if (hands["both_hand_rate"] >= cfg.both_hand_rate_min
            or (hands["hands_span_est_sec"] or 0.0) >= cfg.min_usable_sec):
        reasons = ["hands_only"]
        if agg["torso_rate"] == 0.0:
            reasons.append("no_torso")
        if hands["any_hand_rate"] < cfg.min_hand_ratio:
            reasons.append("low_hand_ratio")       # recorded, not fatal
        if hands["trunc_ratio"] > cfg.max_trunc_ratio:
            reasons.append("hand_truncated")       # warn only
        return ViewClass.HANDS_ONLY.value, reasons

    reasons = ["no_stable_hands"]
    if hands["any_hand_rate"] < cfg.min_hand_ratio:
        reasons.append("low_hand_ratio")
    if agg["torso_rate"] == 0.0:
        reasons.append("no_torso")
    return ViewClass.NO_STABLE_HANDS.value, reasons
