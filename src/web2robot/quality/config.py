"""Thresholds for the stage-1 gate.

Every number below carries its provenance. Three sources only:
  [OFFICIAL]  copied from EgoInfinity action100m_filter/config.py (tuned at 100M scale)
  [MEASURED]  set from a measurement recorded in QUALITY_DIAGNOSIS_DESIGN.md
  [GUESS]     a placeholder with no sample backing yet -- do not defend these

Do not silently promote a [GUESS] to fact. If you tune one, say what data moved it.
"""
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class QCConfig:
    # ---------- frame sampling ----------
    n_frames: int = 24
    """MINIMUM frames sampled per clip (short clips get exactly this).
    [MEASURED] the official prefilter uses 8; at 8 the framing pass rate has a
    12.5% quantisation step, which is too coarse to sit a 0.6 threshold on.
    24 gives a ~4% step at ~1.8s/clip on one GPU."""

    sample_every_sec: float = 3.0
    n_frames_max: int = 96
    """Long clips are sampled proportionally: n = clip(duration/sample_every_sec,
    n_frames, n_frames_max).
    [MEASURED] a fixed 24 samples is not enough on a long clip. cand_cards
    (849s) measured both_wrist_rate 0.42 at n=24 and 0.56 at n=72 -- the
    estimate moved across the 0.40 threshold on sampling noise alone. At p~0.5
    the binomial standard error is 0.10 for n=24, so a threshold that separates
    0.38 from 0.42 was inside its own noise. n_frames_max caps the cost at
    ~7s/clip on one GPU."""

    sample_lo: float = 0.08
    sample_hi: float = 0.92
    """Fraction of the clip to sample between; avoids intros/outros.
    [GUESS] -- not calibrated."""

    # ---------- pose gate (framing completeness) ----------
    kpt_score_thresh: float = 3.0
    """A COCO keypoint counts as visible above this KeypointRCNN score.
    [MEASURED] visible keypoints score 4.4..17.0 and out-of-frame ones -4.9..+2.0
    on the 7 scraped clips -- a ~2.4-wide empty band. 3.0 sits inside it.
    This is the single most load-bearing number in the module and it is the one
    with the widest measured margin."""

    det_score_thresh: float = 0.7
    """Person detection confidence. [OFFICIAL] mirrors the 0.7-ish operating
    point of the official YOLO prefilter (its hand conf is 0.3, but that is a
    hand detector on 8 frames; a body detector on 24 frames can afford strict)."""

    body_frame_rate_min: float = 0.60
    """Fraction of sampled frames that must satisfy the loose framing criterion
    (both wrists + both elbows + >=1 torso keypoint) to call a clip
    THIRD_PERSON_BODY.
    [GUESS] with one positive example. Measured separation on the 7 scraped
    clips is 1.00 vs 0.25 vs 0.00, so anything in 0.3..0.9 gives the same
    verdicts. Do not claim 0.60 is calibrated."""

    hands_only_rate_min: float = 0.40
    """DEPRECATED AS A GATE -- reported only, never decides anything.
    Was: fraction of frames with both WRIST KEYPOINTS visible, used to separate
    hands-only footage from unusable footage.
    [FALSIFIED 2026-08-05] On paired controls built by construction (right half
    of the frame blacked out vs untouched) the one-hand clip scored a HIGHER
    both-wrist rate than the two-hand clip at every detector threshold
    (0.25/0.33/0.38/0.38 vs 0.21/0.29/0.29/0.29). The signal is inverted, so no
    threshold value works. That job moved to hand_gate.both_hand_rate_min.
    Kept in the record because the rate is still worth reporting -- just not
    worth deciding on."""

    # ---------- hand gate (the instrument that can count hands) ----------
    hand_weights: Optional[str] = None
    """WiLoR's YOLO hand detector. None = 用 configs/paths.yaml 里
    weights.hand_detector 的候选列表（HaWoR 与 EgoInfinity 两处是同一个文件，
    也就是官方预筛自己用的检测器）；这里只在想临时指定别的权重时才填。
    用 rt_env 里已经装好的 ultralytics 就能加载，所以这一步在共享机器上
    仍然零安装。"""

    hand_conf: float = 0.3
    """YOLO confidence. [OFFICIAL] action100m_filter/config.py hand_conf."""

    hand_iou_merge: float = 0.10
    """Two boxes overlapping above this IoU are the SAME hand detected twice; the
    lower-confidence one is dropped before counting.
    [MEASURED] the detector double-fires: on one control frame it returned 3 boxes
    for 2 hands, and 2 boxes for 1 hand on the blacked-out version of that same
    frame. Overlap separates the two cases completely --
      genuine hand pairs   IoU = 0.00 in 10/10 measured pairs (hands do not
                           occupy the same pixels)
      duplicate boxes      IoU = 0.20 and 0.24 (containment 0.45 / 0.51)
    Any value in (0.0, 0.20) gives identical results; 0.10 sits in the middle of
    an entirely empty interval. Without this, `n_hands` could report 3 for a
    single pair of hands and inflate both_hand_rate on one-hand footage."""

    both_hand_rate_min: float = 0.25
    """Fraction of frames with >=2 detected hands to call a clip HANDS_ONLY
    rather than unusable.
    [MEASURED] on the paired controls plus two real clips:
      two-hand footage  0.42 / 0.83 / 1.00
      one-hand footage  0.04
      no-hand footage   0.00
    0.25 sits in a 0.38-wide empty gap. This is the second-best-supported
    number in the module after kpt_score_thresh."""

    min_hand_ratio: float = 0.75
    """Fraction of frames with >=1 hand. [OFFICIAL] min_hand_ratio.
    POLICY DIVERGENCE: the official filter REJECTS below this; we only record
    `low_hand_ratio` as a reason, because our span logic already handles the
    common cause (a compilation where the hands appear in one long stretch and
    the rest is B-roll) by trimming instead of discarding."""

    min_hand_size: float = 0.005
    max_hand_size: float = 0.40
    trunc_edge_px: int = 5
    max_trunc_ratio: float = 0.5
    """[OFFICIAL] all four. Note max_hand_size=0.40 is ~7.6x looser than the
    ~0.03 our own measurement wanted (QUALITY_DIAGNOSIS_DESIGN §2.4: the
    official value passes squeeze_soap, which is framing-incomplete) -- it is a
    100M-scale coarse filter. Reported, not fatal.
    Truncation is likewise WARN-ONLY: serve_cake has 65.7% edge-touching frames
    and is usable footage, so gating on it false-rejects."""

    # ---------- usable span (temporal structure) ----------
    min_usable_sec: float = 5.0
    """A contiguous stretch of passing frames at least this long makes a clip
    usable EVEN IF its clip-level rate is below threshold.
    [OFFICIAL-derived] same 5s floor as min_subseg_duration -- the official
    pipeline also judges the longest usable span rather than a whole-file
    average, and trims to it.
    Rationale: scraped footage is a compilation. A tutorial with 60s of usable
    hands plus 200s of product shots and title cards has a low average and high
    usable content; averaging over the whole file measures the editing, not the
    footage."""

    span_gap_tol: int = 1
    """Consecutive failing samples tolerated inside a contiguous run without
    breaking it. [GUESS] Hands leave frame momentarily in real footage; with 0
    tolerance `hhh.hhh` reads as two short spans instead of one usable one.
    1 is the smallest value that admits a single dropout."""

    min_run_density: float = 0.6
    """Fraction of samples inside a span that must pass for the span to count.
    [GUESS] Needed because gap tolerance alone over-claims: an alternating
    `h.h.h.h` detection pattern would otherwise be reported as one continuous
    usable stretch, when in fact the hands are visible about half the time."""

    # ---------- shot cuts ----------
    scene_threshold: float = 0.40
    """ffmpeg select='gt(scene,X)'. [OFFICIAL] action100m_filter/config.py."""

    min_subseg_sec: float = 5.0
    """Longest cut-free sub-segment shorter than this -> reject.
    [OFFICIAL] action100m_filter/config.py min_subseg_duration."""

    cut_ignore_before_sec: float = 0.1
    """Cuts this early are decode artefacts, not edits. [OFFICIAL] stream.py."""

    # ---------- camera motion ----------
    max_bg_flow: float = 2.0
    """Background optical flow px/frame above which the camera is 'moving'.
    [OFFICIAL] config.py (note: the official main.py argparse default disagrees
    at 3.0 -- config.py is the tuned one).
    NOTE: here this is a ROUTING LABEL, not a rejection. A moving camera is what
    HaWoR wants; a static one is what WiLoR+MoGe wants."""

    flow_grid: tuple = (4, 4)
    flow_percentile: int = 20
    """Farneback -> per-cell median -> 20th percentile across cells.
    [OFFICIAL] _camera_motion_score_flow (its docstring says 25 but the code
    says 20; the code wins). The low percentile is what excludes the moving
    foreground hands."""

    flow_max_pairs: int = 24
    """Frame pairs sampled for the flow estimate. [GUESS]"""

    # ---------- background texture (SLAM feasibility) ----------
    min_corner_density: float = 2.0e-4
    """Shi-Tomasi corners per pixel below which the background is 'poor'
    (plain wall / seamless backdrop -> monocular SLAM scale degenerates).
    [MEASURED-adjacent] the failure it targets is real and documented
    (web_apple: plain background -> SLAM scale collapsed to ~0.01 with many
    NaNs), but the density value itself is a [GUESS] -- that clip is no longer
    on disk to measure."""

    # ---------- blur ----------
    min_hand_lapvar: float = 40.0
    """Laplacian variance in the wrist ROI below which frames are motion-blurred.
    [GUESS] -- standard blur metric, threshold uncalibrated."""

    # ---------- hygiene ----------
    min_duration_sec: float = 10.0
    """[MEASURED] VIDEO_SELECTION_GUIDE B3: a 2.9s clip broke the perception
    stack, 21s worked. ~150 frames / ~10s is the recorded floor."""

    min_side_px: int = 240
    """[GUESS] short-side floor. All current footage is 640x360."""

    # ---------- runtime ----------
    device: str = "auto"
    early_exit: bool = True
    """Skip the appearance/motion stages when the pose gate already rejects.
    Set False for calibration runs, where you want every signal on every clip
    (see README 'compute-all vs early-exit')."""

    def to_dict(self):
        return asdict(self)
