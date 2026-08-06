"""Output schema for the stage-1 gate.

The contract with pipeline step 2 (technique routing) lives here. The verdict is
NOT binary: a clip is accepted for a route, deferred to step 2's view
classifier, or rejected. Every accepted-or-deferred clip carries labels.

ViewClass / CameraMotion 定义在 ``web2robot.routing.schema``：它们是路由
判据，不是质检判据；这里只是引用（并转出，老的
``from ...quality.schema import ViewClass`` 依然可用）。
"""
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional, List, Dict, Any

from ..routing.schema import ViewClass, CameraMotion   # noqa: F401  (re-export)


class Verdict(str, Enum):
    ACCEPT = "accept"          # usable as-is on the routed technique stack
    TRIM = "trim"              # usable after cropping to usable_span
    DEFER = "defer"            # framing is hands-only: step 2 must classify the view
    REJECT = "reject"          # not usable by any known route
    UNKNOWN = "unknown"        # a required signal could not be computed -> human look


@dataclass
class ClipReport:
    # identity
    clip_id: str
    path: str
    source: str = "scraped"
    """Where the clip came from: 'official' | 'scraped' | 'selfcap'.
    Decision 4 (2026-08-05): this is not bookkeeping -- it determines which
    criteria are even evaluable."""

    # verdict
    verdict: str = Verdict.UNKNOWN.value
    reasons: List[str] = field(default_factory=list)
    """ALL failing checks, most-decisive first. Never collapse to one reason:
    the F and V criteria overlap in what they reject, so a single reason
    misattributes the cause."""
    needs_human_review: bool = False

    # ---- routing labels: the interface reserved for pipeline step 2 ----
    view_class: str = ViewClass.UNKNOWN.value
    camera_motion: str = CameraMotion.UNKNOWN.value
    bg_texture: str = "unknown"            # 'rich' | 'poor' | 'unknown'
    suggested_route: Optional[str] = None
    """Step 2's decision, precomputed as a SUGGESTION from the rules in
    VIDEO_SELECTION_GUIDE.md §0.1. Step 2 owns the final call."""
    route_rationale: List[str] = field(default_factory=list)

    # usable span: the longest cut-free sub-segment intersected with the
    # longest contiguous stretch of framing-passing frames
    usable_span: Optional[List[float]] = None      # [t0, t1] seconds
    usable_sec: Optional[float] = None

    # ---- raw signals (null = not computed / not computable) ----
    signals: Dict[str, Any] = field(default_factory=dict)
    per_frame: List[Dict[str, Any]] = field(default_factory=list)

    # bookkeeping
    stages_run: List[str] = field(default_factory=list)
    stages_skipped: List[str] = field(default_factory=list)
    error: Optional[str] = None

    def to_dict(self, with_frames: bool = False) -> Dict[str, Any]:
        d = asdict(self)
        if not with_frames:
            d.pop("per_frame", None)
        return d

    def add_reason(self, code: str):
        if code not in self.reasons:
            self.reasons.append(code)


# Reason vocabulary. The first five are copied from the official prefilter
# (action100m_filter/main.py::_judge) so results stay comparable; the rest are ours.
REASONS = {
    # official vocabulary
    "low_hand_ratio":   "hands detected in too few frames (>=1 hand; official "
                        "hand_ratio) -- RECORDED, not fatal here",
    "hand_truncated":   "hand bbox sits on the frame edge -- RECORDED, not fatal "
                        "(serve_cake is 65.7% edge-touching and usable)",
    "bg_moving":        "background flow above threshold -- a ROUTING label here, "
                        "not a rejection (moving -> HaWoR)",
    "cuts_too_short":   "longest cut-free sub-segment below the floor",
    "hand_too_small":   "mean hand bbox area fraction below min_hand_size",
    "hand_too_large":   "mean hand bbox area fraction above max_hand_size",
    # ours
    "no_person":        "no person AND no hand detected in any sampled frame",
    "no_stable_hands":  "too few frames show TWO detected hands (hand detector, "
                        "not wrist keypoints -- the latter measured inverted)",
    "no_torso":         "torso keypoints never visible: root frame is unrecoverable",
    "no_forearm":       "elbow keypoints never visible (reported only; it does not "
                        "gate anything since the hand gate took over)",
    "hands_only":       "hands framed without a torso -- view class undetermined",
    "hand_gate_unavailable":
                        "the hand detector could not be loaded, so the hands-only "
                        "vs unusable boundary is unmeasurable -> human look. Not a "
                        "rejection: the body model's wrist statistic is falsified "
                        "on this boundary and guessing with it is worse",
    "too_short":        "clip shorter than the perception stack's floor",
    "too_small":        "frame short side below the floor",
    "blurry":           "wrist ROI Laplacian variance below threshold",
    "texture_poor":     "background corner density too low for monocular SLAM",
    "decode_error":     "video could not be opened or decoded",
    "span_conflict":    "the usable framing stretch straddles a shot cut -- "
                        "the two trim estimates disagree, look at it",
}
