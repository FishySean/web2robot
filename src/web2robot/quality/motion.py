"""Shot cuts and camera motion.

Both functions are ports of EgoInfinity's own prefilter
(`action100m_filter/stream.py::detect_shot_cuts`,
 `action100m_filter/detect.py::_camera_motion_score_flow`,
 `action100m_filter/main.py::_find_longest_subseg`)
so that our numbers stay comparable with theirs. Behaviour is preserved,
including the 20th-percentile choice (their docstring says 25th; the code says
20 and the code is what was tuned).

One deliberate difference in POLICY, not in computation: for us a moving camera
is a ROUTING LABEL, not a rejection. The official filter rejects `bg_moving`
because it wants static third-person clips; our pipeline sends a moving camera
to HaWoR and a static one to WiLoR+MoGe, so both are keepers.
"""
from typing import List, Tuple, Optional
import re
import subprocess

import numpy as np
import cv2

from ..common.video_io import FFMPEG


# --------------------------------------------------------------------------- #
# shot cuts
# --------------------------------------------------------------------------- #
def detect_shot_cuts(path: str, threshold: float = 0.4, timeout: int = 60,
                     ignore_before: float = 0.1) -> Optional[List[float]]:
    """Cut timestamps in seconds from clip start, or None if ffmpeg failed.

    None and [] are different: [] means 'no cuts found', None means 'we do not
    know'. Per decision 3 (2026-08-05) an unknown signal must not turn into a
    rejection.
    """
    cmd = [FFMPEG, "-nostdin", "-i", path, "-an",
           "-vf", f"select='gt(scene,{threshold})',showinfo",
           "-vsync", "vfr", "-f", "null", "-"]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except (subprocess.TimeoutExpired, OSError):
        return None
    times = re.findall(r"pts_time:([\d.]+)", r.stderr)
    if not times and "Invalid data" in r.stderr:
        return None
    return [float(t) for t in times if float(t) > ignore_before]


def longest_subseg(start: float, end: float, cuts: List[float]
                   ) -> Tuple[float, float]:
    """Longest cut-free span. Port of the official _find_longest_subseg.

    The official pipeline TRIMS to this span and only rejects when the span
    itself is too short -- i.e. a cut is not fatal. That is the precedent for
    our TRIM verdict.
    """
    bounds = [start] + sorted(t for t in cuts if start < t < end) + [end]
    i = max(range(len(bounds) - 1), key=lambda k: bounds[k + 1] - bounds[k])
    return bounds[i], bounds[i + 1]


# --------------------------------------------------------------------------- #
# camera motion
# --------------------------------------------------------------------------- #
def bg_flow_score(gray1: np.ndarray, gray2: np.ndarray,
                  grid: Tuple[int, int] = (4, 4), pct: int = 20) -> float:
    """Background motion in px/frame between two CONSECUTIVE frames.

    Dense Farneback flow -> median magnitude per grid cell -> low percentile
    across cells. The low percentile is the whole trick: moving hands dominate
    a few cells only, so a low percentile reads the background.
    """
    flow = cv2.calcOpticalFlowFarneback(
        gray1, gray2, None, pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
    mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
    H, W = mag.shape
    gh, gw = grid
    ch, cw = H // gh, W // gw
    meds = [float(np.median(mag[r * ch:(r + 1) * ch, c * cw:(c + 1) * cw]))
            for r in range(gh) for c in range(gw)]
    return float(np.percentile(meds, pct))


def camera_motion(pairs: List[Tuple[np.ndarray, np.ndarray]], cfg) -> dict:
    """Median background flow over sampled consecutive pairs."""
    if not pairs:
        return dict(bg_flow=None, bg_flow_p90=None, n_pairs=0)
    scores = [bg_flow_score(cv2.cvtColor(a, cv2.COLOR_BGR2GRAY),
                            cv2.cvtColor(b, cv2.COLOR_BGR2GRAY),
                            cfg.flow_grid, cfg.flow_percentile)
              for a, b in pairs]
    return dict(bg_flow=round(float(np.median(scores)), 4),
                bg_flow_p90=round(float(np.percentile(scores, 90)), 4),
                n_pairs=len(pairs))
