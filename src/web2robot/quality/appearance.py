"""Appearance signals: background texture and motion blur.

Background texture was deferred in decision 2 (2026-08-05) for a reason that no
longer holds -- it was deferred because the reconstructed clips on disk ship
only depth.mp4 with no RGB. This gate runs on raw scraped RGB, so the signal is
computable and is included.

It matters for ROUTING, not for quality: monocular SLAM needs background
features, and a plain backdrop is exactly what collapsed HaWoR's scale to ~0.01
with widespread NaNs on the web_apple clip. Poor texture therefore pushes a clip
toward WiLoR+MoGe even when the camera is moving.
"""
from typing import List, Tuple, Optional

import numpy as np
import cv2


def _person_mask(shape, boxes, pad: float = 0.08) -> np.ndarray:
    """255 outside the (padded) person boxes, 0 inside."""
    H, W = shape[:2]
    m = np.full((H, W), 255, np.uint8)
    for b in boxes:
        if b is None:
            continue
        x0, y0, x1, y1 = b
        px, py = pad * (x1 - x0), pad * (y1 - y0)
        m[max(0, int(y0 - py)):min(H, int(y1 + py)),
          max(0, int(x0 - px)):min(W, int(x1 + px))] = 0
    return m


def background_texture(frames: List[Tuple[int, np.ndarray]],
                       boxes: List[Optional[np.ndarray]]) -> dict:
    """Shi-Tomasi corner density in the non-person region, per pixel.

    Corners rather than gradient energy: SLAM tracks corners, and a smooth
    gradient (a lit seamless backdrop) has energy but nothing trackable.
    """
    dens, counts = [], []
    for (_, img), box in zip(frames, boxes):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mask = _person_mask(img.shape, [box])
        area = int((mask > 0).sum())
        if area < 0.05 * mask.size:      # person fills the frame: undecidable
            continue
        pts = cv2.goodFeaturesToTrack(gray, maxCorners=2000, qualityLevel=0.01,
                                      minDistance=5, mask=mask)
        n = 0 if pts is None else len(pts)
        counts.append(n)
        dens.append(n / area)
    if not dens:
        return dict(corner_density=None, corner_count_med=None, n_frames=0)
    return dict(corner_density=round(float(np.median(dens)), 8),
                corner_count_med=int(np.median(counts)),
                n_frames=len(dens))


def hand_blur(frames: List[Tuple[int, np.ndarray]], pose_frames, roi_frac=0.18
              ) -> dict:
    """Laplacian variance in a box around the visible wrists.

    Measured on the hands, not the whole frame: a clip can have a sharp static
    background and hands smeared by motion, and it is the hands we reconstruct.
    """
    from .pose_gate import WRIST
    vals = []
    for (_, img), pf in zip(frames, pose_frames):
        if not pf.found or pf.kscore is None:
            continue
        H, W = img.shape[:2]
        r = int(roi_frac * min(H, W) / 2)
        for i in WRIST:
            if pf.kscore[i] < 3.0:
                continue
            x, y = int(pf.kps[i, 0]), int(pf.kps[i, 1])
            roi = img[max(0, y - r):min(H, y + r), max(0, x - r):min(W, x + r)]
            if roi.size < 100:
                continue
            g = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            vals.append(float(cv2.Laplacian(g, cv2.CV_64F).var()))
    if not vals:
        return dict(hand_lapvar_med=None, hand_lapvar_p10=None, n_roi=0)
    return dict(hand_lapvar_med=round(float(np.median(vals)), 2),
                hand_lapvar_p10=round(float(np.percentile(vals, 10)), 2),
                n_roi=len(vals))
