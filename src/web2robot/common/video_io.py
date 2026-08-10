"""Video probing and frame sampling. Decode-only; no analysis here."""
from dataclasses import dataclass
from typing import List, Tuple, Optional
import subprocess
import shutil

import numpy as np
import cv2

FFMPEG = shutil.which("ffmpeg") or "ffmpeg"
FFPROBE = shutil.which("ffprobe") or "ffprobe"


@dataclass
class VideoInfo:
    path: str
    ok: bool
    width: int = 0
    height: int = 0
    fps: float = 0.0
    n_frames: int = 0
    duration: float = 0.0
    error: Optional[str] = None


def probe(path: str) -> VideoInfo:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return VideoInfo(path, False, error="decode_error")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    if w == 0 or h == 0 or n <= 0:
        return VideoInfo(path, False, w, h, fps, n, error="decode_error")
    dur = n / fps if fps > 0 else 0.0
    return VideoInfo(path, True, w, h, fps, n, dur)


def plan_n_frames(duration: float, cfg) -> int:
    """How many frames to sample for a clip of this length.

    A fixed count is wrong on long clips: 24 samples over a 14-minute
    compilation is one sample per 35s, and the resulting pass-rate estimate
    moved across a decision threshold purely on sampling noise (see
    QCConfig.sample_every_sec). Scale with duration, floor at n_frames, cap at
    n_frames_max so the GPU cost stays bounded.
    """
    if duration <= 0:
        return cfg.n_frames
    want = int(round(duration * (cfg.sample_hi - cfg.sample_lo) / cfg.sample_every_sec))
    return int(max(cfg.n_frames, min(cfg.n_frames_max, want)))


def sample_frames(path: str, n: int, lo: float = 0.08, hi: float = 0.92
                  ) -> List[Tuple[int, np.ndarray]]:
    """n frames evenly spaced in [lo, hi] of the clip, as (index, BGR array).

    Random seeking, not sequential decode: on a long clip this is far cheaper
    than walking every frame, at the cost of keyframe-snapping accuracy that
    does not matter for aggregate statistics.
    """
    info = probe(path)
    if not info.ok:
        return []
    cap = cv2.VideoCapture(path)
    out = []
    for i in np.linspace(info.n_frames * lo, info.n_frames * hi, n).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, f = cap.read()
        if ok:
            out.append((int(i), f))
    cap.release()
    return out


def sample_pairs(path: str, n_pairs: int, lo: float = 0.08, hi: float = 0.92
                 ) -> List[Tuple[np.ndarray, np.ndarray]]:
    """n_pairs of CONSECUTIVE frames, spread across the clip.

    Consecutive is essential: optical flow between two frames 30s apart is
    meaningless. The official prefilter makes the same distinction
    (`motion_pairs` vs its uniform-sample fallback).
    """
    info = probe(path)
    if not info.ok:
        return []
    cap = cv2.VideoCapture(path)
    pairs = []
    for i in np.linspace(info.n_frames * lo, info.n_frames * hi, n_pairs).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok1, f1 = cap.read()
        ok2, f2 = cap.read()
        if ok1 and ok2:
            pairs.append((f1, f2))
    cap.release()
    return pairs


IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def read_rgb_frames(path, fps: Optional[float] = None,
                    max_frames: Optional[int] = None) -> Tuple[List[np.ndarray], float]:
    """Every frame in order, as RGB. Returns (frames, fps). Video file or image dir.

    Three deliberate differences from `sample_frames`, which the perception
    frontends need and QC does not:

    - **Sequential decode, not random seek.** Hand tracking needs consecutive
      frames; keyframe-snapping would silently duplicate and drop frames.
    - **RGB, not BGR.** Both WiLoR and MoGe want RGB. cv2 hands back BGR, and
      feeding BGR to a hand detector degrades it quietly rather than failing.
    - **Image directories too**, sorted by name — HO-3D and most extracted
      datasets ship as `rgb/0000.jpg`.

    `fps` resamples by nearest-frame stride (a clip at 30 fps asked for 15
    keeps every other frame); the returned fps is what was actually produced,
    not what was asked for, because the stride is an integer.
    """
    from pathlib import Path
    p = Path(path)

    if p.is_dir():
        files = sorted(f for f in p.iterdir() if f.suffix.lower() in IMAGE_EXTS)
        src_fps = fps or 30.0          # 图片目录没有帧率信息，只能由调用方给
        step = 1
        frames = []
        for f in files[::step]:
            im = cv2.imread(str(f))
            if im is not None:
                frames.append(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
            if max_frames and len(frames) >= max_frames:
                break
        return frames, float(src_fps)

    info = probe(str(p))
    if not info.ok:
        return [], 0.0
    step = 1
    if fps and info.fps > 0:
        step = max(1, int(round(info.fps / fps)))
    cap = cv2.VideoCapture(str(p))
    frames, i = [], 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if i % step == 0:
            frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
            if max_frames and len(frames) >= max_frames:
                break
        i += 1
    cap.release()
    return frames, (info.fps / step if info.fps > 0 else (fps or 0.0))
