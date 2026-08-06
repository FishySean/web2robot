"""Visual evidence for the human spot-check.

Standing rule from earlier work: metrics do not establish that data is
trustworthy -- the rendered picture does. So the gate never reports a verdict
without being able to show the frame it was decided on.

Draws the pose skeleton (green = visible above the score threshold, red = not),
the verdict, and which criteria failed, then tiles one row per clip.
"""
import os
from typing import List, Optional

import numpy as np
import cv2

from .pose_gate import LIMBS, WRIST, ELBOW, TORSO, HEAD

GREEN, RED, WHITE = (0, 230, 0), (0, 0, 255), (255, 255, 255)
CYAN = (255, 220, 0)     # hand boxes -- a different instrument, a different colour


def annotate(img: np.ndarray, pf, cfg, title: str, verdict: str,
             reasons: List[str], hf=None) -> np.ndarray:
    im = img.copy()
    if pf.found and pf.kscore is not None:
        vis = pf.kscore >= cfg.kpt_score_thresh
        for a, b in LIMBS:
            if vis[a] and vis[b]:
                cv2.line(im, tuple(pf.kps[a, :2].astype(int)),
                         tuple(pf.kps[b, :2].astype(int)), GREEN, 2)
        for i in range(17):
            cv2.circle(im, tuple(pf.kps[i, :2].astype(int)), 4,
                       GREEN if vis[i] else RED, -1)
        detail = (f"wrist {pf.n_wrist}/2  elbow {pf.n_elbow}/2  "
                  f"torso {pf.n_torso}/4  head {pf.n_head}/5")
    else:
        detail = "no person detected"

    # Hand boxes are drawn even when no person was found -- that combination
    # (hands but no body) IS the hands-only class, and the picture has to show it.
    if hf is not None:
        detail += f"  |  hands {hf.n_hands}"
        if hf.boxes is not None:
            for x1, y1, x2, y2 in hf.boxes.astype(int):
                cv2.rectangle(im, (x1, y1), (x2, y2), CYAN, 2)

    bar = 48
    im = cv2.copyMakeBorder(im, bar, 0, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    col = GREEN if verdict in ("accept", "trim") else (
        (0, 200, 255) if verdict == "defer" else RED)
    cv2.putText(im, f"{title}   f{pf.frame_idx}", (6, 17),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, WHITE, 1)
    cv2.putText(im, f"{verdict.upper()}  {detail}"
                + (f"  [{','.join(reasons[:2])}]" if reasons else ""),
                (6, 39), cv2.FONT_HERSHEY_SIMPLEX, 0.48, col, 1)
    return im


def pick_two(pfs) -> List[int]:
    """Indices of the most- and least-complete framing in the sample."""
    def key(i):
        p = pfs[i]
        return (p.framing_ok, p.n_wrist + p.n_elbow + p.n_torso)
    idx = sorted(range(len(pfs)), key=key)
    return [idx[-1], idx[0]] if len(idx) > 1 else [0]


def pick_evidence(pfs, rep, hfs=None) -> List[int]:
    """Frames the verdict actually rests on.

    A clip can now pass on a contiguous usable SPAN rather than a whole-file
    rate, so showing the single best and single worst frame is no longer honest
    evidence -- it says nothing about whether the claimed span holds up. Pick the
    passing frame nearest the middle of the claimed span, then the least
    complete frame in the whole sample as the counterweight.

    "Passing" is judged by whichever instrument decided the class: body framing
    for THIRD_PERSON_BODY, two detected HANDS otherwise. Using wrist keypoints
    here would illustrate the verdict with a statistic that did not produce it.
    """
    f = rep.signals.get("framing") or {}
    h = rep.signals.get("hands") or {}
    fps = (rep.signals.get("hygiene") or {}).get("fps") or 0.0
    body = rep.view_class == "third_person_body"
    span = f.get("body_span_est") if body else h.get("hands_span_est")
    n_by_idx = {x.frame_idx: x.n_hands for x in (hfs or [])}
    out = []
    if span and fps > 0:
        want = (span[0] + span[1]) / 2.0
        inside = [i for i, p in enumerate(pfs)
                  if span[0] <= p.frame_idx / fps <= span[1]
                  and (p.framing_ok if body
                       else n_by_idx.get(p.frame_idx, 0) >= 2)]
        if inside:
            out.append(min(inside, key=lambda i: abs(pfs[i].frame_idx / fps - want)))
    if not out and not body and n_by_idx:
        # rejected for too few two-hand frames: show a frame that does have both,
        # if one exists at all, so the reject can be argued with rather than
        # merely believed
        best = max(range(len(pfs)), key=lambda i: n_by_idx.get(pfs[i].frame_idx, 0))
        if n_by_idx.get(pfs[best].frame_idx, 0) > 0:
            out.append(best)
    for i in pick_two(pfs):
        if i not in out:
            out.append(i)
    return out[:3]


class Visualizer:
    """Collects annotated evidence during a run; call .save() at the end."""

    def __init__(self, outdir: str, cfg, width: int = 480):
        self.outdir = outdir
        self.cfg = cfg
        self.width = width
        self.rows = []
        os.makedirs(outdir, exist_ok=True)

    def on_pose(self, rep, frames, pfs, hfs=None):
        if not pfs:
            return
        # verdict is not final yet at this point; the framing view class is,
        # and it is what these frames are evidence for.
        tag = {"third_person_body": "accept?", "hands_only": "defer",
               "no_stable_hands": "reject"}.get(rep.view_class, rep.view_class)
        by_idx = {h.frame_idx: h for h in (hfs or [])}
        imgs = []
        for i in pick_evidence(pfs, rep, hfs):
            im = annotate(frames[i][1], pfs[i], self.cfg, rep.clip_id,
                          tag, rep.reasons, by_idx.get(pfs[i].frame_idx))
            h = int(im.shape[0] * self.width / im.shape[1])
            imgs.append(cv2.resize(im, (self.width, h)))
            cv2.imwrite(os.path.join(
                self.outdir, f"{rep.clip_id}__f{pfs[i].frame_idx}.png"), im)
        if imgs:
            h = min(x.shape[0] for x in imgs)
            self.rows.append(np.hstack([x[:h] for x in imgs]))

    def save(self, name: str = "contact_sheet.png") -> Optional[str]:
        if not self.rows:
            return None
        w = min(r.shape[1] for r in self.rows)
        sheet = np.vstack([r[:, :w] for r in self.rows])
        p = os.path.join(self.outdir, name)
        cv2.imwrite(p, sheet)
        return p
