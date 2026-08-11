"""Wrist-trajectory cleanup — bad/missing frame fallback for retargeting input.

Why this exists
---------------
The perception front end (WiLoR/HaWoR + depth) feeds per-frame wrist poses
straight into IK.  Two failure modes reach the robot unfiltered:

  * **missing value** — the hand is not detected.  ``SamplesSequence._fill_gaps``
    already applies an *unbounded* zero-order hold, so a 44-frame tail (2.9 s in
    serve_cake) freezes the arm on whatever the last detected frame happened to
    be, with no record that those frames were invented.
  * **bad value** — the hand *is* detected but the pose is garbage.  Measured on
    fill_jar: the left wrist depth balloons 1.40 m → 2.60 m → 1.40 m over frames
    160–178 (~1.2 s), which is the "11 s left-hand breakdown".  Nothing in the
    pipeline looks at this; every robot's IK gets the same poisoned target.

Both are the same problem — a frame whose target must not be trusted — so this
module converts bad values into missing values and then applies one gap policy.

Bad-frame detection (three stages, all needed)
----------------------------------------------
1. **speed gate** — inter-frame wrist speed above a physical cap.
2. **run closing** — flagged frames separated by a few good ones merge into one
   run.  Essential: the fill_jar excursion *plateaus* at the top (steps of only
   0.7–1.1 cm/frame for 3 frames), so the speed gate alone would mark the ramps
   bad and the plateau good — i.e. hold the arm at the worst pose of all.
3. **excursion test** — a genuine fast reach goes A→B and stays at B; a
   perception blow-up goes A→far→A.  Requiring the run's interior to bulge off
   the chord between its valid endpoints keeps real motion and kills balloons.
   Measured: fill_jar left run f162–177 bulges 114.5 cm off a 4.2 cm chord,
   while every speed-gate hit on ours_webblocks (4 L / 14 R) and fill_jar right
   (3) is correctly released.

Note the bad frames are bad in *global position only* — the 3D hand is a metric
MANO hand translated by the estimated depth, so palm size stays 9.47 cm even at
z = 2.60 m.  Finger articulation on those frames is fine, so callers should keep
using them for finger retargeting; only the wrist target is suspect.

Gap policy
----------
``FILL_REST`` is the **last resort**: it is the only branch that throws the
measured motion away, so the policy is tuned to reach it as rarely as possible.
Where the gap sits matters as much as how long it is:

=========================  ==================================================
interior, <= max_interp     interpolate between both valid ends (lerp + SLERP).
                            No jump at either boundary, unlike a hold.
interior, longer            ``FILL_REST`` — a multi-second straight line
                            through space is fabrication, not interpolation.
leading (clip starts blind)  <= max_hold: hold the first valid pose.  Longer:
                            ``FILL_REST`` — there is nothing before the gap to
                            carry forward, and the arm has to start *somewhere*,
                            so an honest rest pose beats an invented reach.
trailing (clip ends blind)  hold the last valid pose, **however long the tail
                            is** (``hold_tail=True``).  Nothing follows, so a
                            hold introduces no discontinuity and keeps the arm
                            where the last real observation put it — ramping to
                            rest here is a large invented motion at the exact
                            point where we know least.  The frames are still
                            marked ``FILL_HOLD`` (never ``OK``) and long tails
                            are reported in ``report["tail_hold"]``, so "this is
                            held, not measured" still reaches the caller.
=========================  ==================================================

The leading/trailing asymmetry is a deliberate design decision (2026-08-11), not
an oversight: a held *tail* is the cheapest honest answer, a held *head* does not
exist.  ``max_interp_sec`` was raised 1.5 s -> 2.5 s at the same time, for the
same reason — keep interpolating a bit longer before giving up on the motion.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

# ── per-frame status codes (written to trajectory.npz) ────────────────────────

OK          = 0   # real detection, passed plausibility
FILL_INTERP = 1   # short interior gap, interpolated between both valid ends
FILL_HOLD   = 2   # short boundary gap, held from nearest valid frame
FILL_REST   = 3   # long gap — caller should ramp to rest pose; NOT usable data

STATUS_NAMES = {OK: "ok", FILL_INTERP: "interp", FILL_HOLD: "hold", FILL_REST: "rest"}

# ── per-frame cause codes ─────────────────────────────────────────────────────

C_OK      = 0
C_MISSING = 1     # hand not detected
C_BAD     = 2     # detected but rejected as implausible

CAUSE_NAMES = {C_OK: "ok", C_MISSING: "missing", C_BAD: "bad"}


# ── quaternion hygiene ────────────────────────────────────────────────────────

def canonicalize_quats(traj: np.ndarray) -> int:
    """Make the quaternion sign continuous across valid frames, in place.

    q and -q are the same rotation, so this changes no target, but it is a
    prerequisite for interpolating or smoothing them.  Measured sign flips on
    raw input: fill_jar 12 (left) / 13 (right), serve_cake 8 (right).
    """
    valid = ~np.isnan(traj[:, 0])
    vi = np.where(valid)[0]
    flips = 0
    for a, b in zip(vi[:-1], vi[1:]):
        if float(np.dot(traj[b, 3:], traj[a, 3:])) < 0.0:
            traj[b, 3:] *= -1.0
            flips += 1
    return flips


# ── bad-frame detection ───────────────────────────────────────────────────────

def detect_bad_frames(
    pos:            np.ndarray,      # (T, 3) camera-frame wrist positions
    valid:          np.ndarray,      # (T,) bool
    fps:            float,
    speed_cap:      float = 0.08,    # m per frame at 15 fps (~1.2 m/s wrist speed)
    close_gap:      int   = 6,       # merge flagged runs separated by <= this
    excursion_min:  float = 0.12,    # m the interior must bulge off the chord
    dilate:         int   = 2,       # frames to expand each confirmed run by
):
    """Return (bad_mask, runs) where runs is [(start, end, bulge_m, chord_m)]."""
    T = len(valid)
    bad = np.zeros(T, bool)
    vi  = np.where(valid)[0]
    if len(vi) < 3:
        return bad, []

    cap = speed_cap * (fps / 15.0)

    # stage 1 — speed gate between consecutive valid frames
    for a, b in zip(vi[:-1], vi[1:]):
        if np.linalg.norm(pos[b] - pos[a]) / (b - a) > cap:
            bad[a] = bad[b] = True

    # stage 2 — close short good islands between flags (swallows plateaus)
    fb = np.where(bad[vi])[0]
    for a, b in zip(fb[:-1], fb[1:]):
        if b - a <= close_gap:
            bad[vi[a:b + 1]] = True

    # stage 3 — keep only runs that leave and come back
    runs, s = [], None
    for i, t in enumerate(vi):
        if bad[t] and s is None:
            s = i
        elif not bad[t] and s is not None:
            runs.append((s, i - 1)); s = None
    if s is not None:
        runs.append((s, len(vi) - 1))

    kept = []
    for i0, i1 in runs:
        a = vi[i0 - 1] if i0 > 0 else vi[i0]
        b = vi[i1 + 1] if i1 + 1 < len(vi) else vi[i1]
        pa, chord = pos[a], pos[b] - pos[a]
        L = float(np.linalg.norm(chord))
        bulge = 0.0
        for t in vi[i0:i1 + 1]:
            d = pos[t] - pa
            if L > 1e-6:
                u    = chord / L
                proj = float(np.dot(d, u))
                perp = float(np.linalg.norm(d - proj * u))
                # distance to the *segment*, not the infinite line
                off  = max(perp, proj - L, -proj)
            else:
                off = float(np.linalg.norm(d))
            bulge = max(bulge, off)
        if bulge >= excursion_min:
            lo = max(0, vi[i0] - dilate)
            hi = min(T - 1, vi[i1] + dilate)
            bad[lo:hi + 1] = True
            kept.append((lo, hi, bulge, L))
        else:
            bad[vi[i0:i1 + 1]] = False

    bad &= valid          # only ever reject frames that were detected
    return bad, kept


# ── gap classification + fill ─────────────────────────────────────────────────

def _runs_of_true(mask):
    out, s = [], None
    for i, v in enumerate(mask):
        if v and s is None:
            s = i
        elif not v and s is not None:
            out.append((s, i - 1)); s = None
    if s is not None:
        out.append((s, len(mask) - 1))
    return out


def _interp_pose(traj, a, b, ts):
    """Fill frames ``ts`` by lerp (position) + SLERP (rotation) between a and b."""
    rot = Rotation.from_quat(np.stack([traj[a, [4, 5, 6, 3]], traj[b, [4, 5, 6, 3]]]))
    sl  = Slerp([a, b], rot)
    for t in ts:
        w = (t - a) / (b - a)
        traj[t, :3] = (1.0 - w) * traj[a, :3] + w * traj[b, :3]
        xyzw = sl(t).as_quat()
        traj[t, 3:] = np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]], np.float32)


def clean_wrist_trajectory(
    traj_raw:    np.ndarray,     # (T, 7) with NaN rows where undetected
    fps:         float,
    max_interp_sec: float = 2.5,
    max_hold_sec:   float = 0.5,   # leading gaps only (see hold_tail)
    hold_tail:      bool  = True,  # trailing gap: hold, however long
    detect_bad:     bool  = True,
    side:           str   = "",
    verbose:        bool  = True,
):
    """Clean one hand's wrist trajectory.

    Returns ``(traj, status, cause, report)``.  ``traj`` has no NaN rows unless
    the hand was never detected at all, in which case it is returned unchanged
    with every status set to ``FILL_REST``.
    """
    traj = np.array(traj_raw, dtype=np.float32, copy=True)
    T = traj.shape[0]
    status = np.zeros(T, np.int8)
    cause  = np.zeros(T, np.int8)
    report = {"side": side, "n_frames": T}

    valid = ~np.isnan(traj[:, 0])
    report["n_detected"] = int(valid.sum())
    if not valid.any():
        status[:] = FILL_REST
        cause[:]  = C_MISSING
        report.update(never_detected=True, bad_runs=[], quat_flips=0,
                      n_interp=0, n_hold=0, n_rest=T, tail_hold=0)
        if verbose:
            print(f"  [cleanup:{side}] hand never detected → whole clip rest pose")
        return traj, status, cause, report

    report["quat_flips"] = canonicalize_quats(traj)

    # ── bad values → missing values ───────────────────────────────────────────
    bad_runs = []
    if detect_bad:
        bad, bad_runs = detect_bad_frames(traj[:, :3], valid, fps)
        if bad.any():
            traj[bad] = np.nan
            cause[bad] = C_BAD
            valid = ~np.isnan(traj[:, 0])
    report["bad_runs"] = [(int(a), int(b), float(bl), float(ch))
                          for a, b, bl, ch in bad_runs]
    report["n_bad"] = int((cause == C_BAD).sum())
    cause[(cause != C_BAD) & ~valid] = C_MISSING

    # ── classify and fill each gap ────────────────────────────────────────────
    max_interp = max(1, int(round(max_interp_sec * fps)))
    max_hold   = max(1, int(round(max_hold_sec   * fps)))
    report["tail_hold"] = 0        # frames of tail held beyond max_hold

    for a, b in _runs_of_true(~valid):
        L = b - a + 1
        prev = a - 1 if a > 0 else None
        nxt  = b + 1 if b < T - 1 else None

        if prev is not None and nxt is not None:            # ── interior
            _interp_pose(traj, prev, nxt, range(a, b + 1))
            status[a:b + 1] = FILL_INTERP if L <= max_interp else FILL_REST
        elif prev is None:                                  # ── leading
            traj[a:b + 1] = traj[nxt]
            status[a:b + 1] = FILL_HOLD if L <= max_hold else FILL_REST
        else:                                               # ── trailing
            traj[a:b + 1] = traj[prev]
            if hold_tail or L <= max_hold:
                status[a:b + 1] = FILL_HOLD
                if L > max_hold:
                    # held beyond what the boundary budget would allow: still
                    # honest (not OK), but the caller should know how much of
                    # the tail is invented so it can trim or flag the clip.
                    report["tail_hold"] = int(L)
            else:
                status[a:b + 1] = FILL_REST

    report["n_interp"] = int((status == FILL_INTERP).sum())
    report["n_hold"]   = int((status == FILL_HOLD).sum())
    report["n_rest"]   = int((status == FILL_REST).sum())

    if verbose:
        det = 100.0 * report["n_detected"] / T
        print(f"  [cleanup:{side}] detected {report['n_detected']}/{T} ({det:.1f}%)  "
              f"quat_flips={report['quat_flips']}  bad={report['n_bad']}  "
              f"→ interp={report['n_interp']} hold={report['n_hold']} "
              f"rest={report['n_rest']}")
        if report["tail_hold"]:
            print(f"      ⚠ 结尾 {report['tail_hold']} 帧"
                  f"（{report['tail_hold'] / fps:.1f}s）是沿袭最后一次检测保持出来的，"
                  f"不是测到的动作 —— 要么裁掉，要么标注")
        for a, b, bl, ch in bad_runs:
            print(f"      bad run f{a}..{b} ({b-a+1}f): bulge={100*bl:.1f}cm "
                  f"off a {100*ch:.1f}cm chord")
    return traj, status, cause, report


# ── joint-space rest-pose fallback ────────────────────────────────────────────

def blend_to_rest(
    q:        np.ndarray,     # (T, n_dof) arm joints
    status:   np.ndarray,     # (T,) status codes from clean_wrist_trajectory
    q_rest:   np.ndarray,     # (n_dof,) natural rest configuration
    fps:      float,
    ramp_sec: float = 0.5,
):
    """Ease the arm to ``q_rest`` across ``FILL_REST`` runs, in place-safe copy.

    The weight toward rest rises with distance from the nearest trustworthy
    frame on either side (cosine-eased), so both the departure and the return
    are smooth by construction — no jump at re-acquisition.
    """
    q = np.array(q, dtype=np.float64, copy=True)
    T = q.shape[0]
    ramp = max(1, int(round(ramp_sec * fps)))
    w_all = np.zeros(T)

    for a, b in _runs_of_true(status == FILL_REST):
        has_prev = a > 0
        has_next = b < T - 1
        for t in range(a, b + 1):
            ds = [(t - a + 1) / ramp] if has_prev else []
            if has_next:
                ds.append((b - t + 1) / ramp)
            w = min([1.0] + ds) if ds else 1.0
            w_all[t] = 0.5 * (1.0 - np.cos(np.pi * w))     # cosine ease
        q[a:b + 1] = ((1.0 - w_all[a:b + 1, None]) * q[a:b + 1]
                      + w_all[a:b + 1, None] * q_rest[None, :])
    return q.astype(np.float32), w_all


def relax_fingers(Q, w_all, q_open=None):
    """Relax finger joints toward a neutral open hand using the same weights."""
    if Q is None:
        return None
    Q = np.array(Q, dtype=np.float32, copy=True)
    tgt = np.zeros(Q.shape[1], np.float32) if q_open is None else q_open
    Q = (1.0 - w_all[:, None]) * Q + w_all[:, None] * tgt[None, :]
    return Q.astype(np.float32)
