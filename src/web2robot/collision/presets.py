"""Per-root-pose-route parameter presets for the arm-vs-torso collision filter.

Why routes need different numbers
---------------------------------
`--root_solver neural` and `--root_solver grid` place the robot base differently:
the grid search maximises IK feasibility (plus a reach-utilisation term), which
pulls the base *toward* the hands, so the arms approach the trunk at different
angles and much more often.  Measured over 13 official clips: 穿躯帧 23.8%
(neural) vs 28.9% (grid), ρ̄ 0.441 vs 0.387.  One set of collision-filter
margins therefore cannot serve both — a box calibrated on `grid` material
mis-fires on `neural` and vice versa.

Why a preset table and not new defaults
---------------------------------------
`ArmTorsoFilter.__init__`'s own defaults stay exactly as they were (and
`tests/test_module_boundaries.py` pins `enter_thresh == 0.04`), because every
`neural` number we have ever published — the 13-clip table, the demo videos, the
README assets — was produced with them.  Changing a default would silently
invalidate all of it.  So `NEURAL` is *empty on purpose*: taking the neural
route means constructing the filter exactly as before, bit for bit.

Where GRID's numbers come from
------------------------------
`scripts/dev/sweep_arm_torso_params.py`, calibrated against MuJoCo real-mesh
contacts on three pre-filter runs (542 frames: fill_jar, sip_coffee,
-2cNMO9Mm3Q_192.4_209.2).  Two findings shaped the parameterisation:

  * the proxy's *shape* was already near-perfect (pooled AUC 0.9997) — what was
    wrong was its *zero-crossing*: the box was too big, so 183/542 frames read
    as penetrating that the real mesh says are clean.  Shrinking the trunk box
    to `[0.50, 0.70, 1.00] x` the true mesh AABB half-extents gives 漏 0 / 误 0
    and AUC 1.0000.  (The z half-extent is unconstrained by the data — every
    value from 1.00x up ties — so it is left at the true mesh value as the least
    arbitrary choice.)
  * with distance 0 now meaning "real mesh contact", `enter_thresh` alone can no
    longer do two jobs.  The old oversized box supplied the push-out clearance
    implicitly; the calibrated one does not, so the clearance becomes explicit
    as `margin`.  See `ArmTorsoFilter._sdf` for the mapping.

Presets deliberately hold only the *calibrated* quantities (box + the two
thresholds).  Optimiser weights, iteration counts and smoothing are route-
independent and stay at the class defaults.

How well GRID held up out of sample (2026-08-21)
------------------------------------------------
Re-ran all 13 clips.  What matters — residual real-mesh penetration after
filtering — improved and generalised: 28.9% → 13.3% of frames over 13 clips
(37.4% → 17.3% on the 10 clips *not* used for calibration), 12/13 → 9/13 clips
with any residual, IK feasibility unchanged.

The `漏 0 / 误 0` above, however, is **in-sample only**.  Out of sample the
proxy-vs-mesh disagreement did not shrink (180 → 198 frames) and flipped
direction: 423 false alarms became 0, but 17 misses became 222.  Cause is the
proxy's *shape*, not these numbers: the real trunk is round, and forcing an
axis-aligned box to zero false alarms at the corners costs coverage on the
faces (x half-extent down to 0.50x the mesh), so real penetration shallower
than ~1.7 cm is invisible to it.  Measured: on `-0RheyDV3a0_48.6_55.3`'s 90
residual frames the proxy reads +0.08..+0.48 cm ("not touching yet") while the
mesh reads 1.26 cm deep.  Those frames are visually clean (forearms resting on
the chest plate), so this is a detection blind spot rather than a picture
problem — but it means the box cannot be pushed further in this direction.  The
fix is to decouple detection from the push-out target (detect with a
mesh-sized box, push out to the calibrated one), not to re-tune the triple.
See `docs/VERIFICATION.md` and BACKLOG A1.

Where the numbers live
----------------------
2026-08-21: the tables below are read from `configs/robots/m7.yaml`
(`collision.mesh_aabb_half`, `collision.arm_torso.routes`) instead of being
written here.  Same numbers, one home — and the yaml carries the `verified` flag
that this docstring's prose implies: `grid` is `verified: true` (the sweep above),
`NEURAL` is not (an empty override set is not a calibration; its byte-level
guarantee is `scripts/dev/check_neural_bytes.sh`, not a measurement of numbers).
"""

from __future__ import annotations

from web2robot.robots.params import robot_params as _robot_params, values as _values

_C = _robot_params("m7")["collision"]

# The trunk mesh AABB half-extents on waist_pitch_link, for reference: the
# calibrated box below is expressed as fractions of these.
MESH_HALF = tuple(_C["mesh_aabb_half"]["value"])

#: `neural` = today's behaviour, unchanged. Empty by design — see module docstring.
NEURAL: dict = _values(_C["arm_torso"]["routes"]["neural"])

#: `grid` = calibrated 2026-08-20 (see module docstring for the material).
GRID: dict = _values(_C["arm_torso"]["routes"]["grid"])
# Tuple, not the yaml's list: callers pass this straight into
# `M7CapsuleModel(torso_half=...)` and a tuple cannot be mutated in place by one
# caller behind another's back.
GRID["torso_half"] = tuple(GRID["torso_half"])

_BY_ROUTE = {"neural": NEURAL, "grid": GRID}


def arm_torso_preset(route: str) -> dict:
    """Return the ArmTorsoFilter kwargs calibrated for this root-pose route.

    Unknown routes raise rather than silently falling back: a new root solver
    has never been calibrated, and quietly handing it `grid`'s numbers would be
    a measurement nobody made.
    """
    try:
        return dict(_BY_ROUTE[route])
    except KeyError:
        raise ValueError(
            f"no arm-torso collision preset for root route {route!r}; "
            f"known routes: {sorted(_BY_ROUTE)}. Calibrate with "
            f"scripts/dev/sweep_arm_torso_params.py before shipping a new one."
        ) from None
