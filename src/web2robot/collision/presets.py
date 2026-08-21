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
"""

from __future__ import annotations

# The trunk mesh AABB half-extents on waist_pitch_link, for reference: the
# calibrated box below is expressed as fractions of these.
MESH_HALF = (0.139, 0.170, 0.239)

#: `neural` = today's behaviour, unchanged. Empty by design — see module docstring.
NEURAL: dict = {}

#: `grid` = calibrated 2026-08-20 (see module docstring for the material).
GRID: dict = {
    "torso_half": (0.0695, 0.119, 0.239),   # 0.50 / 0.70 / 1.00 x MESH_HALF
    "enter_thresh": 0.02,                   # correct once deeper than (enter-margin)
    "margin": 0.02,                         # ... and push out to this clearance
}

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
