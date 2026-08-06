"""Routing suggestion for pipeline step 2.

Stage 1 does not own this decision -- step 2 does. What stage 1 owes step 2 is
the labels its rules need, plus a precomputed suggestion so the handoff can be
tested before step 2 exists.

The rules are transcribed from VIDEO_SELECTION_GUIDE.md, which records a month
of manual trial and error:

  view       third-person -> EgoInfinity retarget stack (validated on M7)
             first-person -> the alternative method for egocentric footage
  perception HaWoR needs ALL THREE of: real camera translation (parallax),
             textured background, and enough duration/frames. Any one missing
             -> WiLoR + MoGe with no SLAM. Uncertain -> WiLoR (the safe default).

The asymmetry is deliberate and comes from a real failure: with a plain
background HaWoR's SLAM scale collapsed to ~0.01 and its outputs went largely
NaN, which is worse than WiLoR's honest lack of world motion.
"""
from typing import Optional, List, Tuple

from .schema import ViewClass, CameraMotion


def suggest(view_class: str, camera_motion: str, bg_texture: str,
            duration_ok: bool) -> Tuple[Optional[str], List[str]]:
    """-> (route string or None, rationale lines)."""
    why: List[str] = []

    # ---- which retarget stack ----
    if view_class == ViewClass.THIRD_PERSON_BODY.value:
        view_route = "egoinfinity"
        why.append("third-person with body framing -> EgoInfinity retarget stack")
    elif view_class == ViewClass.HANDS_ONLY.value:
        view_route = None
        why.append("hands-only framing: view class undetermined at stage 1, "
                   "step 2 must classify first- vs third-person before routing")
    else:
        return None, ["no usable framing -> no route"]

    # ---- which perception front end ----
    three = {
        "parallax": camera_motion == CameraMotion.MOVING.value,
        "texture": bg_texture == "rich",
        "duration": duration_ok,
    }
    unknown = (camera_motion == CameraMotion.UNKNOWN.value
               or bg_texture == "unknown")
    if unknown:
        perc = "wilor_moge"
        why.append("a HaWoR precondition is unknown -> defaulting to WiLoR+MoGe "
                   "(the safe branch)")
    elif all(three.values()):
        perc = "hawor"
        why.append("camera translates, background is textured, duration is "
                   "sufficient -> HaWoR (all three preconditions met)")
    else:
        missing = [k for k, v in three.items() if not v]
        perc = "wilor_moge"
        why.append(f"HaWoR precondition(s) not met ({', '.join(missing)}) "
                   f"-> WiLoR+MoGe without SLAM")

    route = f"{view_route}/{perc}" if view_route else f"pending_view/{perc}"
    return route, why
