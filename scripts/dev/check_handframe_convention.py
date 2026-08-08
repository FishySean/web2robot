"""Compare hand_frame LOCAL-axis convention across robots, for BOTH hands.

At the home pose, for each robot / each side, express two physical directions
in the hand_frame's own local coordinates:
  finger_dir : wrist(hand_frame) -> a fingertip/knuckle   (where the fingers point)
  thumb_dir  : wrist(hand_frame) -> a thumb link          (where the thumb points)
Also the palm normal = finger_dir x thumb_dir.

⚠️ LESSON (2026-07-24): an earlier version of this check only tested the LEFT
hand and used degenerate r2 bodies, and wrongly concluded "both hands share the
same convention".  In fact BOTH known-good robots (g1, r2) MIRROR the palm
normal between hands (left palm +z <-> right palm -z, or vice-versa).  M7 had
been built with both hands identical -> right hand palm/thumb flipped 180 deg.
So this script now checks BOTH hands and ASSERTS the two known-good dexterous
robots mirror their palm normal.  Never verify one side only.
"""
import numpy as np, mujoco

from web2robot.paths import P

# g1 / r2 是**上游**的机器人资产（参照组，不是我们的），m7 是我们的。
_UPSTREAM_ROBOTS = P.root("egoinfinity") / "retarget" / "robots"


def local_dirs(scene_xml, hand_frame, finger_body, thumb_body):
    m = mujoco.MjModel.from_xml_path(scene_xml)
    d = mujoco.MjData(m)
    kid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "home")
    if kid >= 0:
        mujoco.mj_resetDataKeyframe(m, d, kid)
    mujoco.mj_forward(m, d)
    bid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
    hf = bid(hand_frame)
    R = d.xmat[hf].reshape(3, 3)            # hand_frame world rotation
    p = d.xpos[hf]
    fd = d.xpos[bid(finger_body)] - p
    td = d.xpos[bid(thumb_body)] - p
    fd_l = R.T @ fd
    td_l = R.T @ td
    pn_l = np.cross(fd_l, td_l)
    n = lambda v: v / (np.linalg.norm(v) + 1e-9)
    return n(fd_l), n(td_l), n(pn_l)


def dominant_axis(v):
    i = int(np.argmax(np.abs(v)))
    sign = "+" if v[i] > 0 else "-"
    return f"{sign}{'xyz'[i]}", np.round(v, 2)


# (scene, {side: (hand_frame, finger_body, thumb_body)})
ROBOTS = {
    "g1": (str(_UPSTREAM_ROBOTS / "unitree_g1" / "scene_vis.xml"), {
        "left":  ("left_hand_frame",  "left_three_link",  "left_zero_link"),
        "right": ("right_hand_frame", "right_three_link", "right_zero_link"),
    }),
    "r2": (str(_UPSTREAM_ROBOTS / "robonaut2" / "scene_vis.xml"), {
        "left":  ("left_hand_frame",  "/r2/left_middle_distal",  "/r2/left_thumb_distal"),
        "right": ("right_hand_frame", "/r2/right_middle_distal", "/r2/right_thumb_distal"),
    }),
    "m7": (str(P.asset("m7_scene")), {
        "left":  ("left_hand_frame",  "left_hand_mid_link2",  "left_hand_thumb_rota_link2"),
        "right": ("right_hand_frame", "right_hand_mid_link2", "right_hand_thumb_rota_link2"),
    }),
}


def palm_axis_sign(pn):
    """Return signed dominant axis of palm normal, e.g. '+z' / '-z'."""
    return dominant_axis(pn)[0]


results = {}
for name, (scene, sides) in ROBOTS.items():
    print(f"\n{name}:")
    results[name] = {}
    for side, (hf, fb, tb) in sides.items():
        fd, td, pn = local_dirs(scene, hf, fb, tb)
        results[name][side] = palm_axis_sign(pn)
        print(f"  {side:5s} finger={dominant_axis(fd)[0]}  "
              f"thumb={dominant_axis(td)[0]}  palm_normal={dominant_axis(pn)[0]}  "
              f"pn={np.round(pn, 2)}")

# ── assertions ────────────────────────────────────────────────────────────────
print("\n── convention checks ──")
ok = True
for name in ("g1", "r2", "m7"):
    lp, rp = results[name]["left"], results[name]["right"]
    lax, rax = lp[1], rp[1]           # axis letter
    lsign, rsign = lp[0], rp[0]       # + / -
    mirrored = (lax == rax) and (lsign != rsign)
    status = "MIRRORED ✓" if mirrored else "NOT mirrored ✗"
    print(f"  {name}: left palm {lp} / right palm {rp}  -> {status}")
    if name in ("g1", "r2") and not mirrored:
        print(f"    !! {name} is a known-good robot but palm normal is not mirrored "
              f"-- body-name choice is likely wrong, fix the reference")
        ok = False
    if name == "m7" and not mirrored:
        print(f"    !! M7 palm normal NOT mirrored between hands -- right hand_frame "
              f"quat is wrong (should be left mirrored about finger axis)")
        ok = False

print(f"\nVERDICT: {'PASS ✓ all robots mirror palm normal between hands' if ok else 'FAIL ✗'}")
