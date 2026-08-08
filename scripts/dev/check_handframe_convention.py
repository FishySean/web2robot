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

再往上一档：``--traj <run_dir>`` 拿一段**真实重定向轨迹**逐帧验 M7。home 姿态是资产里
写死的静态姿态，它验的是"MJCF 建对了没有"；逐帧验的是"整段动起来之后，约定有没有在
某些姿态下崩掉"。实测 fill_jar 216 帧 0 违反。默认不带 ``--traj`` 时输出与迁移前逐字
一致（回归基准就是拿默认输出比的）。

用法（``m7_tool.sh`` 会 cd 到上游 ``retarget/``，所以 run_dir 可以写相对路径）::

    scripts/dev/m7_tool.sh check_handframe_convention.py
    scripts/dev/m7_tool.sh check_handframe_convention.py --traj runs/m7/validation/fill_jar
"""
import argparse
from pathlib import Path

import numpy as np, mujoco

from web2robot.paths import P

_ap = argparse.ArgumentParser(description=__doc__,
                              formatter_class=argparse.RawDescriptionHelpFormatter)
_ap.add_argument("--traj", type=Path, default=None,
                 help="一次 s4_retarget.sh 输出目录，额外逐帧验 M7 的 hand_frame 约定")
ARGS = _ap.parse_args()

# M7 修好之后每一帧都该满足的轴向组合：finger 两手同向，thumb / palm 镜像。
M7_EXPECT = {"left": ("+y", "-x", "+z"), "right": ("+y", "+x", "-z")}

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


def check_traj(run_dir):
    """整段轨迹逐帧验 M7 的 hand_frame 约定；返回 True 表示一帧都没违反。

    import 放在函数里：不带 ``--traj`` 的默认路径不该为此多加载 M7Env。
    """
    from web2robot.robots.m7.config import CONFIG as M7
    from web2robot.robots.m7.env import M7Env

    npz = run_dir / "trajectory.npz"
    if not npz.is_file():
        raise SystemExit(f"找不到 {npz}\n--traj 要指向 s4_retarget.sh 的输出目录")
    d = np.load(npz, allow_pickle=True)
    qL, qR = d["q_left"], d["q_right"]
    QLf, QRf = d["q_left_fingers"], d["q_right_fingers"]
    fj = [str(n).replace("left_", "").replace("_joint", "")
          for n in d["left_finger_joint_names"]]

    env = M7Env(mjcf_path=M7.get("scene_path_fingers", M7["scene_path"]),
                start_config=M7["start_config"])
    m, dat = env.model, env.data
    bid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
    bodies = {s: (bid(f"{s}_hand_frame"), bid(f"{s}_hand_mid_link2"),
                  bid(f"{s}_hand_thumb_rota_link2")) for s in ("left", "right")}

    T = len(qL)
    bad, seen = [], {"left": set(), "right": set()}
    for t in range(T):
        env.set_arm_joints("left", qL[t]); env.set_arm_joints("right", qR[t])
        env.set_finger_joints(QLf[t], [f"left_{n}_joint" for n in fj])
        env.set_finger_joints(QRf[t], [f"right_{n}_joint" for n in fj])
        mujoco.mj_forward(m, dat)
        got = {}
        for s, (hf, fb, tb) in bodies.items():
            R, pos = dat.xmat[hf].reshape(3, 3), dat.xpos[hf]
            fd = R.T @ (dat.xpos[fb] - pos)
            td = R.T @ (dat.xpos[tb] - pos)
            got[s] = (dominant_axis(fd)[0], dominant_axis(td)[0],
                      dominant_axis(np.cross(fd, td))[0])
            seen[s].add(got[s])
        if any(got[s] != M7_EXPECT[s] for s in ("left", "right")):
            bad.append((t, got))

    print(f"\n── m7 whole-trajectory check ({run_dir}) ──")
    print(f"  帧数 {T}")
    for s in ("left", "right"):
        print(f"  {s:5s} 整段出现过的轴向组合: {sorted(seen[s])}   期望 {M7_EXPECT[s]}")
    if bad:
        print(f"  违反约定的帧: {len(bad)}/{T}  ✗ 例: {bad[:3]}")
    else:
        print(f"  违反约定的帧: 0/{T}  ✓")
    return not bad


if ARGS.traj is not None:
    ok = check_traj(ARGS.traj) and ok

print(f"\nVERDICT: {'PASS ✓ all robots mirror palm normal between hands' if ok else 'FAIL ✗'}")
