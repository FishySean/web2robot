"""
Standalone finger-following verification render for M7 (Step A-5).

Drives M7's two hands with the REAL per-frame retargeted finger trajectory
(from m7_test_out/trajectory.npz) on a fixed, sensible seed arm pose so the
palms face the camera.  This isolates the two things to verify:
  1. palm orientation (arm pose is fixed & known → any wrong twist is visible)
  2. finger bend direction + magnitude following the real human hand.

Arm IK is deliberately NOT used here: with the borrowed robonaut2 root model the
arm targets are unreachable (IK 0%), so we pin the arms at the config seed pose.
This render says nothing about arm-following on real data — only fingers + the
static hand frame orientation.
"""
from pathlib import Path
import numpy as np
import mujoco
import imageio.v2 as imageio

from web2robot.robots.m7.env import M7Env
from web2robot.robots.m7.config import CONFIG

OUT_DIR = Path("/mnt/vlm/fanshaoheng/phase1_repro/m7_test_out")
TRAJ = np.load(OUT_DIR / "trajectory.npz", allow_pickle=True)

qLf = TRAJ["q_left_fingers"]                    # (T, 12)
qRf = TRAJ["q_right_fingers"]                   # (T, 12)
Lnames = [str(x) for x in TRAJ["left_finger_joint_names"]]
Rnames = [str(x) for x in TRAJ["right_finger_joint_names"]]
T = qLf.shape[0]
fps = int(TRAJ["fps"]) if "fps" in TRAJ else 20

env = M7Env(start_config={"left": np.array(CONFIG["start_config"]["left"]),
                          "right": np.array(CONFIG["start_config"]["right"])})
m, d = env.model, env.data

# two tight close-up panels: left hand (top) and right hand (bottom).
H, W = 360, 640
renderer = mujoco.Renderer(m, height=H, width=W)

def make_cam(az):
    c = mujoco.MjvCamera()
    c.azimuth, c.elevation, c.distance = az, -15, 0.32
    return c
camL = make_cam(60)      # look at the left hand from front-left
camR = make_cam(120)     # look at the right hand from front-right

frames = []
for t in range(T):
    env.reset()
    env.set_finger_joints(qLf[t], Lnames)
    env.set_finger_joints(qRf[t], Rnames)
    mujoco.mj_forward(m, d)
    camL.lookat[:] = d.xpos[env._body_ids["left"]]
    camR.lookat[:] = d.xpos[env._body_ids["right"]]
    renderer.update_scene(d, camL); imgL = renderer.render().copy()
    renderer.update_scene(d, camR); imgR = renderer.render().copy()
    frames.append(np.concatenate([imgL, imgR], axis=0))   # (720,640)

out = OUT_DIR / "m7_fingers_follow.mp4"
imageio.mimsave(out, frames, fps=fps)
print(f"wrote {out}  ({T} frames @ {fps}fps)")

# also print open/close summary so numbers back the video
print("left  index_mcp range :", round(float(qLf[:, 4].min()),3), "->", round(float(qLf[:, 4].max()),3))
print("right index_mcp range :", round(float(qRf[:, 4].min()),3), "->", round(float(qRf[:, 4].max()),3))
print("left  curl mean per frame min/max:",
      round(float(qLf[:, [4,5,6,7,8,9,10,11]].sum(1).min()),2), "->",
      round(float(qLf[:, [4,5,6,7,8,9,10,11]].sum(1).max()),2))

# dump sample frames for inspection
for i, idx in enumerate([0, T//3, 2*T//3, T-1]):
    imageio.imwrite(f"/tmp/m7f_{i}.png", frames[idx])
print("saved 4 sample PNGs to /tmp")
