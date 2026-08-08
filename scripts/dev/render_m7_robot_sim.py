"""M7 'robot-sim' style render for finger verification.

Arms are PINNED at a fixed presenting pose (arms cannot be retargeted on the
borrowed root model) so both hands are held out in front, clearly visible.
Only the FINGERS animate, driven by the real per-frame retargeted trajectory.
Two synced views: full-body front (left) + tight hands close-up (right).
"""
from pathlib import Path
import numpy as np, mujoco, imageio.v2 as imageio
from web2robot.robots.m7.env import M7Env

OUT = Path("/mnt/vlm/fanshaoheng/phase1_repro/m7_test_out")
tr = np.load(OUT / "trajectory.npz", allow_pickle=True)
qLf, qRf = tr["q_left_fingers"], tr["q_right_fingers"]
Ln = [str(x) for x in tr["left_finger_joint_names"]]
Rn = [str(x) for x in tr["right_finger_joint_names"]]
T = qLf.shape[0]
fps = int(tr["fps"]) if "fps" in tr else 15

# presenting arm pose: [sh_pitch, sh_roll, arm_yaw, elbow_pitch, elbow_yaw, wr_pitch, wr_roll]
armL = np.array([-0.9,  0.25, 0, -1.2, 0, 0, 0])
armR = np.array([-0.9, -0.25, 0, -1.2, 0, 0, 0])

env = M7Env()
m, d = env.model, env.data
Rn_ = mujoco.Renderer(m, height=540, width=480)

def cam(az, el, dist, tgt):
    c = mujoco.MjvCamera(); c.azimuth, c.elevation, c.distance = az, el, dist
    c.lookat[:] = tgt; return c

frames = []
for t in range(T):
    env.reset()
    env.set_arm_joints("left", armL); env.set_arm_joints("right", armR)
    env.set_finger_joints(qLf[t], Ln); env.set_finger_joints(qRf[t], Rn)
    mujoco.mj_forward(m, d)
    pL = d.xpos[env._body_ids["left"]]; pR = d.xpos[env._body_ids["right"]]
    mid = 0.5 * (pL + pR)
    # full body front
    Rn_.update_scene(d, cam(180, -12, 1.7, [0.25, 0, 0.15])); full = Rn_.render().copy()
    # tight hands
    Rn_.update_scene(d, cam(180, -18, 0.95, mid)); hands = Rn_.render().copy()
    frames.append(np.concatenate([full, hands], axis=1))

out = OUT / "m7_robot_sim_fingers.mp4"
imageio.mimsave(out, frames, fps=fps, codec="libx264")
print(f"wrote {out} ({T} frames @ {fps}fps, codec h264)")
for f in [10, 30, 50]:
    imageio.imwrite(f"/tmp/sim_{f}.png", frames[f])
print("saved /tmp/sim_10/30/50.png")
