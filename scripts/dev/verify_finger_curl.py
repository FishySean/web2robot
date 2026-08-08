"""Decisive finger check: does set_finger_joints actually curl the mesh?

(1) load the SAME trajectory.npz used by the videos
(2) apply frame-30 right-hand values, READ BACK qpos to prove they're not zeroed
(3) render right hand: all-zeros (open) vs frame-30 (should be a grip), side by side
"""
import numpy as np, mujoco, imageio.v2 as imageio

from _devcli import parser, load_traj
from web2robot.robots.m7.env import M7Env

tr, OUT = load_traj(parser(__doc__).parse_args())
qRf = tr["q_right_fingers"]; Rn = [str(x) for x in tr["right_finger_joint_names"]]
print("trajectory.npz right finger names:", Rn)
print("frame30 values:", np.round(qRf[30], 2))

env = M7Env(); m, d = env.model, env.data

def apply_and_readback(vals):
    env.reset()
    env.set_finger_joints(vals, Rn)
    mujoco.mj_forward(m, d)
    got = {}
    for nm in ["right_hand_mid_joint1","right_hand_mid_joint2",
               "right_hand_ring_joint1","right_hand_index_joint1"]:
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, nm)
        got[nm] = float(d.qpos[m.jnt_qposadr[jid]]) if jid>=0 else None
    return got

print("readback @ all-zeros :", {k:round(v,2) for k,v in apply_and_readback(np.zeros(12)).items()})
print("readback @ frame30   :", {k:round(v,2) for k,v in apply_and_readback(qRf[30]).items()})

R = mujoco.Renderer(m, height=480, width=480)
def cam(tgt):
    c = mujoco.MjvCamera(); c.azimuth, c.elevation, c.distance = 180, -18, 0.35
    c.lookat[:] = tgt; return c

imgs = []
for label, vals in [("OPEN(zeros)", np.zeros(12)), ("frame30(data)", qRf[30])]:
    env.reset()
    env.set_finger_joints(vals, Rn)
    mujoco.mj_forward(m, d)
    R.update_scene(d, cam(d.xpos[env._body_ids["right"]]))
    imgs.append(R.render().copy())
imageio.imwrite(OUT / "finger_open_vs_grip.png", np.concatenate(imgs, axis=1))
print(f"saved {OUT}/finger_open_vs_grip.png  (left=open, right=frame30 data)")
