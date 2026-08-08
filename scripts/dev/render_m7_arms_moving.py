"""Robot-sim with MOVING ARMS via the fixed-transform path (NO trained root model).

Pipeline: camera wrist trajectory -> fixed identity transform + M7 workspace_center
          -> cam_to_root_targets -> WristIK.solve_batch -> per-frame arm angles.
Fingers from the real retargeted trajectory.  Renders full-body front view.
"""
from pathlib import Path
import numpy as np, torch, mujoco, imageio.v2 as imageio
from utils.clip_io import SamplesSequence
from utils.pose_utils import cam_to_root_targets
from kinematics.wrist_ik import WristIK, RobotIKConfig
from web2robot.robots.m7.env import M7Env
from web2robot.robots.m7.config import CONFIG

CLIP = "examples/-QALmP1nHtM_678.2_682.2"
OUT = Path("/mnt/vlm/fanshaoheng/phase1_repro/m7_test_out")

seq = SamplesSequence(CLIP); T = seq.n_frames
left_cam = seq.get_window(0, T)["left_traj"]; right_cam = seq.get_window(0, T)["right_traj"]

iks = {}; seeds = {}; wsc = {}
for side in ["left", "right"]:
    seed = np.array(CONFIG["start_config"][side], dtype=np.float32)
    ik = WristIK(side=side, robot=RobotIKConfig.m7(side), device="cpu", q_default=seed)
    iks[side] = ik; seeds[side] = seed
    tf = ik.chain.forward_kinematics(torch.tensor(seed).unsqueeze(0))
    wsc[side] = tf.get_matrix()[0, :3, 3].numpy()

R_pf = np.tile(np.eye(3), (T, 1, 1)); t_pf = np.zeros((T, 3))
lp, lq, rp, rq = cam_to_root_targets(left_cam, right_cam, R_pf, t_pf, wsc)
qL, iL = iks["left"].solve_batch(torch.tensor(lp), torch.tensor(lq), torch.tensor(np.tile(seeds["left"], (T,1))))
qR, iR = iks["right"].solve_batch(torch.tensor(rp), torch.tensor(rq), torch.tensor(np.tile(seeds["right"], (T,1))))
qL = qL.numpy(); qR = qR.numpy()
print(f"arm IK conv: L={100*iL['converged'].float().mean():.0f}%  R={100*iR['converged'].float().mean():.0f}%")

tr = np.load(OUT / "trajectory.npz", allow_pickle=True)
qLf, qRf = tr["q_left_fingers"], tr["q_right_fingers"]
Ln = [str(x) for x in tr["left_finger_joint_names"]]; Rn = [str(x) for x in tr["right_finger_joint_names"]]

env = M7Env(); m, d = env.model, env.data
R = mujoco.Renderer(m, height=600, width=540)
def cam(az, el, dist, tgt):
    c = mujoco.MjvCamera(); c.azimuth, c.elevation, c.distance = az, el, dist
    c.lookat[:] = tgt; return c

frames = []
for t in range(T):
    env.reset()
    env.set_arm_joints("left", qL[t].astype(np.float64)); env.set_arm_joints("right", qR[t].astype(np.float64))
    env.set_finger_joints(qLf[t], Ln); env.set_finger_joints(qRf[t], Rn)
    mujoco.mj_forward(m, d)
    R.update_scene(d, cam(180, -10, 1.7, [0.2, 0, 0.05])); a = R.render().copy()
    R.update_scene(d, cam(230, -15, 1.3, [0.25, 0, 0.1])); b = R.render().copy()
    frames.append(np.concatenate([a, b], axis=1))

out = OUT / "m7_arms_moving.mp4"
imageio.mimsave(out, frames, fps=int(tr["fps"]) if "fps" in tr else 15, codec="libx264")
print(f"wrote {out}")
for f in [5, 25, 45, 60]:
    if f < T: imageio.imwrite(f"/tmp/arms_{f}.png", frames[f])
print("saved /tmp/arms_*.png")
