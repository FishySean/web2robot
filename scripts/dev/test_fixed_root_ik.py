"""Test the user's hypothesis: does M7 IK need a trained root model, or is a
FIXED transform + workspace_center enough to make the arms move?

We bypass estimate_root_poses entirely. We set:
  R_per_frame = constant rotation, t_per_frame = constant
  workspace_center = M7's own comfortable reach (from FK at start_config)
then run cam_to_root_targets + WristIK.solve_batch and report IK convergence.
"""
import numpy as np, torch
from scipy.spatial.transform import Rotation
from utils.clip_io import SamplesSequence
from utils.pose_utils import cam_to_root_targets, rescale_bilateral_separation
from kinematics.wrist_ik import WristIK, RobotIKConfig
from web2robot.robots.m7.config import CONFIG

seq = SamplesSequence("examples/-QALmP1nHtM_678.2_682.2")
T = seq.n_frames
left_cam = seq.get_window(0, T)["left_traj"]
right_cam = seq.get_window(0, T)["right_traj"]

# --- IK solvers (CPU), seeded at start_config like the real pipeline ---
iks = {}; seeds = {}
for side in ["left", "right"]:
    cfg = RobotIKConfig.m7(side)
    seed = np.array(CONFIG["start_config"][side], dtype=np.float32)
    iks[side] = WristIK(side=side, robot=cfg, device="cpu", q_default=seed)
    seeds[side] = seed

# --- workspace_center = M7's own hand position in root frame at start_config ---
wsc = {}
for side in ["left", "right"]:
    q = torch.tensor(seeds[side]).unsqueeze(0)
    tf = iks[side].chain.forward_kinematics(q)
    wsc[side] = tf.get_matrix()[0, :3, 3].numpy()
print("workspace_center (M7 reach @ start):", {k: np.round(v, 3) for k, v in wsc.items()})

# --- FIXED transform: map camera frame -> robot torso frame ---
# camera: +x right, +y down, +z forward (typical).  robot torso: +x forward, +z up.
# Try a few fixed rotations; t is irrelevant because workspace_center re-centers position.
candidates = {
    "identity":        np.eye(3),
    "cam2robot_zxy":   Rotation.from_euler("xyz", [-90, 0, -90], degrees=True).as_matrix(),
    "cam2robot_flip":  Rotation.from_euler("xyz", [90, 0, 90], degrees=True).as_matrix(),
}

for name, R in candidates.items():
    R_pf = np.tile(R, (T, 1, 1)); t_pf = np.zeros((T, 3))
    lp, lq, rp, rq = cam_to_root_targets(left_cam, right_cam, R_pf, t_pf, wsc)
    rates = {}
    for side, (p, qa) in [("left", (lp, lq)), ("right", (rp, rq))]:
        qs, info = iks[side].solve_batch(torch.tensor(p), torch.tensor(qa),
                                         torch.tensor(np.tile(seeds[side], (T, 1))))
        conv = info["converged"].float().mean().item()
        pe = info["pos_err"][info["converged"]].mean().item() if info["converged"].any() else float("nan")
        rates[side] = (100*conv, pe)
    print(f"{name:16s}  L={rates['left'][0]:5.1f}% (pe={rates['left'][1]:.3f})   "
          f"R={rates['right'][0]:5.1f}% (pe={rates['right'][1]:.3f})")
