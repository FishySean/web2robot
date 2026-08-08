"""First-pass calibration check for m7 SAMPLE_CONFIG (no JAX needed).

Runs taskspace_traj (the training-time sampler) for both arms using WristIK,
then FK's the sampled joint trajectories to inspect the wrist WORKSPACE that the
training data will cover.  Reports, per side:
  - xyz bounding box of wrist positions (torso/world frame)
  - fraction of frames in front of torso (x>0), which we want high for manipulation
  - arm-crossing rate (train.py rejects samples where left.y <= right.y)
This tells us whether the sampled configs are natural forward-reaching poses
BEFORE we spend time on the full JAX training.
"""
import numpy as np, torch
from kinematics.wrist_ik import WristIK, RobotIKConfig
from web2robot.robots.m7.config import CONFIG
from web2robot.robots.m7.sample_config import SAMPLE_CONFIG
from sim.traj_sampler import taskspace_traj

N = 256
rng = np.random.default_rng(0)
iks = {}
for side in ("left", "right"):
    iks[side] = WristIK(side=side, robot=RobotIKConfig.m7(side), device="cpu",
                        q_default=np.array(CONFIG["start_config"][side], np.float32))

res = {}
for side in ("left", "right"):
    q_ref = np.array(CONFIG["start_config"][side], np.float32)
    traj = taskspace_traj(N, side, rng, iks[side], SAMPLE_CONFIG, q_ref=q_ref)  # (N,T,7)
    Nn, T, nd = traj.shape
    with torch.no_grad():
        pos, _ = iks[side]._fk(torch.tensor(traj.reshape(-1, nd), dtype=torch.float32))
    pos = pos.cpu().numpy().reshape(Nn, T, 3)
    res[side] = pos
    bb_lo = pos.reshape(-1, 3).min(0); bb_hi = pos.reshape(-1, 3).max(0)
    front = (pos[..., 0] > 0).mean()
    print(f"[{side}] wrist workspace (torso frame):")
    print(f"    x[fwd] {bb_lo[0]:+.3f}..{bb_hi[0]:+.3f}   "
          f"y[lat] {bb_lo[1]:+.3f}..{bb_hi[1]:+.3f}   "
          f"z[up] {bb_lo[2]:+.3f}..{bb_hi[2]:+.3f}")
    print(f"    frac frames in front (x>0): {front*100:.1f}%")

# arm crossing: left.y should stay > right.y (per train.py reject rule)
ly = res["left"][..., 1]; ry = res["right"][..., 1]
cross = (ly <= ry).mean()
print(f"\narm-crossing rate (left.y <= right.y): {cross*100:.1f}%  (train.py rejects these)")
print(f"left.y  mean {ly.mean():+.3f}   right.y mean {ry.mean():+.3f}")
