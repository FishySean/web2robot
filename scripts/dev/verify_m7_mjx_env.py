"""Step 0g: verify the m7 JaxVecEnv warms up and MJX FK matches plain MuJoCo.

The root-frame model is trained on wrist trajectories produced by MJX forward
kinematics (mjx.kinematics inside JaxVecEnv).  If MJX FK disagreed with the
plain-MuJoCo FK we already validated (verify_m7_mjx_fk.py: 0mm/0deg vs m7.xml),
the training targets would be silently wrong.  So here we:

  1. Build JaxVecEnv(m7) and warmup() it  -> proves m7_mjx.xml JIT-compiles
     under MJX on the GPU (also smoke-tests the cuDNN-version warning is benign).
  2. Push N random arm configs through BOTH MJX (JaxVecEnv.step_joints) and a
     plain MuJoCo MjData on the SAME m7_mjx.xml, and compare hand_frame world
     pos/quat.

VERDICT MATCH => the training FK is trustworthy; safe to proceed to smoke test.
"""
import time

import numpy as np
import jax
import jax.numpy as jnp
import mujoco

from web2robot.robots.m7.config import ENV_SPEC
from web2robot.robots.m7.env import _ARM_JOINTS, _EE_BODY
from sim.robot_config import RobotConfig
from sim.vec_env_jax import JaxVecEnv

# 机器人定义只出纯 dict（ENV_SPEC），框架类型在框架侧包 —— 和上游
# sim/robots/__init__.py 里的适配是同一件事。见 src/web2robot/robots/m7/__init__.py。
ENV_CONFIG = RobotConfig(**ENV_SPEC)

N = 64
rng = np.random.default_rng(0)

# ── build + warmup (JIT compile) ──────────────────────────────────────────────
print(f"building JaxVecEnv(m7, num_envs={N}) ...", flush=True)
env = JaxVecEnv(ENV_CONFIG, num_envs=N)
print("warming up (JIT compile of mjx.kinematics on m7_mjx.xml) ...", flush=True)
t0 = time.time()
env.warmup()
jax.block_until_ready(env.reset())
print(f"  warmup OK in {time.time() - t0:.1f}s", flush=True)

# ── sample N random arm configs within joint limits ───────────────────────────
model = mujoco.MjModel.from_xml_path(str(ENV_CONFIG.mjcf_path))
data = mujoco.MjData(model)

lo = {}
hi = {}
for side, joints in _ARM_JOINTS.items():
    jids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in joints]
    rng_lo = np.array([model.jnt_range[j][0] for j in jids])
    rng_hi = np.array([model.jnt_range[j][1] for j in jids])
    lo[side], hi[side] = rng_lo, rng_hi

q = {side: rng.uniform(lo[side], hi[side], size=(N, 7)).astype(np.float32)
     for side in _ARM_JOINTS}

# ── MJX FK ────────────────────────────────────────────────────────────────────
joint_dict = {side: jnp.asarray(q[side]) for side in _ARM_JOINTS}
state = env.step_joints(env.reset(), joint_dict)
obs = env.get_obs(state)
mjx_pos = {s: np.asarray(obs[s]["pos"]) for s in _ARM_JOINTS}
mjx_quat = {s: np.asarray(obs[s]["quat"]) for s in _ARM_JOINTS}

# ── plain MuJoCo FK on the SAME xml, per env ─────────────────────────────────
qadr = {side: [model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j)]
               for j in _ARM_JOINTS[side]] for side in _ARM_JOINTS}
bid = {side: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, _EE_BODY[side])
       for side in _ARM_JOINTS}

mj_pos = {s: np.zeros((N, 3)) for s in _ARM_JOINTS}
mj_quat = {s: np.zeros((N, 4)) for s in _ARM_JOINTS}
for k in range(N):
    mujoco.mj_resetData(model, data)
    for side in _ARM_JOINTS:
        for i, adr in enumerate(qadr[side]):
            data.qpos[adr] = q[side][k, i]
    mujoco.mj_forward(model, data)
    for side in _ARM_JOINTS:
        mj_pos[side][k] = data.xpos[bid[side]]
        mj_quat[side][k] = data.xquat[bid[side]]

# ── compare ───────────────────────────────────────────────────────────────────
def quat_ang_deg(a, b):
    d = np.abs(np.sum(a * b, axis=-1)).clip(0, 1)
    return np.degrees(2 * np.arccos(d))

worst_pos = 0.0
worst_ang = 0.0
for side in _ARM_JOINTS:
    dp = np.linalg.norm(mjx_pos[side] - mj_pos[side], axis=-1)
    da = quat_ang_deg(mjx_quat[side], mj_quat[side])
    print(f"[{side}] pos err max {dp.max()*1000:.4f} mm  mean {dp.mean()*1000:.4f} mm | "
          f"ori err max {da.max():.4f} deg  mean {da.mean():.4f} deg")
    worst_pos = max(worst_pos, dp.max())
    worst_ang = max(worst_ang, da.max())

ok = worst_pos < 1e-4 and worst_ang < 1e-2  # <0.1mm, <0.01deg
print(f"\nVERDICT: {'MATCH ✓' if ok else 'MISMATCH ✗'}  "
      f"(worst pos {worst_pos*1000:.4f} mm, worst ori {worst_ang:.4f} deg)")
