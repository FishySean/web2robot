"""Verify m7_mjx.xml FK matches m7.xml for both hand_frames across random arm configs.
Sets identical arm joint angles on both models (waist & fingers at zero), runs
mj_forward, compares hand_frame world pos & quat.  This is the go/no-go check that
the MJX training model produces the same wrist trajectories as the real robot.

这是 M7 资产的 go/no-go 验收脚本之一。跑法（薄壳设好解释器和 PYTHONPATH）::

    scripts/dev/m7_tool.sh verify_m7_mjx_fk.py
"""
import numpy as np, mujoco

from web2robot.paths import P

mF = mujoco.MjModel.from_xml_path(str(P.asset("m7_mjcf")))   # full
mX = mujoco.MjModel.from_xml_path(str(P.asset("m7_mjx")))    # mjx
dF, dX = mujoco.MjData(mF), mujoco.MjData(mX)

ARMJ = {s: [f"{s}_shoulder_pitch_joint", f"{s}_shoulder_roll_joint", f"{s}_arm_yaw_joint",
            f"{s}_elbow_pitch_joint", f"{s}_elbow_yaw_joint", f"{s}_wrist_pitch_joint",
            f"{s}_wrist_roll_joint"] for s in ("left", "right")}

def qadr(m, jn):
    j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jn)
    return m.jnt_qposadr[j]

def bid(m, bn): return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, bn)

# joint limits from full model (sample within range)
lims = {}
for s in ("left", "right"):
    for jn in ARMJ[s]:
        j = mujoco.mj_name2id(mF, mujoco.mjtObj.mjOBJ_JOINT, jn)
        lims[jn] = mF.jnt_range[j]

rng = np.random.default_rng(0)
max_pos_err = 0.0
max_ang_err = 0.0
for trial in range(200):
    mujoco.mj_resetData(mF, dF); mujoco.mj_resetData(mX, dX)
    for s in ("left", "right"):
        for jn in ARMJ[s]:
            lo, hi = lims[jn]
            v = rng.uniform(lo, hi)
            dF.qpos[qadr(mF, jn)] = v
            dX.qpos[qadr(mX, jn)] = v
    mujoco.mj_forward(mF, dF); mujoco.mj_forward(mX, dX)
    for s in ("left", "right"):
        hf = f"{s}_hand_frame"
        pF, pX = dF.xpos[bid(mF, hf)], dX.xpos[bid(mX, hf)]
        qF, qX = dF.xquat[bid(mF, hf)], dX.xquat[bid(mX, hf)]
        pe = np.linalg.norm(pF - pX)
        dot = abs(float(np.dot(qF, qX)))
        ang = np.degrees(2 * np.arccos(min(1.0, dot)))
        max_pos_err = max(max_pos_err, pe)
        max_ang_err = max(max_ang_err, ang)

print(f"over 200 random arm configs, both hands:")
print(f"  max hand_frame position error : {max_pos_err*1000:.4f} mm")
print(f"  max hand_frame orientation err: {max_ang_err:.4f} deg")
print("VERDICT:", "MATCH ✓" if (max_pos_err < 1e-4 and max_ang_err < 0.01) else "MISMATCH ✗")
