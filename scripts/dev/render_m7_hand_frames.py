"""Render M7 right & left hand close-ups at specific frames, save PNGs for
side-by-side comparison with the human input frames.  Uses the REAL retargeted
finger trajectory (independent of arm IK / root model)."""
from pathlib import Path
import numpy as np, mujoco, imageio.v2 as imageio
from web2robot.robots.m7.env import M7Env
from web2robot.robots.m7.config import CONFIG

OUT = Path("/mnt/vlm/fanshaoheng/phase1_repro/m7_test_out")
tr = np.load(OUT / "trajectory.npz", allow_pickle=True)
qLf, qRf = tr["q_left_fingers"], tr["q_right_fingers"]
Ln = [str(x) for x in tr["left_finger_joint_names"]]
Rn = [str(x) for x in tr["right_finger_joint_names"]]

env = M7Env(start_config={"left": np.array(CONFIG["start_config"]["left"]),
                          "right": np.array(CONFIG["start_config"]["right"])})
m, d = env.model, env.data
R = mujoco.Renderer(m, height=480, width=480)

def cam(az, tgt):
    c = mujoco.MjvCamera(); c.azimuth, c.elevation, c.distance = az, -20, 0.28
    c.lookat[:] = tgt; return c

for f in [10, 30, 50]:
    env.reset()
    env.set_finger_joints(qLf[f], Ln)
    env.set_finger_joints(qRf[f], Rn)
    mujoco.mj_forward(m, d)
    R.update_scene(d, cam(120, d.xpos[env._body_ids["right"]])); imgR = R.render().copy()
    R.update_scene(d, cam(60,  d.xpos[env._body_ids["left"]]));  imgL = R.render().copy()
    imageio.imwrite(f"/tmp/m7R_{f}.png", imgR)
    imageio.imwrite(f"/tmp/m7L_{f}.png", imgL)
    print(f"frame {f}: right curl(mid/ring/pinky mcp+pip)={qRf[f,[6,7,8,9,10,11]].sum():.2f} "
          f"index_mcp={qRf[f,4]:.2f}  left curl={qLf[f,[6,7,8,9,10,11]].sum():.2f}")
print("saved /tmp/m7R_*.png and /tmp/m7L_*.png")
