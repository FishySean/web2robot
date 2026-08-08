"""Render M7 right & left hand close-ups at specific frames, save PNGs for
side-by-side comparison with the human input frames.  Uses the REAL retargeted
finger trajectory (independent of arm IK / root model).

用法::

    scripts/dev/m7_tool.sh render_m7_hand_frames.py <run_dir> [--frames 10 30 50] [--out DIR]

``run_dir`` 是一次 ``s4_retarget.sh`` 的输出目录（里面要有 ``trajectory.npz``）。
迁移前这里写死了 ``/mnt/vlm/fanshaoheng/phase1_repro/m7_test_out``（重构前的一次性
目录）—— 见 ``_devcli.py`` 里为什么改成必填参数。
"""
import numpy as np, mujoco, imageio.v2 as imageio

from _devcli import parser, load_traj
from web2robot.robots.m7.env import M7Env
from web2robot.robots.m7.config import CONFIG

p = parser(__doc__)
p.add_argument("--frames", type=int, nargs="+", default=[10, 30, 50])
args = p.parse_args()
tr, OUT = load_traj(args)

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

for f in args.frames:
    env.reset()
    env.set_finger_joints(qLf[f], Ln)
    env.set_finger_joints(qRf[f], Rn)
    mujoco.mj_forward(m, d)
    R.update_scene(d, cam(120, d.xpos[env._body_ids["right"]])); imgR = R.render().copy()
    R.update_scene(d, cam(60,  d.xpos[env._body_ids["left"]]));  imgL = R.render().copy()
    imageio.imwrite(OUT / f"m7R_{f}.png", imgR)
    imageio.imwrite(OUT / f"m7L_{f}.png", imgL)
    print(f"frame {f}: right curl(mid/ring/pinky mcp+pip)={qRf[f,[6,7,8,9,10,11]].sum():.2f} "
          f"index_mcp={qRf[f,4]:.2f}  left curl={qLf[f,[6,7,8,9,10,11]].sum():.2f}")
print(f"saved {OUT}/m7R_*.png and {OUT}/m7L_*.png")
