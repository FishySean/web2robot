"""Compute corrected M7 hand_frame quats so its LOCAL axes match the pipeline
convention (reference = robonaut2, verified clean): finger=+y, thumb=-x, palm=+z.

Method (per side, uses measured physical directions -> no hand-derived sign errors):
  target world axes:
     y_world = finger_dir_world                       (local +y = fingers)
     z_world = normalize(finger_dir x thumb_dir)       (local +z = palm normal)
     x_world = y_world x z_world                        (right-handed)
  R_new_world = [x_world | y_world | z_world]
  new local quat = quat( R_parent_world^T @ R_new_world )

这个脚本是 ``assets/robots/m7/m7.xml`` 里两个 hand_frame quat 的**出处** ——
右手当初被建成和左手完全一样（palm/thumb 翻了 180°），是用它算出正确的镜像 quat
再写回 MJCF 的。留着是为了资产可追溯：改了 hand_frame 就重跑它，再用
``check_handframe_convention.py`` 验收。
"""
import numpy as np, mujoco

from web2robot.paths import P

SCENE = str(P.asset("m7_scene"))
SIDES = {
    "left":  ("left_hand_frame",  "left_wrist_roll_link",
              "left_hand_mid_link2",  "left_hand_thumb_rota_link2"),
    "right": ("right_hand_frame", "right_wrist_roll_link",
              "right_hand_mid_link2", "right_hand_thumb_rota_link2"),
}

m = mujoco.MjModel.from_xml_path(SCENE)
d = mujoco.MjData(m)
kid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "home")
if kid >= 0: mujoco.mj_resetDataKeyframe(m, d, kid)
mujoco.mj_forward(m, d)
bid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
n = lambda v: v / (np.linalg.norm(v) + 1e-9)

for side, (hf, parent, fbody, tbody) in SIDES.items():
    Rframe = d.xmat[bid(hf)].reshape(3, 3)
    Rparent = d.xmat[bid(parent)].reshape(3, 3)
    p = d.xpos[bid(hf)]
    finger_w = n(d.xpos[bid(fbody)] - p)
    thumb_w  = n(d.xpos[bid(tbody)] - p)

    y_w = finger_w
    z_w = n(np.cross(finger_w, thumb_w))
    x_w = n(np.cross(y_w, z_w))
    R_new_world = np.column_stack([x_w, y_w, z_w])

    R_new_local = Rparent.T @ R_new_world
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, R_new_local.flatten())
    # sanity: recompute local axes of physical dirs under R_new_world
    fl = R_new_world.T @ finger_w
    tl = R_new_world.T @ thumb_w
    pl = R_new_world.T @ z_w
    print(f"{side}:")
    print(f'  NEW quat = "{q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}"')
    print(f"  check finger_local={np.round(fl,2)} thumb_local={np.round(tl,2)} palm_local={np.round(pl,2)}")
