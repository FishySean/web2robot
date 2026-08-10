"""Re-render M7 validation clips with hand_frame local XYZ axes drawn as 3D arrows.

Axis colors: X=red (thumb side), Y=green (finger dir), Z=blue (palm normal).

这个脚本当初是用来**看出**右手翻了 180° 的：M7 建模时两只手被做成完全一样，而 g1/r2
两台已知正确的机器人都是左右镜像 palm normal 的。修好之后（``fix_m7_handframe.py``
算出正确的镜像 quat 写回 ``m7.xml``）它就变成回归用的眼睛 —— 期望看到的是
**左手蓝箭头朝外、右手蓝箭头朝反方向**，绿箭头两只手同向。
数值版的判据在 ``check_handframe_convention.py``（两只手都验 + 拿 g1/r2 当参照）。

用法（``run_dir`` 是一次 ``s4_retarget.sh`` 的输出目录）::

    scripts/dev/m7_tool.sh render_handframe_axes.py outputs/legacy_runs/runs/m7/validation/fill_jar

2026-08-10 修：这个脚本原本吃**片段名**、自己拼 ``runs/m7/validation/{clip}/``，
而 ``m7_tool.sh`` 会 cd 到上游 retarget/ —— 于是产物直接写进第三方 checkout 里。
上一轮把另外 5 个出片脚本改成 ``run_dir`` + ``--out`` 时漏了这一个（只改了注释），
渲出来的 ``robot_sim_axes_h264.mp4`` 就落在了 ``external/`` 下。现在统一走
``_devcli``，产物目录一律过 ``P.check_output_dir``。
"""
import os

import numpy as np, mujoco, cv2

from _devcli import parser, load_traj
from web2robot.robots.m7.config import CONFIG as M7
from web2robot.robots.m7.env import M7Env


def render_with_axes(d, out_path, L=0.12, H=540, W=960):
    label = out_path.parent.name
    out_path = str(out_path)
    qL, qR = d["q_left"], d["q_right"]
    QLf, QRf = d["q_left_fingers"], d["q_right_fingers"]
    fj = [n.replace("left_", "").replace("_joint", "") for n in d["left_finger_joint_names"]]
    fps = float(d["fps"]); T = len(qL)
    env = M7Env(mjcf_path=M7.get("scene_path_fingers", M7["scene_path"]),
                start_config=M7["start_config"])
    r = mujoco.Renderer(env.model, height=H, width=W)
    cam = mujoco.MjvCamera()
    cam.azimuth = float(M7["cam_azimuth"]); cam.elevation = float(M7["cam_elevation"])
    cam.distance = float(M7["cam_distance"]); cam.lookat[:] = M7["cam_lookat"]
    bid = {s: mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, M7["wrist_body"][s])
           for s in ("left", "right")}
    cols = [(0.9, 0.1, 0.1, 1), (0.1, 0.9, 0.1, 1), (0.1, 0.3, 1.0, 1)]  # X Y Z
    tmp = out_path.replace("_h264.mp4", "_tmp.mp4")
    vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    for t in range(T):
        env.set_arm_joints("left", qL[t]); env.set_arm_joints("right", qR[t])
        env.set_finger_joints(QLf[t], [f"left_{n}_joint" for n in fj])
        env.set_finger_joints(QRf[t], [f"right_{n}_joint" for n in fj])
        mujoco.mj_forward(env.model, env.data)
        r.update_scene(env.data, camera=cam)
        scn = r.scene
        for s in ("left", "right"):
            p = env.data.xpos[bid[s]].copy(); Rm = env.data.xmat[bid[s]].reshape(3, 3)
            for ax in range(3):
                tip = p + L * Rm[:, ax]
                g = scn.geoms[scn.ngeom]
                mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_ARROW,
                                    np.zeros(3), np.zeros(3), np.zeros(9),
                                    np.array(cols[ax], np.float32))
                mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_ARROW, 0.006, p, tip)
                scn.ngeom += 1
        img = r.render()[:, :, ::-1].copy()
        # 白字打在 MuJoCo 的白背景上等于没写（2026-08-10 修）：图例和帧号都先铺一条
        # 黑底，图例每个词再用它对应的箭头颜色，这样不看代码也知道哪根轴是哪根。
        cv2.rectangle(img, (0, 0), (W, 34), (0, 0, 0), -1)
        x = 12
        for word, col in (("X=thumb", cols[0]), ("Y=finger", cols[1]),
                          ("Z=palm-normal", cols[2])):
            bgr = tuple(int(255 * c) for c in col[2::-1])   # RGBA(0-1) → BGR(0-255)
            cv2.putText(img, word, (x, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55, bgr, 2)
            x += cv2.getTextSize(word, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)[0][0] + 14
        cv2.rectangle(img, (0, H - 30), (W, H), (0, 0, 0), -1)
        cv2.putText(img, f"{label} f{t}", (12, H - 9),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        vw.write(img)
    r.close(); vw.release()
    os.system(f"ffmpeg -y -loglevel error -i {tmp} -c:v libx264 -pix_fmt yuv420p {out_path}")
    os.remove(tmp)
    print("WROTE", out_path, f"({T} frames @ {fps:.0f}fps)", flush=True)


if __name__ == "__main__":
    args = parser(__doc__).parse_args()
    tr, OUT = load_traj(args)
    render_with_axes(tr, OUT / "robot_sim_axes_h264.mp4")
