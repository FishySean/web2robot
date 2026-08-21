"""把物体位姿画到片段画面上，供人眼判断。**只画，不判断好坏。**

官方发布里没有 RGB 视频（只有 ``depth.mp4`` / ``mask.mp4`` / ``bg_template.png``），
所以底图用 ``depth.mp4`` —— 上游 ``test.py`` 的 ``input_viz`` 也是用它当底图，
两边看起来一致，好对照。

投影用的内参和上游画图口径一致：``fx = fy = focal``，``cx = w/2``、``cy = h/2``
（实测 ``camera.json`` 里的 cx/cy 就是图像中心，所以这两套是同一个东西）。

关于包围盒：官方**没说** ``object_obb.bin`` 那 8 个角点是哪一帧姿态下的。实测
角点质心离**首帧**位姿最近（6 个物体里 4 个），所以这里按"首帧"处理 —— 先用
``inv(T_0)`` 把角点搬到物体自身坐标系，再用 ``T_t`` 搬回每一帧。这是个**假设**，
写在这里免得以后有人把它当事实。盒子只是给人眼看的参考，不参与任何误差计算。
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

from web2robot.twin.object_pose import ObjectPoseSet, posquat_to_mats

#: 每个物体一个颜色（BGR）。任务物体单独用亮黄，其余按顺序取。
TASK_COLOR = (0, 235, 235)
OTHER_COLORS = ((160, 160, 160), (200, 130, 90), (110, 190, 110),
                (190, 110, 190), (110, 110, 210), (170, 170, 90))


def project(pts_cam: np.ndarray, fx: float, cx: float, cy: float,
            fy: Optional[float] = None):
    """``(N,3)`` 相机系点 → ``(N,2)`` 像素 + ``(N,)`` 有效掩码（z>0 且有限）。"""
    pts = np.asarray(pts_cam, dtype=np.float64).reshape(-1, 3)
    fy = fx if fy is None else fy
    z = pts[:, 2]
    ok = np.isfinite(pts).all(axis=1) & (z > 1e-6)
    uv = np.full((len(pts), 2), np.nan)
    zs = np.where(ok, z, 1.0)
    uv[:, 0] = fx * pts[:, 0] / zs + cx
    uv[:, 1] = fy * pts[:, 1] / zs + cy
    ok &= np.isfinite(uv).all(axis=1)
    return uv, ok


def box_edges(corners: np.ndarray) -> List[tuple]:
    """8 个角点 → 12 条棱，**不依赖角点顺序、也不依赖长宽比**。

    官方没给角点顺序，硬编码 ``(0,1),(1,2)…`` 会画出一团乱线。

    第一版用的是"每个角点连 3 个最近邻"，**画出来是错的**：细长盒子上，薄面的
    对角线比长棱还短（``√(a²+b²) < c``），于是薄面被画成带叉的四边形、长棱一条
    没画 —— 在 ``--oo8_XIuOM_799.5_809.8`` 那根圆柱上肉眼就能看出来。

    现在用方向计数，对任意长宽比都精确：长方体的 28 条连线里，**棱**只有 3 个方向、
    每个方向 4 条；面对角线 6 个方向、每个 2 条；体对角线 4 个方向、每个 1 条。
    所以"成员数正好是 4 的方向类"就是那 3 组棱，一共 12 条。棱和面对角线不可能
    平行（除非盒子退化），所以不会混。

    退化盒（角点重合、共面）凑不出 3 组 4 条时，退回最近邻那套 —— 画得不全好过
    抛异常打断出片。
    """
    c = np.asarray(corners, dtype=np.float64).reshape(8, 3)
    pairs = [(i, j) for i in range(8) for j in range(i + 1, 8)]
    dirs, keep = [], []
    for i, j in pairs:
        v = c[j] - c[i]
        n = np.linalg.norm(v)
        if n < 1e-9 or not np.isfinite(n):
            continue
        u = v / n
        nz = np.flatnonzero(np.abs(u) > 1e-9)
        if len(nz) and u[nz[0]] < 0:        # 方向不分正反，统一符号
            u = -u
        dirs.append(u)
        keep.append((i, j))

    classes: List[list] = []                # [(方向, [棱…]), …]
    reps: List[np.ndarray] = []
    for u, e in zip(dirs, keep):
        for k, r in enumerate(reps):
            if 1.0 - abs(float(u @ r)) < 1e-6:
                classes[k].append(e)
                break
        else:
            reps.append(u)
            classes.append([e])

    edge_classes = [cl for cl in classes if len(cl) == 4]
    if len(edge_classes) == 3:
        return sorted(e for cl in edge_classes for e in cl)

    d = np.linalg.norm(c[:, None, :] - c[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    fallback = set()
    for i in range(8):
        for j in np.argsort(d[i])[:3]:
            fallback.add((min(i, int(j)), max(i, int(j))))
    return sorted(fallback)


def draw_pose_axes(img, pose7: np.ndarray, fx: float, cx: float, cy: float,
                   axis_len: float = 0.08, label: str = "", color=TASK_COLOR):
    """在物体原点画 XYZ 三轴（和上游 ``draw_frame`` 同一套配色：X 红 Y 绿 Z 蓝）。"""
    import cv2
    pose7 = np.asarray(pose7, dtype=np.float64).reshape(7)
    if not np.isfinite(pose7).all():
        return False
    T = posquat_to_mats(pose7[None])[0]
    pos, R = T[:3, 3], T[:3, :3]
    pts = np.concatenate([pos[None], (pos[None] + axis_len * R.T)], axis=0)
    uv, ok = project(pts, fx, cx, cy)
    if not ok[0]:
        return False
    o = tuple(np.round(uv[0]).astype(int))
    for i, (col, name) in enumerate(zip([(0, 0, 220), (0, 220, 0), (220, 0, 0)], "XYZ")):
        if ok[i + 1]:
            cv2.arrowedLine(img, o, tuple(np.round(uv[i + 1]).astype(int)),
                            col, 2, tipLength=0.25)
    cv2.circle(img, o, 5, color, -1)
    cv2.circle(img, o, 5, (0, 0, 0), 1)
    if label:
        cv2.putText(img, label, (o[0] + 8, o[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    return True


def draw_box(img, corners_cam: np.ndarray, fx: float, cx: float, cy: float,
             color=TASK_COLOR, thickness: int = 1):
    """把 8 个角点连成线框。有角点投影不出来（z<=0）的那几条棱直接不画。"""
    import cv2
    uv, ok = project(corners_cam, fx, cx, cy)
    for i, j in box_edges(corners_cam):
        if ok[i] and ok[j]:
            cv2.line(img, tuple(np.round(uv[i]).astype(int)),
                     tuple(np.round(uv[j]).astype(int)), color, thickness)


def obb_in_object_frame(obb: np.ndarray, pose0_7: np.ndarray) -> np.ndarray:
    """发布的 OBB 角点（相机系，假设首帧姿态）→ 物体自身坐标系。"""
    T0 = posquat_to_mats(np.asarray(pose0_7, np.float64).reshape(7)[None])[0]
    if not np.isfinite(T0).all():
        return np.asarray(obb, np.float64).reshape(8, 3)
    R, t = T0[:3, :3], T0[:3, 3]
    return (np.asarray(obb, np.float64).reshape(8, 3) - t) @ R


def overlay_object_poses(frames: Sequence[np.ndarray], poses: ObjectPoseSet,
                         axis_len: float = 0.08, show_others: bool = True,
                         trail: int = 30) -> List[np.ndarray]:
    """逐帧画：任务物体的三轴 + 包围盒 + 质心轨迹，其他物体只画一个小三轴。

    ``frames`` 短于片段帧数时按最后一帧补（``depth.mp4`` 解出来偶尔差 1 帧）；
    长了就截断。不可信帧（``valid=False``）画成虚一点的颜色并标 ``?``，因为
    "官方每帧都给位姿"不等于"每帧都可信"，人眼看片子时必须能区分这两件事。
    """
    import cv2
    if not frames:
        return []
    n = poses.n_frames
    cam = poses.camera
    h, w = frames[0].shape[:2]
    fx = cam.focal if cam is not None else float(max(h, w))
    cx, cy = w / 2.0, h / 2.0

    task = poses.task_track if poses.tracks else None
    local_box = None
    if task is not None and task.obb is not None:
        local_box = obb_in_object_frame(task.obb, task.poses[0])

    out = []
    for t in range(n):
        img = np.ascontiguousarray(frames[min(t, len(frames) - 1)]).copy()
        if show_others:
            for k, tr in enumerate(poses.tracks):
                if task is not None and tr.oid == task.oid:
                    continue
                col = OTHER_COLORS[k % len(OTHER_COLORS)]
                draw_pose_axes(img, tr.poses[t], fx, cx, cy, axis_len * 0.45,
                               f"o{tr.oid}", col)
        if task is not None:
            good = bool(task.valid[t])
            col = TASK_COLOR if good else (90, 90, 200)
            if trail > 0:
                lo = max(0, t - trail)
                uv, ok = project(task.poses[lo:t + 1, :3], fx, cx, cy)
                pts = np.round(uv[ok]).astype(int)
                for a, b in zip(pts[:-1], pts[1:]):
                    cv2.line(img, tuple(a), tuple(b), col, 1)
            if local_box is not None:
                T = posquat_to_mats(task.poses[t][None])[0]
                if np.isfinite(T).all():
                    draw_box(img, local_box @ T[:3, :3].T + T[:3, 3], fx, cx, cy, col)
            st = "" if task.state is None else str(task.state[t])
            draw_pose_axes(img, task.poses[t], fx, cx, cy, axis_len,
                           f"obj{task.oid}{'' if good else ' ?'}", col)
            cv2.putText(img, f"{t + 1}/{n}  {poses.source}  obj{task.oid}  {st}",
                        (8, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        out.append(img)
    return out
