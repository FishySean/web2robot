"""物体 6D 位姿的数据结构、几何换算和落盘格式。**不碰文件系统里的片段**。

读片段是 :mod:`web2robot.twin.sources` 的事；这里只管"拿到位姿之后怎么表示、
怎么存、怎么挑出任务物体"，所以能用纯 numpy 数组单测。

## 表示约定

7 维位姿 = ``[x, y, z, qw, qx, qy, qz]``，四元数 **wxyz**。这不是随便定的：上游
``utils/clip_io.py:_joints_to_wrist_pose`` 的手腕 7 维位姿就是 wxyz，MuJoCo 的
``data.xquat`` 也是 wxyz。物体位姿要和手腕位姿放在一起算误差（EgoEngine §3.2.2
的跟踪误差），两边约定必须一致，否则错得很安静。

坐标系是**相机系**（和 ``hand_joints.bin`` 里的手部关键点同系），不是世界系，也不是
机器人根系。原因是官方孪生就存在相机系，而手部轨迹也在相机系 —— 保持同系，下游
要换系时只需一个变换，不用猜谁已经变过了。存的时候把 ``frame`` 字段写进 npz。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

#: 7 维位姿里四元数的分量顺序。改这个等于改磁盘格式，别改。
QUAT_ORDER = "wxyz"

#: 物体状态字符串（官方 ``pose_track.json`` 的 ``state_per_frame`` 取值域）。
#: 带 ``grasped`` 的表示这一帧物体在手里 —— 挑"任务物体"就靠它。
GRASP_STATES = ("grasped_l", "grasped_r", "grasped_both")


@dataclass(frozen=True)
class CameraIntrinsics:
    """针孔内参 + 重力方向。``fx == fy == focal``，官方片段就只给一个焦距。

    ``cx``/``cy`` 官方写的是图像中心（实测 ``camera.json`` 里 cx=w/2、cy=h/2），
    上游 ``test.py`` 画图时也是自己用 ``w/2, h/2`` 算的，两者一致。
    """
    focal: float
    cx: float
    cy: float
    width: int
    height: int
    gravity_up: Optional[np.ndarray] = None

    def as_dict(self) -> Dict[str, float]:
        return {"fx": float(self.focal), "fy": float(self.focal),
                "cx": float(self.cx), "cy": float(self.cy)}


@dataclass
class ObjectTrack:
    """一个物体的整段轨迹。

    Attributes
    ----------
    oid:
        物体 id。官方的 id **不连续**（实测 ``-20k07PjLTA_48.0_52.4`` 是
        0/2/3/4/5，没有 1），所以不能拿数组下标当 id 用。
    poses:
        ``(T, 7)`` float32，相机系，wxyz。
    valid:
        ``(T,)`` bool。官方孪生**每帧都有位姿**（不留 NaN），可信与否靠
        ``signals.json`` 的 ``per_object.<oid>.trust`` 逐帧布尔。没有 signals
        的片段全 True，并在 :class:`ObjectPoseSet.notes` 里记一笔。
    state:
        ``(T,)`` 字符串状态（``static`` / ``grasped_l`` / ``grasped_r`` /
        ``grasped_both`` / ``moving``），可能为 None。
    obb:
        ``(8, 3)`` float32 有向包围盒角点（相机系，首帧姿态下），可能为 None。
        只用来可视化，不参与误差。
    mesh_path:
        ``objects/obj_<oid>.ply`` 的路径字符串，可能为 None。
    scale:
        mesh 到公制的缩放系数。官方发布的 mesh 在**归一化的 canonical 系**里
        （实测顶点包围盒边长 ≈0.99，也就是被塞进单位立方体），公制尺寸要靠
        ``pose_track.json`` 的 ``scale_correction``（实测 0.301）还原：
        ``相机系顶点 = R @ (scale * v) + t``。位姿矩阵本身是**纯刚体**
        （det(R)=1，实测），缩放不在里面 —— 谁要拿 mesh 做渲染或
        FoundationPose，漏了这个系数就会差 3 倍多。
    """
    oid: int
    poses: np.ndarray
    valid: np.ndarray
    state: Optional[np.ndarray] = None
    obb: Optional[np.ndarray] = None
    mesh_path: Optional[str] = None
    scale: float = 1.0

    def __post_init__(self):
        self.poses = np.asarray(self.poses, dtype=np.float32).reshape(-1, 7)
        self.valid = np.asarray(self.valid, dtype=bool).reshape(-1)
        if len(self.valid) != len(self.poses):
            raise ValueError(f"valid 长度 {len(self.valid)} != 位姿帧数 {len(self.poses)}")
        if self.state is not None:
            self.state = np.asarray(self.state, dtype="<U16").reshape(-1)
            if len(self.state) != len(self.poses):
                raise ValueError(
                    f"state 长度 {len(self.state)} != 位姿帧数 {len(self.poses)}")
        if self.obb is not None:
            self.obb = np.asarray(self.obb, dtype=np.float32).reshape(8, 3)

    @property
    def n_frames(self) -> int:
        return len(self.poses)

    @property
    def grasped_frac(self) -> float:
        """在手里的帧占比。没有 state 信息时返回 0.0（而不是猜）。"""
        if self.state is None or self.n_frames == 0:
            return 0.0
        return float(np.isin(self.state, GRASP_STATES).mean())

    @property
    def travel(self) -> float:
        """有效帧上质心走过的路程（米）。挑任务物体时的第二判据。"""
        p = self.poses[self.valid, :3]
        if len(p) < 2:
            return 0.0
        return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())


@dataclass
class ObjectPoseSet:
    """一整段片段的物体位姿集合 + 溯源信息。"""
    tracks: List[ObjectTrack]
    n_frames: int
    fps: float
    frame: str = "camera"
    source: str = "official"
    clip: str = ""
    camera: Optional[CameraIntrinsics] = None
    task_object_id: int = -1
    notes: List[str] = field(default_factory=list)

    def __post_init__(self):
        for tr in self.tracks:
            if tr.n_frames != self.n_frames:
                raise ValueError(
                    f"物体 {tr.oid} 有 {tr.n_frames} 帧，片段是 {self.n_frames} 帧")
        if self.task_object_id < 0 and self.tracks:
            self.task_object_id = select_task_object(self.tracks)

    @property
    def oids(self) -> List[int]:
        return [tr.oid for tr in self.tracks]

    def track(self, oid: int) -> ObjectTrack:
        for tr in self.tracks:
            if tr.oid == oid:
                return tr
        raise KeyError(f"片段里没有物体 id={oid}（有的是 {self.oids}）")

    @property
    def task_track(self) -> ObjectTrack:
        return self.track(self.task_object_id)


# ── 几何换算 ──────────────────────────────────────────────────────────────────

def mats_to_posquat(mats: np.ndarray) -> np.ndarray:
    """``(..., 4, 4)`` 齐次变换 → ``(..., 7)`` ``[pos | quat_wxyz]``。

    自己写而不调 ``scipy.spatial.transform``，是为了两件事：
    (1) 非法/退化的旋转块（全零、含 NaN）要**原样传出 NaN**，scipy 会直接抛异常，
    在批量处理整段轨迹时那等于一帧坏了全段没了；
    (2) 符号统一取 ``qw >= 0``，让整段轨迹的四元数符号不乱跳（和
    :func:`web2robot.trajectory.canonicalize_quats` 同一个用意）。
    """
    mats = np.asarray(mats, dtype=np.float64)
    if mats.shape[-2:] != (4, 4):
        raise ValueError(f"要 (...,4,4)，给的是 {mats.shape}")
    lead = mats.shape[:-2]
    M = mats.reshape(-1, 4, 4)
    out = np.full((len(M), 7), np.nan, dtype=np.float32)
    out[:, :3] = M[:, :3, 3]

    R = M[:, :3, :3]
    with np.errstate(invalid="ignore"):    # 坏帧（NaN/全零）走 det 会刷警告
        det = np.linalg.det(np.where(np.isfinite(R), R, 0.0))
    ok = np.isfinite(R).all(axis=(1, 2)) & (np.abs(det - 1.0) < 1e-3)
    for i in np.flatnonzero(ok):
        out[i, 3:] = _rotmat_to_quat_wxyz(R[i])
    return out.reshape(*lead, 7)


def posquat_to_mats(poses: np.ndarray) -> np.ndarray:
    """``(..., 7)`` → ``(..., 4, 4)``。非法四元数（NaN / 零范数）给出全 NaN 的 3×3。"""
    poses = np.asarray(poses, dtype=np.float64)
    if poses.shape[-1] != 7:
        raise ValueError(f"要 (...,7)，给的是 {poses.shape}")
    lead = poses.shape[:-1]
    P = poses.reshape(-1, 7)
    out = np.tile(np.eye(4, dtype=np.float32), (len(P), 1, 1))
    out[:, :3, 3] = P[:, :3]
    q = P[:, 3:]
    n = np.linalg.norm(q, axis=1)
    bad = ~np.isfinite(q).all(axis=1) | (n < 1e-8)
    q = np.where(bad[:, None], np.array([1.0, 0, 0, 0]), q / np.where(n < 1e-8, 1.0, n)[:, None])
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    out[:, 0, 0] = 1 - 2 * (y * y + z * z); out[:, 0, 1] = 2 * (x * y - w * z); out[:, 0, 2] = 2 * (x * z + w * y)
    out[:, 1, 0] = 2 * (x * y + w * z); out[:, 1, 1] = 1 - 2 * (x * x + z * z); out[:, 1, 2] = 2 * (y * z - w * x)
    out[:, 2, 0] = 2 * (x * z - w * y); out[:, 2, 1] = 2 * (y * z + w * x); out[:, 2, 2] = 1 - 2 * (x * x + y * y)
    out[bad, :3, :3] = np.nan
    out[~np.isfinite(P[:, :3]).all(axis=1), :3, 3] = np.nan
    return out.reshape(*lead, 4, 4)


def _rotmat_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    """Shepperd 分支法。取绝对值最大的分量做除数，避免接近 180° 时数值爆掉。"""
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        q = np.array([0.25 * s, (R[2, 1] - R[1, 2]) / s,
                      (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s])
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        q = np.array([(R[2, 1] - R[1, 2]) / s, 0.25 * s,
                      (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s])
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        q = np.array([(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s,
                      0.25 * s, (R[1, 2] + R[2, 1]) / s])
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        q = np.array([(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s,
                      (R[1, 2] + R[2, 1]) / s, 0.25 * s])
    q /= np.linalg.norm(q) + 1e-12
    return q if q[0] >= 0 else -q


# ── 挑任务物体 ────────────────────────────────────────────────────────────────

def select_task_object(tracks: Sequence[ObjectTrack]) -> int:
    """从一堆物体里挑出"任务物体"的 id。

    EgoEngine §3.1 里任务物体是**人给的**（首帧点 prompt）。官方片段没有这个
    prompt，但给了逐帧状态，所以判据按可靠性从强到弱排：

    1. **在手里的帧占比**最高（``grasped_*``）—— 被操作的那个就是任务物体；
    2. 并列则取**可信帧数**多的（``trust``）；
    3. 再并列取**走得最远**的（静止的桌子、背景墙 travel≈0）；
    4. 还并列就取 id 最小的，保证同一段片段每次挑出来的是同一个（可复现）。

    全静止（一个 ``grasped`` 帧都没有）的片段仍然会返回一个 id —— 这时候第 3 条
    在起作用，调用方要自己看 ``grasped_frac`` 决定信不信。
    """
    if not tracks:
        raise ValueError("没有物体轨迹，挑不出任务物体")
    best = max(tracks, key=lambda tr: (round(tr.grasped_frac, 6),
                                       int(tr.valid.sum()),
                                       round(tr.travel, 6),
                                       -tr.oid))
    return int(best.oid)


# ── 落盘 / 读回 ───────────────────────────────────────────────────────────────

def save_object_poses(path, poses: ObjectPoseSet) -> Path:
    """写 ``object_poses.npz``。键的含义见 :func:`load_object_poses`。

    命名和 ``root_frames.npz`` 一个风格：主数据是定长数组，元信息用 0 维数组塞进
    同一个 npz，不另开 json —— 下游 ``np.load`` 一次拿全。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n_obj = len(poses.tracks)
    T = poses.n_frames
    all_poses = (np.stack([tr.poses for tr in poses.tracks], axis=1)
                 if n_obj else np.zeros((T, 0, 7), np.float32))
    all_valid = (np.stack([tr.valid for tr in poses.tracks], axis=1)
                 if n_obj else np.zeros((T, 0), bool))
    has_state = n_obj > 0 and all(tr.state is not None for tr in poses.tracks)
    all_state = (np.stack([tr.state for tr in poses.tracks], axis=1)
                 if has_state else np.zeros((T, 0), dtype="<U16"))
    obbs = np.stack([tr.obb if tr.obb is not None else np.full((8, 3), np.nan, np.float32)
                     for tr in poses.tracks]) if n_obj else np.zeros((0, 8, 3), np.float32)

    task = poses.task_track if n_obj else None
    cam = poses.camera
    np.savez(
        path,
        # ── 主数据：任务物体（下游最常用的就是这两个）
        object_poses     = (task.poses if task is not None
                            else np.zeros((T, 7), np.float32)),
        object_valid     = (task.valid if task is not None
                            else np.zeros((T,), bool)),
        # ── 全部物体
        object_poses_all = all_poses.astype(np.float32),
        object_valid_all = all_valid,
        object_state_all = all_state,
        object_ids       = np.array(poses.oids, dtype=np.int32),
        object_obb       = obbs.astype(np.float32),
        mesh_paths       = np.array([tr.mesh_path or "" for tr in poses.tracks], dtype="<U256"),
        object_scale     = np.array([tr.scale for tr in poses.tracks], dtype=np.float32),
        # ── 元信息
        task_object_id   = np.int32(poses.task_object_id),
        n_frames         = np.int32(T),
        fps              = np.float32(poses.fps),
        frame            = np.array(poses.frame),
        source           = np.array(poses.source),
        clip             = np.array(poses.clip),
        quat_order       = np.array(QUAT_ORDER),
        notes            = np.array(poses.notes, dtype="<U256"),
        camera_focal     = np.float32(cam.focal if cam else np.nan),
        camera_cx        = np.float32(cam.cx if cam else np.nan),
        camera_cy        = np.float32(cam.cy if cam else np.nan),
        camera_wh        = np.array([cam.width, cam.height] if cam else [0, 0], np.int32),
        gravity_up       = (np.asarray(cam.gravity_up, np.float32)
                            if cam is not None and cam.gravity_up is not None
                            else np.full(3, np.nan, np.float32)),
    )
    return path


def load_object_poses(path) -> ObjectPoseSet:
    """读回 :func:`save_object_poses` 写的 npz。

    这是模块二（分级求解）唯一需要的入口 —— 它不认识 backend，只认这个格式。
    """
    z = np.load(Path(path), allow_pickle=False)
    if str(z["quat_order"]) != QUAT_ORDER:
        raise ValueError(f"四元数约定不是 {QUAT_ORDER}：{z['quat_order']}")
    has_state = z["object_state_all"].shape[1] == z["object_poses_all"].shape[1]
    scales = (z["object_scale"] if "object_scale" in z.files
              else np.ones(len(z["object_ids"]), np.float32))
    tracks = []
    for i, oid in enumerate(z["object_ids"]):
        obb = z["object_obb"][i]
        mp = str(z["mesh_paths"][i])
        tracks.append(ObjectTrack(
            oid=int(oid),
            poses=z["object_poses_all"][:, i],
            valid=z["object_valid_all"][:, i],
            state=z["object_state_all"][:, i] if has_state else None,
            obb=None if not np.isfinite(obb).all() else obb,
            mesh_path=mp or None,
            scale=float(scales[i]),
        ))
    wh = z["camera_wh"]
    g = z["gravity_up"]
    cam = None
    if np.isfinite(z["camera_focal"]):
        cam = CameraIntrinsics(float(z["camera_focal"]), float(z["camera_cx"]),
                              float(z["camera_cy"]), int(wh[0]), int(wh[1]),
                              None if not np.isfinite(g).all() else g)
    return ObjectPoseSet(
        tracks=tracks, n_frames=int(z["n_frames"]), fps=float(z["fps"]),
        frame=str(z["frame"]), source=str(z["source"]), clip=str(z["clip"]),
        camera=cam, task_object_id=int(z["task_object_id"]),
        notes=[str(s) for s in z["notes"]],
    )
