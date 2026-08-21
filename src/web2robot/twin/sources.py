"""物体位姿的来源（backend）。**换方案只在这里加一个函数**，不动别处。

## 为什么是两条路

EgoEngine §3.1 的数字孪生 = FoundationStereo 深度 + SAM2 分割 + FoundationPose
（RGBD + 物体 mesh）→ 6D 轨迹。落到我们手上分成两种情况：

``official``
    官方 EgoInfinity 片段**已经把这一步做完并发布了**（HF
    ``Rice-RobotPI-Lab/egoinfinity``，每段都带 ``object_pose.bin`` /
    ``object_obb.bin`` / ``pose_track.json`` / ``objects/obj_*.ply``）。
    这条直接读文件，零推理、零 GPU，是下游分级求解现在的输入。

``sam2_foundationpose``
    EgoEngine 原文那条，**未实现**。缺的不是代码，是两样输入：

    1. **物体 mesh** —— FoundationPose 是 model-based 的，没有 mesh 跑不了。
       官方片段的 mesh 是上游 SAM3D 那条链路产出的（``sam3d_runner.py``
       → ``sam3_meshes/obj_<oid>.ply``），我们这个 checkout 里没有它的中间产物。
    2. **公制深度** —— EgoEngine 用 FoundationStereo，那是**双目**方法；我们的
       目标素材是单目网络视频，没有第二个视角。官方片段自带 ``depth.mp4`` /
       ``depth.npz``（相对深度已对齐到公制），但那正是我们想绕开的依赖。
       这个问题按任务说明先记下来，不在这一步解决。

所以 ``track_objects(..., source="sam2_foundationpose")`` 会**明确报错**并说清缺
什么，而不是静默退化到 ``official``。

## 官方文件格式（自己解出来的，官方没写文档）

===========================  ==================================================
``object_pose.bin``          float32 ``(T, n_obj, 4, 4)``，**帧在外、物体在内**。
                             齐次变换，相机系，末行恒 ``[0,0,0,1]``。物体顺序 =
                             ``pose_track.json`` 的 key 按整数排序。这个布局是
                             拿 ``pose_track.json`` 里的 ``T_seq`` 逐位比对定
                             下来的：``(n_obj,T,4,4)`` 那种读法差 1.35（错），
                             ``(T,n_obj,4,4)`` 差 0（对）。
``object_obb.bin``           float32 ``(n_obj, 8, 3)``，有向包围盒 8 个角点。
``pose_track.json``          按物体 id（字符串）索引；``state_per_frame`` /
                             ``T_seq`` / ``n_observed`` / ``tracking_status`` /
                             ``mode``（官方跑的是 ``position_first``）。
``objects/obj_<oid>.ply``    物体 mesh。**id 不连续**，实测有 0/2/3/4/5。顶点在
                             **归一化 canonical 系**（包围盒边长 ≈0.99），公制尺寸
                             要乘 ``pose_track.json`` 的 ``scale_correction``
                             （实测 0.301）；位姿矩阵本身是纯刚体，不含缩放。
``signals.json``             ``per_object.<oid>.trust`` 是**逐帧布尔**可信标记，
                             ``state`` 是逐帧状态字符串。
``camera.json``              ``focal`` / ``cx`` / ``cy`` / ``width`` / ``height``
                             / ``gravity_up``。实测 cx=w/2、cy=h/2，和上游
                             ``test.py`` 画图时自己算的那套一致。
===========================  ==================================================
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np

from web2robot.twin.object_pose import (
    CameraIntrinsics, ObjectPoseSet, ObjectTrack, mats_to_posquat,
)

#: 官方孪生至少要有这两个文件才算"这段片段带孪生"
REQUIRED = ("object_pose.bin",)


def read_official_twin(clip_dir, n_frames: Optional[int] = None,
                       fps: Optional[float] = None) -> ObjectPoseSet:
    """读官方片段自带的数字孪生。

    Parameters
    ----------
    clip_dir:
        片段目录（``data/clips_official/<clip>/`` 或 ``examples/<clip>/``）。
    n_frames, fps:
        给了就用给的（调用方通常已经从 ``SamplesSequence`` 拿到了，口径要和
        重定向那边完全一致）；没给就从 ``hand_meta.json`` / ``scene.json`` 读。

    Raises
    ------
    FileNotFoundError
        片段里没有 ``object_pose.bin``。仓库里 ``examples/`` 那 5 段就是这种
        情况（只发布了重定向输入，没发布孪生），要用 ``data/clips_official/``。
    ValueError
        文件长度和 ``帧数 × 物体数 × 16`` 对不上 —— 宁可报错，也不要按错误的
        布局读出一段"看起来没问题"的轨迹。
    """
    clip_dir = Path(clip_dir)
    notes: List[str] = []
    pose_bin = clip_dir / "object_pose.bin"
    if not pose_bin.exists():
        raise FileNotFoundError(
            f"{clip_dir} 里没有 object_pose.bin —— 这段片段没带官方数字孪生。"
            f"仓库自带的 examples/ 只发布了重定向输入；带孪生的片段在 "
            f"data/clips_official/（从 HF Rice-RobotPI-Lab/egoinfinity 拉的）。")

    if n_frames is None:
        n_frames = _read_n_frames(clip_dir)
    if fps is None:
        fps = _read_fps(clip_dir)

    oids, tracks_meta = _read_pose_track(clip_dir)
    raw = np.fromfile(pose_bin, dtype=np.float32)

    if oids is None:                        # 没有 pose_track.json，只能反推物体数
        n_obj, oids = _infer_n_obj(raw.size, n_frames, clip_dir)
        notes.append("没有 pose_track.json，物体 id 按 0..n-1 反推")
    else:
        n_obj = len(oids)
    want = n_frames * n_obj * 16
    if raw.size != want:
        raise ValueError(
            f"object_pose.bin 有 {raw.size} 个 float，按 {n_frames} 帧 × {n_obj} 物体 "
            f"× 16 应该是 {want} 个。布局是 (T, n_obj, 4, 4)，对不上就别猜。")
    mats = raw.reshape(n_frames, n_obj, 4, 4)

    obbs = _read_obb(clip_dir, n_obj)
    trust, states = _read_signals(clip_dir, oids, n_frames, notes)

    tracks = []
    for i, oid in enumerate(oids):
        poses = mats_to_posquat(mats[:, i])
        finite = np.isfinite(poses).all(axis=1)
        st = states.get(oid)
        if st is None:
            st = tracks_meta.get(oid, {}).get("state_per_frame")
            st = np.asarray(st, dtype="<U16") if st is not None else None
        tracks.append(ObjectTrack(
            oid=oid, poses=poses, valid=trust[oid] & finite, state=st,
            obb=None if obbs is None else obbs[i],
            mesh_path=_mesh_path(clip_dir, oid),
            scale=float(tracks_meta.get(oid, {}).get("scale_correction", 1.0) or 1.0),
        ))

    bad = [oid for oid in oids
           if tracks_meta.get(oid, {}).get("tracking_status", "ok") != "ok"]
    if bad:
        notes.append(f"官方标了跟踪不 ok 的物体：{bad}")

    return ObjectPoseSet(
        tracks=tracks, n_frames=n_frames, fps=fps, frame="camera",
        source="official", clip=clip_dir.name, camera=read_camera(clip_dir),
        notes=notes,
    )


def read_sam2_foundationpose_twin(clip_dir, **kwargs) -> ObjectPoseSet:
    """EgoEngine §3.1 原文那条链路。**未实现** —— 缺 mesh 和公制深度。

    故意抛异常而不是退化到 ``official``：静默降级会让"我们自己跑出来的孪生"和
    "官方发布的孪生"在结果里混成一团，之后没法说清哪个数字是谁的。
    """
    raise NotImplementedError(
        "sam2_foundationpose backend 还没实现（EgoEngine §3.1）。缺两样输入：\n"
        "  1) 物体 mesh —— FoundationPose 是 model-based 的，没 mesh 跑不了；\n"
        "     官方 mesh 出自上游 SAM3D 链路（sam3d_runner.py → sam3_meshes/），\n"
        "     我们这个 checkout 没有它的中间产物。\n"
        "  2) 公制深度 —— EgoEngine 用的 FoundationStereo 是双目方法，我们的目标\n"
        "     素材是单目网络视频。这个深度来源问题按任务说明先记下来，不在这步解决。\n"
        "现在能用的是 source='official'（读官方片段自带的孪生）。")


#: backend 注册表。加新方案 = 在这里加一行，别处不用动。
SOURCES: Dict[str, Callable[..., ObjectPoseSet]] = {
    "official": read_official_twin,
    "sam2_foundationpose": read_sam2_foundationpose_twin,
}


def track_objects(clip_dir, source: str = "official", **kwargs) -> ObjectPoseSet:
    """按 backend 名字取物体 6D 轨迹。``--object_tracking on`` 走的就是这个函数。"""
    if source not in SOURCES:
        raise ValueError(f"不认识的物体位姿来源 {source!r}，有的是 {sorted(SOURCES)}")
    return SOURCES[source](clip_dir, **kwargs)


# ── 片段里的零碎文件 ──────────────────────────────────────────────────────────

def read_camera(clip_dir) -> Optional[CameraIntrinsics]:
    """``camera.json`` → 内参。没有这个文件就退回 ``scene.json`` 里的 camera 段。

    退回是有意义的：``scene.json`` 每段都有（重定向就靠它拿焦距），``camera.json``
    只在带孪生的片段里有。退回时 ``width``/``height`` 拿不到，填 0。
    """
    clip_dir = Path(clip_dir)
    p = clip_dir / "camera.json"
    if p.exists():
        c = json.loads(p.read_text())
        g = c.get("gravity_up")
        return CameraIntrinsics(
            focal=float(c["focal"]), cx=float(c["cx"]), cy=float(c["cy"]),
            width=int(c["width"]), height=int(c["height"]),
            gravity_up=None if g is None else np.asarray(g, np.float32))
    s = clip_dir / "scene.json"
    if s.exists():
        c = json.loads(s.read_text()).get("camera", {})
        if "focal" in c:
            g = c.get("gravity_up")
            return CameraIntrinsics(
                focal=float(c["focal"]), cx=0.0, cy=0.0, width=0, height=0,
                gravity_up=None if g is None else np.asarray(g, np.float32))
    return None


def _read_n_frames(clip_dir: Path) -> int:
    meta = clip_dir / "hand_meta.json"
    if meta.exists():
        return int(json.loads(meta.read_text())["n_frames"])
    sig = clip_dir / "signals.json"
    if sig.exists():
        return int(json.loads(sig.read_text())["n_frames"])
    raise FileNotFoundError(f"{clip_dir} 里既没有 hand_meta.json 也没有 signals.json，"
                            f"拿不到帧数")


def _read_fps(clip_dir: Path) -> float:
    s = clip_dir / "scene.json"
    if s.exists():
        return float(json.loads(s.read_text()).get("fps", 20.0))
    return 20.0


def _read_pose_track(clip_dir: Path):
    """→ ``(oids | None, {oid: 那个物体的元信息})``。``T_seq`` 太大，不留在内存里。"""
    p = clip_dir / "pose_track.json"
    if not p.exists():
        return None, {}
    raw = json.loads(p.read_text())
    oids = sorted((int(k) for k in raw), key=int)
    meta = {}
    for k, v in raw.items():
        meta[int(k)] = {kk: vv for kk, vv in v.items() if kk != "T_seq"}
    return oids, meta


def _infer_n_obj(n_float: int, n_frames: int, clip_dir: Path):
    per = n_frames * 16
    if per == 0 or n_float % per:
        raise ValueError(
            f"{clip_dir}: object_pose.bin 有 {n_float} 个 float，除不尽 "
            f"{n_frames} 帧 × 16，没有 pose_track.json 也就没法定物体数")
    n = n_float // per
    return n, list(range(n))


def _read_obb(clip_dir: Path, n_obj: int) -> Optional[np.ndarray]:
    p = clip_dir / "object_obb.bin"
    if not p.exists():
        return None
    raw = np.fromfile(p, dtype=np.float32)
    if raw.size != n_obj * 24:
        return None                     # 只用来画图，对不上就不画，不拦住主流程
    return raw.reshape(n_obj, 8, 3)


def _read_signals(clip_dir: Path, oids, n_frames: int, notes: List[str]):
    """→ ``({oid: (T,) bool 可信}, {oid: (T,) 状态字符串})``。

    没有 ``signals.json`` 的片段全部当可信，并在 notes 里记一笔 —— 下游看
    ``valid`` 全 True 时要知道这是"没信息"而不是"都很好"。
    """
    trust = {oid: np.ones(n_frames, bool) for oid in oids}
    states: Dict[int, np.ndarray] = {}
    p = clip_dir / "signals.json"
    if not p.exists():
        notes.append("没有 signals.json，逐帧 trust 全按可信处理")
        return trust, states
    per = json.loads(p.read_text()).get("per_object", {})
    if not per:
        notes.append("signals.json 里没有 per_object，逐帧 trust 全按可信处理")
        return trust, states
    for oid in oids:
        d = per.get(str(oid))
        if not d:
            continue
        t = d.get("trust")
        if t is not None and len(t) == n_frames:
            trust[oid] = np.asarray(t, dtype=bool)
        s = d.get("state")
        if s is not None and len(s) == n_frames:
            states[oid] = np.asarray(s, dtype="<U16")
    return trust, states


def _mesh_path(clip_dir: Path, oid: int) -> Optional[str]:
    p = clip_dir / "objects" / f"obj_{oid}.ply"
    return str(p) if p.exists() else None
