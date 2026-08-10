"""EgoInfinity ``retarget`` 吃的 clip 目录格式 —— 写出这一侧。

这是**下游框架的输入契约**，跟用哪个感知前端无关。HaWoR、WiLoR+MoGe、以后换的任何
前端，最后都要落成这三个文件，所以这一层刻意**零前端依赖**（不 import torch、
不 import joblib、不 import hawor）：纯 numpy + json，能被单测直接钉住。

三个文件：

- ``hand_joints.bin`` —— 裸 float32，形状 ``(T, MAX_HANDS, 21, 3)``，**相机系、米制**。
  没检测到的手/帧写 NaN。上游 ``utils/clip_io.py`` 用 ``np.fromfile`` 按
  ``hand_meta.json`` 里的 ``joints_shape`` reshape，所以两边必须一致 —— 不一致不会
  报错，只会 reshape 出一份错位的轨迹。
- ``hand_meta.json`` —— 帧数、形状、每帧的左右手标记。
- ``scene.json`` —— 相机焦距、重力方向、fps、片段 id。

## 为什么槽位是固定的左0右1

上游按**槽位**取手（``joints[:, 0]`` 当左、``joints[:, 1]`` 当右），``is_right_per_frame``
只是它的自检。所以这里把槽位写死成常量并且每帧都填同样的 ``[False, True]``，
不按"这一帧检测到了几只手"去压缩 —— 压缩过的槽位会让左右手在中途对调，而对调之后
IK 照样能解出来，画面上就是机器人两只手互换了任务，看一眼很难发现。

## 为什么 NaN 而不是 0

0 是一个**合法的相机系坐标**（就在光心上）。用 0 填缺失，下游分不出"手在光心"和
"没检测到"，而 ``trajectory/traj_cleanup.py`` 的空洞判据正是靠 NaN 找洞的。
"""
import json
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

#: 手的槽位。上游按下标取，不是按 ``is_right_per_frame`` 取。
HAND_LEFT = 0
HAND_RIGHT = 1
MAX_HANDS = 2

#: MANO/WiLoR 的 21 点约定（HO-3D 同）。0=腕，4=拇指尖，8=食指尖。
N_JOINTS = 21
MANO_WRIST = 0
MANO_THUMB_TIP = 4
MANO_INDEX_TIP = 8

#: 相机 y 轴朝下 → 世界"上"方向在相机系里是 -y。名义值，不是标定出来的。
DEFAULT_GRAVITY_UP = [0.0, -1.0, 0.0]

#: 拿不到标定焦距时的兜底（像素）。
DEFAULT_FOCAL = 600.0

JOINTS_DTYPE = np.float32


def empty_joints(n_frames: int) -> np.ndarray:
    """全 NaN 的 ``(T, 2, 21, 3)`` float32，给逐手填。"""
    return np.full((n_frames, MAX_HANDS, N_JOINTS, 3), np.nan, dtype=JOINTS_DTYPE)


def write_clip(
    out_dir:  Path,
    joints:   np.ndarray,
    fps:      float,
    focal:    float = DEFAULT_FOCAL,
    gravity_up: Optional[Sequence[float]] = None,
    clip_id:  Optional[str] = None,
) -> dict:
    """把 ``joints`` 写成一个 clip 目录，返回落盘的 ``{"meta":…, "scene":…}``。

    Parameters
    ----------
    joints
        ``(T, 2, 21, 3)``，相机系、米制、缺失为 NaN。dtype 不是 float32 会被转 ——
        但形状不对直接报错：形状错了 ``np.fromfile`` 那头是**静默**错位的。
    clip_id
        默认取 ``out_dir`` 的目录名。

    Raises
    ------
    ValueError
        形状不对，或者整段一只手都没有（写出来下游必然拒，不如现在就说）。
    """
    joints = np.asarray(joints)
    want = (len(joints), MAX_HANDS, N_JOINTS, 3)
    if joints.shape != want:
        raise ValueError(f"joints 形状要 (T, {MAX_HANDS}, {N_JOINTS}, 3)，收到 {joints.shape}")
    if not np.isfinite(joints).any():
        raise ValueError("整段没有任何有效关节（全 NaN）—— 这段不该往下游送")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n_frames = len(joints)

    np.ascontiguousarray(joints, dtype=JOINTS_DTYPE).tofile(out_dir / "hand_joints.bin")

    meta = {
        "n_frames": int(n_frames),
        "max_hands": MAX_HANDS,
        "joints_shape": [int(n_frames), MAX_HANDS, N_JOINTS, 3],
        # 槽位是固定的，所以每帧都一样；见模块 docstring 的"为什么"
        "is_right_per_frame": [[False, True] for _ in range(n_frames)],
    }
    scene = {
        "camera": {
            "focal": float(focal),
            "gravity_up": list(gravity_up if gravity_up is not None else DEFAULT_GRAVITY_UP),
        },
        "fps": float(fps),
        "id": clip_id if clip_id is not None else out_dir.name,
    }
    for name, obj in (("hand_meta.json", meta), ("scene.json", scene)):
        with open(out_dir / name, "w") as f:
            json.dump(obj, f, indent=1)
    return {"meta": meta, "scene": scene}


def valid_frame_counts(joints: np.ndarray) -> dict:
    """每只手有多少帧是有效的。出片前打一行日志用 —— 只看 T 看不出来手丢没丢。"""
    return {"left":  int(np.isfinite(joints[:, HAND_LEFT,  MANO_WRIST, 0]).sum()),
            "right": int(np.isfinite(joints[:, HAND_RIGHT, MANO_WRIST, 0]).sum())}


__all__ = ["HAND_LEFT", "HAND_RIGHT", "MAX_HANDS", "N_JOINTS",
           "MANO_WRIST", "MANO_THUMB_TIP", "MANO_INDEX_TIP",
           "DEFAULT_GRAVITY_UP", "DEFAULT_FOCAL", "JOINTS_DTYPE",
           "empty_joints", "write_clip", "valid_frame_counts"]
