"""从"机器人手偏了多少"推"物体被带偏到哪" —— 论文里那一步 rollout 的替代品。

EgoEngine §3.2.2 里的 T̂ot 是**在仿真里执行完这一块动作之后**物体的位姿：物体在
MuJoCo 里、手一碰它就动。我们现在的重定向链子里没有物体（`robot_sim.mp4` 是空场景
里的机器人），所以拿不到那个 rollout。这里用的替代是一条运动学假设：

    **抓住的时候，物体和手是刚连的。** 于是手的位姿偏差 D = T_ach ∘ T_ref⁻¹ 直接
    作用在物体上：T̂o = D ∘ To。没抓住的帧，机器人动不到物体，T̂o = To（误差 0）。

这条假设的已知偏差方向：真实里手偏了物体可能滑动、可能掉、也可能被桌子挡住而**不**
跟着偏，所以这里给出的是"如果没滑"的位姿。它会**低估**失败（滑落根本不体现），
不会凭空造出误差。这一点必须知道，因为判决就是拿它打的分做的。

所以接口留了口子：:func:`predict_object_poses` 只是默认实现，
:func:`~web2robot.refine.blocks.plan_blocks` 收的是**已经算好的** ``achieved``
数组 —— 以后真接上带物体的仿真，换掉这个函数就行，打分和判决一行都不用改。

坐标系：物体位姿在相机系（模块一的 ``frame="camera"``），而 IK 的手部位姿在**逐帧
的 root 系**里。所以手的偏差要先共轭换系再作用到物体上（:func:`conjugate_delta`）。
上游 ``cam_to_root_targets`` 的约定是 ``p_root = R_tᵀ (p_cam − t_t)``，即
``(R_t, t_t)`` 是 root 在相机系里的位姿，换回去就是 ``R_t · p_root + t_t``。
``workspace_center`` 那个平移偏移在 ref 和 ach 里是同一个，作差就抵消了，不用管。
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from web2robot.twin.object_pose import mats_to_posquat, posquat_to_mats

# 模块一写在 object_state_all 里的状态串 → 用哪只手的偏差。
HAND_LEFT, HAND_RIGHT = 0, 1
STATE_TO_HANDS = {
    "grasped_l": (HAND_LEFT,),
    "grasped_r": (HAND_RIGHT,),
    "grasped_both": (HAND_LEFT, HAND_RIGHT),
}


def pose_inverse(poses: np.ndarray) -> np.ndarray:
    """``(T,7)`` 位姿求逆。"""
    M = posquat_to_mats(poses)
    out = np.tile(np.eye(4), (len(M), 1, 1))
    Rt = np.swapaxes(M[:, :3, :3], 1, 2)
    out[:, :3, :3] = Rt
    out[:, :3, 3] = -np.einsum("tij,tj->ti", Rt, M[:, :3, 3])
    return mats_to_posquat(out)


def pose_compose(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """``(T,7)`` ∘ ``(T,7)`` —— 先 b 再 a（矩阵意义上 ``A @ B``）。"""
    A, B = posquat_to_mats(a), posquat_to_mats(b)
    return mats_to_posquat(A @ B)


def pose_delta(ref: np.ndarray, ach: np.ndarray) -> np.ndarray:
    """手的位姿偏差 D = T_ach ∘ T_ref⁻¹（左乘，即"在世界里被搬动了多少"）。

    左乘不是右乘：右乘 ``T_ref⁻¹ ∘ T_ach`` 是"在手自己的局部系里偏了多少"，
    那个量作用不到物体上。
    """
    return pose_compose(ach, pose_inverse(ref))


def conjugate_delta(delta: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """把 root 系里的偏差 D 换到相机系：``D_cam = T ∘ D_root ∘ T⁻¹``。

    ``(R, t)`` 是逐帧的 root 在相机系里的位姿（``root_frames.npz`` 那两个数组）。
    这个变换保住的是**旋转角**，不是平移大小：展开后平移是
    ``R·d + (I − R·R_d·Rᵀ)·t``，只有当偏差是纯平移（``R_d = I``）时 ``t`` 才掉出去。
    这不是 bug —— 绕一个离原点很远的点转一点点，换到别的系里看就是一大段平移。
    判据写成"``D_cam ∘ T = T ∘ D``"（单测里就是这么验的），比"范数不变"靠得住。
    """
    D = posquat_to_mats(delta)
    T = np.tile(np.eye(4), (len(D), 1, 1))
    T[:, :3, :3] = np.asarray(R, dtype=np.float64)
    T[:, :3, 3] = np.asarray(t, dtype=np.float64)
    Ti = np.tile(np.eye(4), (len(D), 1, 1))
    Rt = np.swapaxes(T[:, :3, :3], 1, 2)
    Ti[:, :3, :3] = Rt
    Ti[:, :3, 3] = -np.einsum("tij,tj->ti", Rt, T[:, :3, 3])
    return mats_to_posquat(T @ D @ Ti)


def grasp_hands(states: Sequence[str]) -> np.ndarray:
    """逐帧状态串 → ``(T,2)`` bool，哪只手这帧抓着物体。

    认不出来的状态串（``static`` / ``moving`` / 空串）一律算"没抓" —— 猜一个手位
    比留空危险，留空只是这帧不参与打分。
    """
    out = np.zeros((len(states), 2), dtype=bool)
    for i, s in enumerate(states):
        for h in STATE_TO_HANDS.get(str(s), ()):
            out[i, h] = True
    return out


def predict_object_poses(obj_ref: np.ndarray,
                         hand_ref: np.ndarray,
                         hand_ach: np.ndarray,
                         root_R: Optional[np.ndarray] = None,
                         root_t: Optional[np.ndarray] = None,
                         grasp: Optional[np.ndarray] = None) -> np.ndarray:
    """刚连假设下的"执行后物体位姿" ``(T,7)``。

    Parameters
    ----------
    obj_ref
        ``(T,7)`` 参考物体位姿（相机系，模块一的 ``object_poses``）。
    hand_ref, hand_ach
        ``(T,2,7)`` 两只手的参考（IK 目标）与实际（IK 解 FK 回来）位姿。
        手位 0=左 1=右，和 clip 契约一致。
    root_R, root_t
        ``(T,3,3)`` / ``(T,3)``，逐帧 root 在相机系的位姿。不给就认为手部位姿本来
        就在相机系里（单测里图省事用得上，真实链路里必须给）。
    grasp
        ``(T,2)`` bool，哪只手这帧抓着。不给就当**全程都抓着**（最悲观：所有帧都
        计入误差）。真实链路里传 :func:`grasp_hands` 的结果。

    两只手同时抓（``grasped_both``）时取**偏差大的那只**。理由是判决的代价不对称：
    多升级一次只是多花算力，漏掉一块坏数据是要进训练集的。
    """
    obj_ref = np.asarray(obj_ref, dtype=np.float64)
    hand_ref = np.asarray(hand_ref, dtype=np.float64)
    hand_ach = np.asarray(hand_ach, dtype=np.float64)
    T = len(obj_ref)
    if obj_ref.shape != (T, 7):
        raise ValueError(f"obj_ref 要 (T,7)，收到 {obj_ref.shape}")
    if hand_ref.shape != (T, 2, 7) or hand_ach.shape != (T, 2, 7):
        raise ValueError("hand_ref / hand_ach 都要 (T,2,7)，收到 "
                         f"{hand_ref.shape} / {hand_ach.shape}")
    if grasp is None:
        grasp = np.ones((T, 2), dtype=bool)
    grasp = np.asarray(grasp, dtype=bool)
    if grasp.shape != (T, 2):
        raise ValueError(f"grasp 要 (T,2)，收到 {grasp.shape}")

    deltas = np.stack([pose_delta(hand_ref[:, h], hand_ach[:, h]) for h in (0, 1)], 1)
    if root_R is not None:
        deltas = np.stack([conjugate_delta(deltas[:, h], root_R, root_t)
                           for h in (0, 1)], 1)

    # 偏差大小：平移范数 —— 只用来在两只手之间挑一只，不是打分用的那个 e_t
    mag = np.linalg.norm(deltas[:, :, :3], axis=-1)
    mag = np.where(grasp & np.isfinite(mag), mag, -np.inf)
    pick = np.argmax(mag, axis=1)
    any_grasp = np.isfinite(mag).any(axis=1) & (mag.max(axis=1) > -np.inf)

    chosen = deltas[np.arange(T), pick]
    out = obj_ref.copy()
    if any_grasp.any():
        moved = pose_compose(chosen[any_grasp], obj_ref[any_grasp])
        out[any_grasp] = moved
    return out
