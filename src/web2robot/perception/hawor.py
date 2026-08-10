"""HaWoR 前端 —— 相机运动时的手部感知（第③步的两条路线之一）。

HaWoR 出的是**世界系**手部 MANO 参数 + 一条 SLAM 轨迹；下游重定向要的是**相机系**
关节点。这个模块负责这一段换算，以及从 HaWoR 的产物目录里把东西读出来。

为什么走 HaWoR：单目深度是这条链路上最硬的瓶颈，WiLoR+MoGe 在 HO-3D 上手腕深度误差
约 11 cm、深度跟随是负相关；换 HaWoR（用它自己的标尺）后降到 0.6 cm、跟随转正。
所以**相机在动**的片段走这条路，相机固定的片段走 WiLoR+MoGe（见
:mod:`web2robot.perception.wilor`）。路线选择是第②步的事。

## 依赖是注入进来的

``run_mano`` / ``run_mano_left`` / ``load_slam_cam`` 都是 HaWoR 仓库里的函数，只在
``envs/hawor_env`` 里存在。这个模块**不在 import 时拉它们** —— 换算本身
（:func:`world_to_camera`、:func:`aperture`）是纯 numpy，能被单测直接钉住，而需要
HaWoR 的部分收在 :func:`load_hawor_dir` / :func:`mano_joints_camera` 两个函数里，
runner 由调用方传进来。这样"坐标变换写错了"这类错在单测里就能抓到，不用开 GPU。

## 坐标变换

SLAM 给的是 world→cam：``J_cam = R_w2c @ J_world + t_w2c``，逐帧一个 R/t。
写成 einsum 是因为要对 (T, 21, 3) 一次做完：

    J_cam[t, k, i] = R_w2c[t, i, j] * J_world[t, k, j] + t_w2c[t, i]

**转置方向是这里最容易错的一处**，而错了不会报错：R 转置相当于用了逆旋转，手会跑到
相机后面去，深度全是负的但流水线照样往下跑。所以 :func:`world_to_camera` 有单测，
且这里记一句：``"tij,tkj->tki"`` —— j 是被求和的那一维，它在 R 里是**列**（第 2 维）。
"""
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from web2robot.perception.to_clip import (
    DEFAULT_FOCAL, HAND_LEFT, HAND_RIGHT, MANO_INDEX_TIP, MANO_THUMB_TIP, MANO_WRIST,
    empty_joints,
)

#: HaWoR ``world_space_res.pth`` 里 5 个张量的第 0 维就是手，左 0 右 1，
#: 和 clip 格式的槽位约定正好一致（不是巧合，是当初照着对齐的）。
HAWOR_HAND_INDEX = {"left": HAND_LEFT, "right": HAND_RIGHT}


def world_to_camera(joints_world: np.ndarray,
                    R_w2c: np.ndarray,
                    t_w2c: np.ndarray) -> np.ndarray:
    """``(T, K, 3)`` 世界系 → 相机系。逐帧一个 ``R_w2c`` ``(T, 3, 3)`` / ``t_w2c`` ``(T, 3)``。

    纯 numpy，没有副作用 —— 这是整条 HaWoR 链路里唯一真正"能算错"的一步，所以单独
    拎成函数并且有单测（见 ``tests/test_perception_modules.py``）。
    """
    joints_world = np.asarray(joints_world)
    R_w2c, t_w2c = np.asarray(R_w2c), np.asarray(t_w2c)
    if joints_world.ndim != 3 or joints_world.shape[-1] != 3:
        raise ValueError(f"joints_world 要 (T, K, 3)，收到 {joints_world.shape}")
    T = len(joints_world)
    if R_w2c.shape != (T, 3, 3) or t_w2c.shape != (T, 3):
        raise ValueError(f"位姿帧数/形状不匹配：joints T={T}，"
                         f"R_w2c={R_w2c.shape}，t_w2c={t_w2c.shape}")
    return np.einsum("tij,tkj->tki", R_w2c, joints_world) + t_w2c[:, None, :]


def aperture(joints: np.ndarray) -> np.ndarray:
    """抓握开合信号：拇指尖到食指尖的距离（米），``(T,)``。

    用它而不是用手指关节角，是因为它**与感知前端无关** —— 换前端后这条曲线还能直接
    对比，而关节角的定义各家不同。夹爪类机器人（DAS Gripper）也只需要这一个标量。
    """
    joints = np.asarray(joints)
    return np.linalg.norm(joints[:, MANO_THUMB_TIP] - joints[:, MANO_INDEX_TIP], axis=-1)


def read_focal(hawor_dir: Path, default: float = DEFAULT_FOCAL) -> float:
    """读 HaWoR 的 ``est_focal.txt``；没有或读不出就返回 ``default``。

    兜底而不是报错：焦距只影响可视化投影，重定向吃的是米制 3D 点，缺焦距不该卡住
    整段转换。但**要打日志**说用了兜底值，别让 600 这个数悄悄进 scene.json。
    """
    f = Path(hawor_dir) / "est_focal.txt"
    if not f.is_file():
        return default
    try:
        return float(f.read_text().split()[0])
    except (ValueError, IndexError):
        return default


def slam_path(hawor_dir: Path, n_frames: int) -> Path:
    """HaWoR SLAM 结果的路径。文件名里带帧数，所以帧数要对得上。"""
    return Path(hawor_dir) / "SLAM" / f"hawor_slam_w_scale_0_{n_frames}.npz"


def load_hawor_dir(hawor_dir: Path, n_frames: int, load_slam_cam: Callable):
    """读一个 HaWoR 产物目录，返回 ``(mano_params, R_w2c, t_w2c)``。

    ``mano_params`` 是 ``(trans, rot, hand_pose, betas, valid)`` 五个 float32 torch
    张量，第 0 维是手（左 0 右 1）。``load_slam_cam`` 由调用方从 HaWoR 仓库传进来。

    这里 import torch/joblib 是**函数内**懒加载：只有真要读 HaWoR 产物时才需要它们，
    而 :func:`world_to_camera` 这类纯换算不该为此拉进 torch。
    """
    import joblib
    import torch

    hawor_dir = Path(hawor_dir)
    res = hawor_dir / "world_space_res.pth"
    if not res.is_file():
        raise FileNotFoundError(f"找不到 {res} —— 这个目录不像 HaWoR 的输出")
    slam = slam_path(hawor_dir, n_frames)
    if not slam.is_file():
        raise FileNotFoundError(
            f"找不到 {slam}\nSLAM 文件名里带帧数，n_frames={n_frames} 可能不对；"
            f"看看 {hawor_dir / 'SLAM'} 里实际有哪些文件")

    mano_params = [torch.tensor(np.array(x)).float() for x in joblib.load(res)]
    R_w2c, t_w2c, _, _ = load_slam_cam(str(slam))
    return mano_params, R_w2c.numpy(), t_w2c.numpy()


def mano_joints_camera(mano_params, hand: str, R_w2c, t_w2c,
                       run_mano: Callable, device: str = "cuda") -> np.ndarray:
    """跑一只手的 MANO，返回相机系关节 ``(T, 21, 3)``，无效帧为 NaN。

    ``run_mano`` 是 HaWoR 的 ``run_mano``（右手）或 ``run_mano_left``（左手）——
    **左右是两个不同的函数**，传错了手型会镜像反，而位置照样合理，画面上只是手背
    朝内，很容易看漏。所以 ``hand`` 和 ``run_mano`` 要成对给，
    :func:`hawor_to_joints` 已经替调用方配好了。
    """
    trans, rot, hpose, betas, valid = mano_params
    hi = HAWOR_HAND_INDEX[hand]
    sl = slice(hi, hi + 1)
    out = run_mano(trans[sl].to(device), rot[sl].to(device), hpose[sl].to(device),
                   betas=betas[sl].to(device))
    joints_world = out["joints"][0].detach().cpu().numpy()          # (T, 21, 3)
    joints_cam = world_to_camera(joints_world, R_w2c, t_w2c)
    # HaWoR 自己的 valid，再叠一层"数值是有限的" —— 它标 valid 的帧偶尔会带 inf，
    # 而 inf 会一路传到 IK 里变成解不出来但不报错。
    ok = (valid[hi].detach().cpu().numpy().astype(bool)
          & np.isfinite(joints_cam).all(axis=(1, 2)))
    out_joints = np.full_like(joints_cam, np.nan)
    out_joints[ok] = joints_cam[ok]
    return out_joints


def hawor_to_joints(hawor_dir: Path, n_frames: int,
                    load_slam_cam: Callable,
                    run_mano: Callable, run_mano_left: Callable,
                    hands=("left", "right"),
                    device: str = "cuda") -> np.ndarray:
    """一个 HaWoR 目录 → clip 格式的 ``(T, 2, 21, 3)`` 相机系关节。

    ``hands`` 可以只给一只（HO-3D 那类单手序列），没给的那只整段留 NaN。
    三个 callable 由调用方从 HaWoR 仓库传进来，见 :mod:`~web2robot.perception.hawor`
    的模块说明。
    """
    runners = {"left": run_mano_left, "right": run_mano}
    mano_params, R_w2c, t_w2c = load_hawor_dir(hawor_dir, n_frames, load_slam_cam)
    T = mano_params[0].shape[1]
    joints = empty_joints(T)
    for hand in hands:
        joints[:, HAWOR_HAND_INDEX[hand]] = mano_joints_camera(
            mano_params, hand, R_w2c, t_w2c, runners[hand], device=device)
    return joints


__all__ = ["HAWOR_HAND_INDEX", "MANO_WRIST", "MANO_THUMB_TIP", "MANO_INDEX_TIP",
           "world_to_camera", "aperture", "read_focal", "slam_path",
           "load_hawor_dir", "mano_joints_camera", "hawor_to_joints"]
