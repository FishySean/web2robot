"""数字孪生 —— 从片段里恢复**物体**的 6D 位姿轨迹。

方法来源是 EgoEngine（arXiv 2606.12604）§3.1 "digital twin"：SAM2 分割（手用手部
关键点当 prompt，任务物体用首帧点 prompt 往后跟）+ FoundationPose（RGBD + 物体
mesh）估计时序一致的 6D 轨迹 :math:`\\{T_o^t\\}`。EgoEngine 的深度来自
FoundationStereo。

这一层的定位和 :mod:`web2robot.retarget` 一样：**放"怎么用"，不放求解器本身**。
物体位姿的来源做成可插拔的 backend（:mod:`~web2robot.twin.sources`），因为我们手上
有两条完全不同的路：

- ``official`` —— 官方 EgoInfinity 片段**已经带**数字孪生了（``object_pose.bin`` /
  ``object_obb.bin`` / ``objects/obj_*.ply`` / ``pose_track.json``），直接读。
  这条现在就能跑，是下游模块（分级求解）的输入。
- ``sam2_foundationpose`` —— EgoEngine 原文那条。**未实现**，卡在两件东西上：
  物体 mesh 和公制深度（我们的素材是单目网络视频，没有 FoundationStereo 那种
  双目深度源）。理由和缺什么都写在 :mod:`~web2robot.twin.sources` 里。

对外只有三个动作：``track_objects`` 拿轨迹、``save_object_poses`` / ``load_object_poses``
落盘和读回。落盘格式（``object_poses.npz``）跟着 ``root_frames.npz`` 的风格走，
放在同一个输出目录里，好让后面的模块直接读。

**零上游 import**：需要上游的东西一律以路径/数组/callable 传进来，所以这一层能纯
numpy 单测，不需要 GPU、不需要 ``external/`` 在位。
"""
from web2robot.twin.object_pose import (
    QUAT_ORDER, CameraIntrinsics, ObjectPoseSet, ObjectTrack,
    load_object_poses, mats_to_posquat, posquat_to_mats, save_object_poses,
    select_task_object,
)
from web2robot.twin.sources import SOURCES, read_official_twin, track_objects

__all__ = ["QUAT_ORDER", "CameraIntrinsics", "ObjectTrack", "ObjectPoseSet",
           "mats_to_posquat", "posquat_to_mats", "select_task_object",
           "save_object_poses", "load_object_poses",
           "track_objects", "read_official_twin", "SOURCES"]
