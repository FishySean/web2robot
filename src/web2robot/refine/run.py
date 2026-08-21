"""把判决串起来：物体位姿 + 手部轨迹 → ``action_refine.json``。

分两条入口，共用同一份逻辑（不许走岔，这是模块一那边定下的规矩）：

* 重定向流水线里 —— ``test.py --action_refine mpc`` 调 :func:`refine_run`，
  它顺手把 ``hand_poses.npz`` 也落盘（IK 目标 + IK 实际，root 系）。
* 事后重判 —— ``python -m web2robot.refine --run <那个目录>`` 读上面两个 npz，
  换 λ / 预算 / 块长再判一遍，**不用重跑 IK**。调阈值的时候差别很大。

写出去的三个文件：

======================  ==========================================================
``action_refine.json``  逐块判决 + 整段摘要，人和程序都能读
``action_refine.npz``   逐帧 e / ep / eR、执行后物体位姿、逐帧抓握掩码
``hand_poses.npz``      ``hand_ref`` / ``hand_ach`` ``(T,2,7)`` + root 系逐帧位姿
======================  ==========================================================
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np

from web2robot.refine.attach import grasp_hands, predict_object_poses
from web2robot.refine.blocks import RefineConfig, format_plans, plan_blocks
from web2robot.refine.score import step_errors
from web2robot.twin.object_pose import load_object_poses


def task_object_arrays(poses) -> tuple:
    """:class:`ObjectPoseSet` → ``(ref (T,7), valid (T,), grasp (T,2))``。

    抓握掩码来自模块一存下来的逐帧状态串（``grasped_l`` / ``grasped_r`` /
    ``grasped_both``）。状态缺失的片段（孪生里没有 signals.json 也没有
    ``state_per_frame``）拿到的是全 False，此时退回"全程都抓着"—— 最悲观的假设，
    宁可多判几块要升级，也别把没量过的块当成好的。
    """
    tr = poses.task_track
    ref = np.asarray(tr.poses, dtype=np.float64)
    valid = np.asarray(tr.valid, dtype=bool)
    if tr.state is None:
        grasp = np.ones((len(ref), 2), dtype=bool)
    else:
        grasp = grasp_hands(list(tr.state))
        if not grasp.any():
            grasp = np.ones((len(ref), 2), dtype=bool)
    return ref, valid, grasp


def refine_run(out_dir, poses, hand_ref: np.ndarray, hand_ach: np.ndarray,
               root_R: Optional[np.ndarray] = None,
               root_t: Optional[np.ndarray] = None,
               cfg: RefineConfig = RefineConfig(),
               requested: str = "none",
               write_hand_poses: bool = True,
               verbose: bool = True) -> dict:
    """打分 + 判决 + 落盘，返回整段摘要 dict。

    ``poses`` 可以是 :class:`~web2robot.twin.object_pose.ObjectPoseSet`，也可以是
    ``object_poses.npz`` 的路径。帧数不一致时直接报错 —— 逐帧对齐错了，误差是假的。
    """
    out_dir = Path(out_dir)
    if not hasattr(poses, "tracks"):
        poses = load_object_poses(poses)
    ref, valid, grasp = task_object_arrays(poses)

    hand_ref = np.asarray(hand_ref, dtype=np.float64)
    hand_ach = np.asarray(hand_ach, dtype=np.float64)
    T = len(ref)
    for name, arr in (("hand_ref", hand_ref), ("hand_ach", hand_ach)):
        if arr.shape != (T, 2, 7):
            raise ValueError(f"{name} 要 ({T},2,7)，收到 {arr.shape} —— "
                             "帧数和物体位姿对不上，误差就是假的")

    ach = predict_object_poses(ref, hand_ref, hand_ach, root_R, root_t, grasp)
    plans, summary = plan_blocks(ref, ach, valid, cfg, requested)
    err = step_errors(ref, ach, valid, cfg.weights)

    summary = dict(summary)
    summary.update({"clip": poses.clip, "task_object_id": int(poses.task_object_id),
                    "source": poses.source, "frame": poses.frame,
                    "refined": False,
                    "refined_note": "第一期只出判决，轨迹仍是 Replay（照抄参考）。"
                                    "落盘的 trajectory.npz 没有被这个模块改过。"})

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "action_refine.json").write_text(json.dumps(
        {"summary": summary, "blocks": [p.as_dict() for p in plans]},
        ensure_ascii=False, indent=2))
    np.savez(out_dir / "action_refine.npz",
             e=err["e"].astype(np.float32), ep=err["ep"].astype(np.float32),
             eR=err["eR"].astype(np.float32),
             object_poses_ref=ref.astype(np.float32),
             object_poses_ach=ach.astype(np.float32),
             object_valid=valid, grasp=grasp,
             block_start=np.array([p.score.start for p in plans], np.int32),
             block_stop=np.array([p.score.stop for p in plans], np.int32),
             block_status=np.array([p.score.status for p in plans]),
             block_mode=np.array([p.mode for p in plans]),
             block_escalate=np.array([p.escalate for p in plans]),
             horizon=np.int32(cfg.horizon),
             per_frame_budget=np.float32(cfg.per_frame_budget),
             lam_p=np.float32(cfg.weights.lam_p), lam_R=np.float32(cfg.weights.lam_R))
    if write_hand_poses:
        kw = {}
        if root_R is not None:
            kw = {"root_R": np.asarray(root_R, np.float32),
                  "root_t": np.asarray(root_t, np.float32)}
        np.savez(out_dir / "hand_poses.npz",
                 hand_ref=hand_ref.astype(np.float32),
                 hand_ach=hand_ach.astype(np.float32), **kw)

    if verbose:
        print(format_plans(plans))
        sc = summary["status_counts"]
        print(f"动作精修判决：{summary['n_blocks']} 块（H={cfg.horizon}）"
              f" ok={sc['ok']} over={sc['over']} unknown={sc['unknown']}"
              f"  需升级 {summary['n_escalate']} 块"
              f"（其中 {summary['n_blocked_by_next']} 块是被下一块拖下来的）")
        if requested != "none" and summary["needs_escalation"]:
            print(f"  ⚠ 请求的是 --action_refine {requested}，但 {requested} 求解器"
                  f"还没实现：落盘的轨迹仍是 Replay，action_refine.json 里"
                  f"refined=false。别把这份产物当成精修过的。")
    return summary
