"""两条根位姿路线的横向对比 —— 逐帧生成模型 vs 静态网格搜索。

## 在比什么

同一段片段、同一套 IK、同一个 ``converged`` 判据下，两种根位姿求法各自能让
多少帧的手腕 IK 解出来：

* ``neural``  —— 现有路线。``NeuralRootFrameEstimator`` 逐窗口估计 → best-of-N 取锚点
  → 向锚点混合 → 插值到逐帧 → 平滑。输出**逐帧** ``(R_t, t_t)``。
* ``grid``    —— 新增路线。Qwen-RobotManip 公式 (3)：在轨迹质心周围撒网格，取使
  **关键帧** IK 可行率最大的那一个位姿。输出**整条轨迹一个常量** ``(R, t)``。

这两者不是同一件事的两种算法，是两种**粒度**（逐帧解 / 静态解），所以对比表里
要并列成两行，而不是二选一。静态解的先天劣势是没法跟着人走动，先天优势是不需要
训练、不需要 checkpoint、结果确定（同一输入跑两次完全一样）。

## 口径统一在哪几处（不然数字不可比）

1. **打分用的 IK 和判据**：网格搜索给候选打分用的是 ``opt.ik_left/ik_right``，和上游
   ``select_best_anchor`` 给聚类中心打分用的**完全同一对求解器**，判据也是同一个
   ``info["converged"] = (pos_err < tol_pos) & (ori_err < tol_ori)``。没有自己另立标准。
2. **最终评测的 IK**：两条路线算完根位姿后，都用 ``opt.ik_*_traj``（null-space 关掉的
   那对，也就是流水线真正出轨迹用的那对）在**全部帧**上跑一遍，报同一个
   ``ik_rate = (conv_l.sum() + conv_r.sum()) / (2T)``，和 ``test.py`` 打印的那个数同源。
3. **输入完全相同**：不开坏帧兜底、不做 bilateral 缩放（M7 的 CONFIG 里没这两项），
   两条路线吃同一份 ``left_traj/right_traj``。
4. ``workspace_center=None``。M7 的 CONFIG 本来就没有这个键（``.get()`` 返回 None），
   所以这不是为了对比而改的设置，而是 M7 现在跑的就是这样。**这一点很要紧**：
   只要传了 workspace_center，位置会被 ``pos - pos.mean(0) + center`` 重新居中，
   候选平移 t 被整个抹掉，网格搜索会退化成空操作（见 ``root_grid.py`` docstring）。

## 跑法

    scripts/dev/m7_tool.sh compare_root_pose_solvers.py \
        --ckpt runs/m7/taskspace_v2/checkpoints/final.pt --device cuda:2 --seed 0

    # 只跑官方片段（ours_* 是我方自采的，出片不用它们）
    scripts/dev/m7_tool.sh compare_root_pose_solvers.py --official-only

产物落 ``outputs/dev/root_pose_compare/``：``results.json`` + ``table.md``。
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "external/EgoInfinity/retarget"))

from models.root_opt import RootPoseOptimizer                                # noqa: E402
from models.vn_transformer import NeuralRootFrameEstimator                   # noqa: E402
from sim.robots import ROBOT_CONFIGS as _ROBOT_CONFIGS                       # noqa: E402
from utils.clip_io import SamplesSequence                                    # noqa: E402
from utils.pose_utils import (blend_keyframes, cam_to_root_targets,          # noqa: E402
                              estimate_root_poses, interpolate_root_frames,
                              select_best_anchor, smooth_root_frames)

from web2robot.paths import P                                                # noqa: E402
from web2robot.retarget import (gravity_yaw_candidates, make_keyframe_scorer,  # noqa: E402
                                sample_best_anchor, select_extremal_keyframes,
                                solve_root_pose_grid)

# M7 臂展，scripts/dev/measure_m7_reach.py 实测（seed=0，20 万次采样 + 128 个限位角点）。
# 不写死在 root_grid.py 里：那是 M7 这台机器人的属性，不是搜索算法的属性。
R_MAX_M7 = 1.007


# ── 上游流程的两个可复用片段 ───────────────────────────────────────────────────

def load_model(ckpt_path: Path, device: torch.device):
    """照抄 ``test.py:_load_model`` —— 基线必须用和流水线一模一样的加载方式。"""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    saved = ckpt.get("args", {})
    model = NeuralRootFrameEstimator(
        d_model=saved.get("d_model", 64), num_heads=saved.get("num_heads", 4),
        num_layers=saved.get("num_layers", 4),
        dim_feedforward=saved.get("dim_feedforward", 128), dropout=0.0,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt.get("epoch", "?")


def eval_full_trajectory(R_pf, t_pf, left_cam, right_cam, opt) -> dict:
    """给定逐帧根位姿，用流水线出轨迹那对 IK 在**全部帧**上评一遍。

    这是两条路线唯一的公共评测口径。静态解把常量位姿广播成逐帧再进来，所以
    两边走的是同一段代码、同一个求解器、同一个 ``converged`` 判据。
    """
    lp, lq, rp, rq = cam_to_root_targets(left_cam, right_cam, R_pf, t_pf, None)
    out = {}
    with torch.no_grad():
        _, info_l = opt.ik_left_traj.solve_batch(
            torch.tensor(lp, dtype=torch.float32), torch.tensor(lq, dtype=torch.float32))
        _, info_r = opt.ik_right_traj.solve_batch(
            torch.tensor(rp, dtype=torch.float32), torch.tensor(rq, dtype=torch.float32))
    conv_l = info_l["converged"].cpu().numpy().astype(bool)
    conv_r = info_r["converged"].cpu().numpy().astype(bool)
    T = len(conv_l)
    out["ik_rate"]  = float((conv_l.sum() + conv_r.sum()) / (2 * T))
    out["ik_left"]  = float(conv_l.mean())
    out["ik_right"] = float(conv_r.mean())
    out["pos_err_mean"] = float(np.concatenate([
        info_l["pos_err"].cpu().numpy(), info_r["pos_err"].cpu().numpy()]).mean())
    return out


# ── 路线 A：现有的逐帧生成模型 ─────────────────────────────────────────────────

def solve_neural(model, left_cam, right_cam, g_cam_t, seq_len, seq_fps, opt,
                 device, window_secs=2.0, n_clusters=5, n_samples=1, seed=0,
                 torso_alpha=0.3, torso_alpha_rot=0.7, torso_smooth_sigma=10.0):
    """复刻 ``test.py`` 的 Step 1~5，返回逐帧 ``(R_pf, t_pf)`` 和锚点自己的可行率。

    参数默认值全部取 ``test.py`` 的 argparse 默认值 —— 基线必须是"流水线现在的样子"，
    不是调过参的样子。``n_samples=1`` 也是默认值（单发），和网格搜索的"一次到位"对等。
    """
    stride = max(4, int(round(window_secs * seq_fps)))
    best = sample_best_anchor(
        estimate_fn=lambda: estimate_root_poses(
            model, left_cam, right_cam, g_cam_t, seq_len, seq_fps,
            window_secs, stride, device),
        select_fn=lambda Rs, ts: select_best_anchor(
            Rs, ts, left_cam, right_cam, opt, None, n_clusters=n_clusters),
        n_samples=n_samples, seed=seed, log=lambda *_: None)

    Rs_b, ts_b = blend_keyframes(best.Rs, best.ts, torso_alpha,
                                 alpha_rot=torso_alpha_rot,
                                 R_anchor=best.R_anchor, t_anchor=best.t_anchor)
    if len(best.kf_positions) == seq_len:
        R_pf, t_pf = Rs_b, ts_b
    else:
        R_pf, t_pf = interpolate_root_frames(best.kf_positions, Rs_b, ts_b, seq_len)
    R_pf, t_pf = smooth_root_frames(R_pf, t_pf, sigma=torso_smooth_sigma)
    return R_pf, t_pf, float(best.ik_rate), best.R_anchor, best.t_anchor


# ── 路线 B：静态网格搜索 ───────────────────────────────────────────────────────

# 朝向候选（重力定竖直轴 + 偏航全圆枚举）在 web2robot.retarget.gravity_yaw_candidates，
# 不在这里 —— 它是方法的一部分，得能被单测、能被 test.py 复用。

def make_scorer(left_cam, right_cam, kf_idx, opt):
    """把上游的 ``cam_to_root_targets`` 和两个 ``WristIK`` 注入模块里那份胶水。

    批处理约定（每个候选广播成 |K| 帧、B×|K| 条目标一次 solve_batch）在
    :func:`web2robot.retarget.make_keyframe_scorer` 里，这里只负责"把上游那两样
    东西包成 callable"，并且把 ``workspace_center`` 绑成 ``None``。
    """
    def to_root(left, right, R_pf, t_pf):
        return cam_to_root_targets(left, right, R_pf, t_pf, None)

    iks = {"left": opt.ik_left, "right": opt.ik_right}

    def converged(side, pos, quat):
        with torch.no_grad():
            _, info = iks[side].solve_batch(torch.tensor(pos, dtype=torch.float32),
                                            torch.tensor(quat, dtype=torch.float32))
        return info["converged"].cpu().numpy()

    return make_keyframe_scorer(left_cam[kf_idx], right_cam[kf_idx], to_root, converged)


# ── 每个片段跑一遍 ─────────────────────────────────────────────────────────────

def run_clip(clip: Path, model, device, args) -> dict:
    seq = SamplesSequence(clip)
    T, fps = seq.n_frames, seq.fps
    win = seq.get_window(0, T)
    left_cam, right_cam, g_cam = win["left_traj"], win["right_traj"], win["g_cam"]
    if left_cam is None or right_cam is None:
        return {"clip": clip.name, "skipped": "缺双手跟踪数据"}

    cfg = _ROBOT_CONFIGS["m7"]
    wsc = cfg.get("workspace_center")
    assert wsc is None, ("M7 的 CONFIG 出现了 workspace_center —— 位置会被重新居中，"
                         "网格搜索会退化成空操作，这个对比脚本的前提就不成立了")
    opt = RootPoseOptimizer(model, device=str(device), ik_robot=cfg["ik_robot"],
                            workspace_center=None, start_config=cfg["start_config"],
                            tol_pos=args.tol_pos)

    row = {"clip": clip.name, "n_frames": int(T), "fps": float(fps),
           "official": not clip.name.startswith("ours_")}

    # ── A: 逐帧生成模型 ──
    t0 = time.time()
    g_cam_t = torch.tensor(g_cam, dtype=torch.float32).unsqueeze(0).to(device)
    R_pf, t_pf, anchor_rate, R_anchor, t_anchor = solve_neural(
        model, left_cam, right_cam, g_cam_t, T, fps, opt, device,
        n_samples=args.n_samples, seed=args.seed)
    row["neural"] = {
        "anchor_ik_rate": anchor_rate,
        "t_anchor": np.round(t_anchor, 4).tolist(),
        "t_spread": float(np.linalg.norm(t_pf - t_pf.mean(0), axis=1).max()),
        "secs": round(time.time() - t0, 1),
        **eval_full_trajectory(R_pf, t_pf, left_cam, right_cam, opt),
    }

    # ── B: 静态网格搜索 ──
    # 两个变体：朝向借生成模型的锚点（把差别干净地隔离在"平移怎么定"上），
    # 以及朝向由重力+手的方向直接算（完全不碰 checkpoint，这条才是能进论文
    # "无需训练"那一栏的版本）。
    ee = np.stack([left_cam[:, :3], right_cam[:, :3]], axis=1)        # (T, 2, 3)
    variants = {}
    if args.rotation in ("anchor", "both"):
        variants["grid_anchor"] = [R_anchor]
    if args.rotation in ("gravity", "both"):
        variants["grid_gravity"] = gravity_yaw_candidates(
            -np.asarray(g_cam, dtype=float), left_cam, right_cam, n_yaw=args.yaws)

    # 关键帧只选一次：给候选打分的 scorer 和 solve_root_pose_grid 内部必须是同一组
    # 关键帧。选择函数本身是确定性的（同输入同输出），这里显式算出来传下去，
    # 免得读代码的人以为是两组。
    kf = select_extremal_keyframes(ee, max_keyframes=args.max_keyframes)
    scorer = make_scorer(left_cam, right_cam, kf.indices, opt)
    # chunk 只影响速度，不影响结果（tests/test_root_grid.py 里有一条专门钉这个）。
    # 太小的话 3 万多个候选会被切成几百批，每批只有几千条目标，时间全花在
    # kernel launch 上，GPU 利用率个位数 —— 所以按"单次 solve 的目标条数"倒推。
    chunk = args.chunk or max(1, args.ik_batch // max(1, len(kf.indices) * 2))

    for name, R_cands in variants.items():
        t0 = time.time()
        sol = solve_root_pose_grid(
            ee, scorer,
            r_max=R_MAX_M7, spacing=args.spacing, z_radius=args.z_radius,
            max_keyframes=args.max_keyframes, chunk=chunk,
            R_candidates=R_cands,
            log=(print if args.verbose else (lambda *_: None)))
        assert np.array_equal(sol.keyframes.indices, kf.indices)
        row[name] = {
            "keyframe_ik_rate": sol.ik_rate,
            "n_keyframes": int(len(sol.keyframes.indices)),
            "keyframe_source": sol.keyframes.source,
            "n_candidates": int(sol.n_candidates), "n_scored": int(sol.n_scored),
            "n_rotations": len(R_cands), "best_R_index": sol.best_R_index,
            # 同分候选个数：公式 (3) 的 argmax 常常是个集合（K 只看位置，不看腕部
            # 朝向），这个数越大说明目标函数在这段片段上越不能唯一定解 —— 是要写进
            # 对比表的诊断量，不是内部细节。
            "n_tied": int(sol.n_tied), "tie_break": sol.tie_break,
            # 选中的偏航偏离"人正对着自己双手"多少度 —— 零点的物理含义见
            # gravity_yaw_candidates 的 docstring。
            "yaw_deg": (round(sol.best_R_index * 360.0 / len(R_cands), 1)
                        if name == "grid_gravity" else None),
            "chunk": int(chunk),
            "t": np.round(sol.t, 4).tolist(),
            "R": np.round(sol.R, 4).tolist(),
            "t_offset_vs_anchor": float(np.linalg.norm(sol.t - t_anchor)),
            "secs": round(time.time() - t0, 1),
            **eval_full_trajectory(np.broadcast_to(sol.R, (T, 3, 3)),
                                   np.broadcast_to(sol.t, (T, 3)),
                                   left_cam, right_cam, opt),
        }

    parts = "  ".join(f"{k.replace('grid_', '')} {row[k]['ik_rate']*100:5.1f}%"
                      f"(kf {row[k]['keyframe_ik_rate']*100:.0f} 同分{row[k]['n_tied']})"
                      for k in variants)
    any_v = row[next(iter(variants))]
    print(f"  {clip.name:28s} T={T:4d}  neural {row['neural']['ik_rate']*100:5.1f}%  "
          f"{parts}   (K={any_v['n_keyframes']:2d} {any_v['keyframe_source']}, "
          f"打分 {any_v['n_scored']}/{any_v['n_candidates']}, "
          f"{sum(row[k]['secs'] for k in variants):.0f}s)")
    return row


# ── 报表 ───────────────────────────────────────────────────────────────────────

GRID_LABELS = {"grid_anchor": "grid（朝向借锚点）", "grid_gravity": "grid（重力定向，无模型）"}


def write_table(rows: list, out_dir: Path, meta: dict) -> Path:
    variants = [k for k in ("grid_anchor", "grid_gravity") if any(k in r for r in rows)]
    ok = [r for r in rows if all(v in r for v in variants) and "neural" in r]
    lines = [
        "# 根位姿两条路线对比：逐帧生成模型 vs 静态网格搜索", "",
        f"- 机器人：M7（r_max = {R_MAX_M7:.3f} m，`scripts/dev/measure_m7_reach.py` 实测）",
        f"- 网格：横向半径 {R_MAX_M7:.3f} m，格距 {meta['spacing']} m，"
        f"竖直半径 {meta['z_radius'] if meta['z_radius'] is not None else R_MAX_M7/2:.3f} m",
        f"- checkpoint：`{meta['ckpt']}`（epoch {meta['epoch']}），seed={meta['seed']}，"
        f"n_samples={meta['n_samples']}",
        f"- IK：{meta['device']}，tol_pos={meta['tol_pos']}，判据 = 上游 `info[\"converged\"]`",
        "",
        "`ik_rate` 是**全部帧**上 `(conv_l + conv_r) / 2T`，三行同一个求解器、同一个判据，"
        "就是 `test.py` 打印的那个数。",
        "",
        "两个 grid 变体的区别只在朝向哪来：`朝向借锚点` 用生成模型解出的锚点朝向"
        "（把差别隔离在平移上），`重力定向` 由片段自带的 gravity_up + 双手质心方向"
        "直接算，**完全不碰 checkpoint**。", "",
    ]

    head = "| 片段 | 帧数 | neural |" + "".join(f" {GRID_LABELS[v]} |" for v in variants) \
           + " grid 关键帧 | 打分候选 | grid 用时 |"
    lines += [head, "|---" * (3 + len(variants) + 3) + "|"]
    for r in ok:
        cells = []
        for v in variants:
            d = r[v]["ik_rate"] - r["neural"]["ik_rate"]
            cells.append(f" {r[v]['ik_rate']*100:.1f}% ({d*100:+.1f}) |")
        a = r[variants[0]]
        lines.append(
            f"| `{r['clip']}`{'' if r['official'] else ' (ours)'} | {r['n_frames']} | "
            f"{r['neural']['ik_rate']*100:.1f}% |" + "".join(cells) +
            f" {a['n_keyframes']} ({a['keyframe_source']}) | "
            f"{a['n_scored']}/{a['n_candidates']} | "
            f"{sum(r[v]['secs'] for v in variants):.0f}s |")

    for label, subset in (("官方片段", [r for r in ok if r["official"]]),
                          ("我方自采", [r for r in ok if not r["official"]]),
                          ("全部", ok)):
        if not subset:
            continue
        n = float(np.mean([r["neural"]["ik_rate"] for r in subset]))
        cells = []
        for v in variants:
            g = float(np.mean([r[v]["ik_rate"] for r in subset]))
            cells.append(f" **{g*100:.1f}% ({(g-n)*100:+.1f})** |")
        lines.append(f"| **{label}均值（{len(subset)} 段）** | | **{n*100:.1f}%** |"
                     + "".join(cells) + " | | |")

    skipped = [r for r in rows if r not in ok]
    if skipped:
        lines += ["", "跳过的片段：" + "、".join(
            f"`{r['clip']}`（{r.get('skipped', '结果不全')}）" for r in skipped)]

    # ── 目标函数唯一性 ──
    # 公式 (3) 的 argmax 是个**集合**：K 按位置极值选，完全不看腕部朝向，容易的片段
    # 上成千上万个候选都是关键帧 100%。`同分` 就是这个集合的大小；它一大，"关键帧
    # 可行率"和"全部帧可行率"的差就不再是运气问题，而是目标函数没定死解。
    # 本模块的 tie_break 在同分集合内取最内部点（仍是精确 argmax 的成员）。
    lines += ["", "## 目标函数唯一性（公式 3 的 argmax 是不是唯一）", "",
              "`kf` = 关键帧 K 上的可行率（就是被最大化的那个目标），`全部帧` = 同一个"
              "解在所有帧上的可行率，`同分` = 和最优解**得分完全相同**的候选个数。"
              "同分数大而两列差得多，说明 K（只覆盖位置极值、不看腕部朝向）没能把解定死"
              "—— 这时选哪个同分成员就决定了成败，本模块取同分集合的最内部点。", ""]
    head2 = "| 片段 |" + "".join(f" {GRID_LABELS[v]}：kf / 全部帧 / 同分 |" for v in variants)
    lines += [head2, "|---" * (1 + len(variants)) + "|"]
    for r in ok:
        cells = [f" {r[v]['keyframe_ik_rate']*100:.1f}% / {r[v]['ik_rate']*100:.1f}% / "
                 f"{r[v]['n_tied']} |" for v in variants]
        lines.append(f"| `{r['clip']}` |" + "".join(cells))

    path = out_dir / "table.md"
    path.write_text("\n".join(lines) + "\n")
    return path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--examples", default="examples", help="片段目录（相对上游 retarget/）")
    p.add_argument("--official-only", action="store_true",
                   help="只跑官方片段，跳过 ours_*")
    p.add_argument("--clips", nargs="*", default=None, help="只跑这几段（片段名）")
    p.add_argument("--ckpt", default="runs/m7/taskspace_v2/checkpoints/final.pt")
    p.add_argument("--device", default="cuda:2" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n_samples", type=int, default=1, help="生成模型抽几次（流水线默认 1）")
    p.add_argument("--spacing", type=float, default=0.05, help="网格格距，米")
    p.add_argument("--rotation", choices=("anchor", "gravity", "both"), default="both",
                   help="grid 变体的朝向来源：借生成模型锚点 / 重力+手方向算 / 两个都跑")
    p.add_argument("--yaws", type=int, default=12,
                   help="重力定向下把偏航一圈枚举成几个候选（步长 360/N）。1=只用启发式零点")
    p.add_argument("--z-radius", type=float, default=None,
                   help="竖直搜索半径，米。默认 r_max/2")
    p.add_argument("--max-keyframes", type=int, default=None,
                   help="关键帧上限。默认不截断（凸包顶点全要）")
    p.add_argument("--chunk", type=int, default=0,
                   help="每批送进 IK 的候选数。0=自动，按 --ik-batch 除以关键帧数算")
    p.add_argument("--ik-batch", type=int, default=262144,
                   help="自动 chunk 时，单次 solve_batch 的目标条数预算。"
                        "实测吞吐 4k→7k/s、16k→26k/s、64k→61k/s、262k→72k/s（显存 1.3 GB），"
                        "小批完全被 kernel launch 吃掉，所以默认开到饱和区")
    p.add_argument("--tol_pos", type=float, default=0.01, help="同 test.py 默认值")
    p.add_argument("--out", default=None)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    device = torch.device(args.device)
    ckpt = Path(args.ckpt).resolve()
    model, epoch = load_model(ckpt, device)
    print(f"checkpoint: {ckpt}  (epoch {epoch})   device={device}")

    ex = Path(args.examples).resolve()
    clips = sorted(d for d in ex.iterdir() if d.is_dir())
    if args.clips:
        clips = [d for d in clips if d.name in set(args.clips)]
    if args.official_only:
        clips = [d for d in clips if not d.name.startswith("ours_")]
    print(f"片段 {len(clips)} 段：" + "、".join(d.name for d in clips) + "\n")

    out_dir = P.check_output_dir(Path(args.out).resolve() if args.out
                                else P.repo_root / "outputs/dev/root_pose_compare")
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = {"ckpt": str(ckpt), "epoch": epoch, "device": str(device), "seed": args.seed,
            "n_samples": args.n_samples, "spacing": args.spacing,
            "z_radius": args.z_radius, "tol_pos": args.tol_pos, "r_max": R_MAX_M7,
            "rotation": args.rotation, "yaws": args.yaws}

    rows = []
    for clip in clips:
        try:
            rows.append(run_clip(clip, model, device, args))
        except Exception as e:                                  # noqa: BLE001
            print(f"  {clip.name:28s} 失败：{type(e).__name__}: {e}")
            rows.append({"clip": clip.name, "skipped": f"{type(e).__name__}: {e}"})
        # 每段都落盘：11 段 × 两个变体的整轮跑起来是几十分钟量级，中途被挤掉的话
        # 已经跑完的片段不该跟着没了。
        (out_dir / "results.json").write_text(
            json.dumps({"meta": meta, "rows": rows}, indent=2, ensure_ascii=False))
        write_table(rows, out_dir, meta)

    print(f"\n{out_dir / 'table.md'}")
    print(out_dir / "results.json")


if __name__ == "__main__":
    main()
