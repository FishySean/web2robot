"""根位姿两条路线的对比表 —— 补上"画面级"三列：穿躯帧数 / 最深穿透 / ρ̄。

## 为什么要有这一张（和 ``compare_root_pose_solvers.py`` 的分工）

那一张比的是 ``ik_rate``：**手腕位姿够不够得着**。够得着不等于这一帧能用 ——
把底座吸到手边同样能让 IK 全解出来，代价是手臂折起来插进躯干。所以
``ik_rate`` 单独看会把"穿躯换来的高可行率"记成进步。这张表在同一批片段上并列：

* ``ik_rate``   —— 够不够得着（从 ``metrics.npz`` 直接读，和 ``test.py`` 打印的同源）
* ``穿躯帧数`` —— 碰撞过滤器**跑完之后**还有多少帧真穿；判据用 MuJoCo 真实网格
  contacts，不是我方代理几何，也就是说这是对我方后处理的**独立复核**
* ``最深穿透`` —— 同上，最坏那一帧多深 [cm]
* ``ρ̄``        —— 平均臂展利用率 ``‖p_wrist − p_waist_pitch‖ / r_max``。
  Ego2Robot（2608.02580）公式里有一项把 ρ 往 0.65 拽，理由正是"别把底座摆到
  手边"。ρ̄ 明显偏小 = 手臂普遍折着 = 该项在起作用的证据。

三列一起看才有话说：`ik_rate` 高而 `穿躯帧数` 也高的解是**假的赢**。

## 口径

* 输入是 ``run_collcmp.sh`` 的产物（13 段官方片段 × {grid, neural}，两条碰撞过滤
  都开着），所以这里量到的穿透是**残留**，不是原始穿透。
* 逐帧同时设置**手臂 7 轴和手指 12 轴**（``trajectory.npz`` 里都有）。这一点和
  ``audit_mujoco_contacts.py`` 不同 —— 那个脚本只设手臂、手指停在 home 位，
  所以两边的"最深穿透"不该期待逐值相同。contact 按非躯干那一侧的 body 名分成
  ``arm`` / ``hand``（含手指），表里合计，明细进 ``results.json``。
* ``STRUCTURAL``（凸包在静息位就互插的 6 组 body 对）和 ``TORSO_BODIES`` 直接从
  ``audit_mujoco_contacts`` import，**不在这里重抄一份** —— 这两个集合一旦两处
  各写一份就会悄悄漂。
* ``r_max = 1.007 m``，``scripts/dev/measure_m7_reach.py`` 实测，定义是关节限位内
  ``max ‖p_ee‖``（在 IK 链根 ``waist_pitch_link`` 系下），所以 ρ 的分子分母同一个系。

## 跑法

    scripts/dev/m7_tool.sh collcmp_table.py

产物落 ``outputs/dev/collcmp_table/``：``results.json`` + ``table.md``。
只读不改，不碰 ``outputs/retarget/collcmp/`` 里的任何文件。
"""
import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from audit_mujoco_contacts import STRUCTURAL, TORSO_BODIES   # noqa: E402
from web2robot.collision import M7CapsuleModel               # noqa: E402
from web2robot.collision.presets import arm_torso_preset      # noqa: E402
from web2robot.paths import P                                # noqa: E402
from web2robot.robots.m7.env import M7Env                    # noqa: E402

#: M7 臂展，``measure_m7_reach.py`` 实测（和 compare_root_pose_solvers.py 同一个数）
R_MAX_M7 = 1.007
SOLVERS = ("neural", "grid")
#: 官方 examples/ 自带的 5 段，其余 8 段是从官方 HF 数据集新拉的。分开报均值：
#: 前者是论文里出现过的素材，后者是"没调过的新片段"，混在一起会掩盖泛化差距。
EXAMPLES = ("fill_jar", "serve_cake", "sip_coffee", "squeeze_soap",
            "-QALmP1nHtM_678.2_682.2")


def discover(root: Path):
    """``<片段名>_<solver>`` → {片段名: {solver: 目录}}，只收两个 solver 都齐的。"""
    found: dict[str, dict[str, Path]] = {}
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        for s in SOLVERS:
            if d.name.endswith(f"_{s}"):
                found.setdefault(d.name[: -len(s) - 1], {})[s] = d
    return {k: v for k, v in found.items() if len(v) == len(SOLVERS)}


def scan(env, cap, ql, qr, fl, fr, fn_l, fn_r):
    """逐帧扫一遍：MuJoCo 网格穿透深度 + 我方代理有符号距离 + 臂展利用率。

    抽成独立函数是为了让 ``sweep_arm_torso_params.py`` 用**同一套判据**打分 ——
    校准和验收如果各用一份实现，调出来的参数就没法拿这张表验。
    """
    model, data = env.model, env.data
    n = len(ql)

    def bname(g):
        return mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[g])) or "?"

    torso_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "waist_pitch_link")
    ee_bid = {s: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{s}_hand_frame")
              for s in ("left", "right")}

    mj = {"arm": np.zeros(n), "hand": np.zeros(n)}   # 穿透深度 [m]，正数
    ours = np.zeros(n)                               # 有符号距离，负 = 穿透
    ours_side = np.zeros((n, 2))                     # 逐侧有符号距离（left, right）
    rho = np.zeros((n, 2))
    for t in range(n):
        env.set_arm_joints("left", ql[t])
        env.set_arm_joints("right", qr[t])
        env.set_finger_joints(fl[t], fn_l)
        env.set_finger_joints(fr[t], fn_r)          # 内部已 mj_forward
        for i in range(data.ncon):
            c = data.contact[i]
            b1, b2 = bname(int(c.geom1)), bname(int(c.geom2))
            if tuple(sorted((b1, b2))) in STRUCTURAL:
                continue
            pair = {b1, b2}
            if not pair & TORSO_BODIES or pair <= TORSO_BODIES:
                continue
            other = (pair - TORSO_BODIES).pop()
            if not other.startswith(("left", "right")):
                continue
            kind = "hand" if "hand" in other else "arm"
            mj[kind][t] = max(mj[kind][t], -float(c.dist))
        p = cap.arm_torso_penetrations(data, margin=0.0, include_fingers=True)
        ours[t] = min(p["left"], p["right"])
        ours_side[t] = (p["left"], p["right"])
        p_torso = data.xpos[torso_bid]
        for j, s in enumerate(("left", "right")):
            rho[t, j] = np.linalg.norm(data.xpos[ee_bid[s]] - p_torso) / R_MAX_M7
    return {"mj_arm": mj["arm"], "mj_hand": mj["hand"], "ours": ours,
            "ours_side": ours_side, "rho": rho}


def summarize(sc, extra=None):
    """``scan`` 的逐帧数组 → 一行统计。校准和对比表共用，口径只有一份。"""
    mj_arm, mj_hand, ours, rho = sc["mj_arm"], sc["mj_hand"], sc["ours"], sc["rho"]
    worst = np.maximum(mj_arm, mj_hand)
    mesh_bad, ours_bad = worst > 0, ours < 0
    # 校准要的两个分布（单位 cm，都取"我方代理的有符号距离" ``ours``）：
    #   miss_d   —— 网格判穿、我方判没穿的那些帧上，我方还差多少才会报警
    #              （> 0 就是"没看见"，膨胀余量至少要加这么多才追得上）
    #   false_d  —— 我方判穿、网格判没穿的那些帧上，我方报的深度有多深
    #              （放松余量 m 会把 |d| < m 的这些帧全部放行）
    miss = mesh_bad & ~ours_bad
    false_alarm = ours_bad & ~mesh_bad
    pct = lambda a, q: float(np.percentile(a, q)) if len(a) else float("nan")  # noqa: E731
    out = {
        "n_frames": int(len(worst)),
        "pen_frames": int(mesh_bad.sum()),
        "pen_frames_arm": int((mj_arm > 0).sum()),
        "pen_frames_hand": int((mj_hand > 0).sum()),
        "pen_max_cm": float(worst.max() * 100),
        "pen_max_arm_cm": float(mj_arm.max() * 100),
        "pen_max_hand_cm": float(mj_hand.max() * 100),
        "pen_sum_cm": float(worst.sum() * 100),
        "ours_pen_frames": int(ours_bad.sum()),
        "ours_pen_max_cm": float(-ours.min() * 100) if ours_bad.any() else 0.0,
        "n_both": int((mesh_bad & ours_bad).sum()),
        "n_miss": int(miss.sum()),
        "n_false": int(false_alarm.sum()),
        "miss_d_cm": [pct(ours[miss] * 100, q) for q in (50, 90, 100)],
        "false_d_cm": [pct(-ours[false_alarm] * 100, q) for q in (50, 90, 100)],
        "rho_mean": float(rho.mean()),
        "rho_min": float(rho.min()),
        "rho_lt_04_frac": float((rho < 0.4).mean()),
    }
    out.update(extra or {})
    return out


def measure(run_dir: Path, env: M7Env, cap: M7CapsuleModel):
    """一个 run 目录 → 一行统计（外加逐帧数组，主程序会把它摘出去存 npz）。"""
    traj = np.load(run_dir / "trajectory.npz", allow_pickle=True)
    sc = scan(env, cap,
              traj["q_left"].astype(np.float64), traj["q_right"].astype(np.float64),
              traj["q_left_fingers"].astype(np.float64),
              traj["q_right_fingers"].astype(np.float64),
              [str(x) for x in traj["left_finger_joint_names"]],
              [str(x) for x in traj["right_finger_joint_names"]])
    metrics = np.load(run_dir / "metrics.npz", allow_pickle=True)
    out = summarize(sc, {"ik_rate": float(metrics["ik_rate"])})
    out["_frames"] = {"mesh": np.maximum(sc["mj_arm"], sc["mj_hand"]),
                      "ours": sc["ours"], "ours_side": sc["ours_side"]}
    return out


def _mean(rows, key):
    return float(np.mean([r[key] for r in rows])) if rows else float("nan")


def _agg(rows):
    """帧数加权的穿躯占比 + 未加权的 ρ̄ / ik_rate 均值（各按各自的自然口径）。"""
    n = sum(r["n_frames"] for r in rows) or 1
    return {
        "n_clips": len(rows),
        "n_frames": n,
        "ik_rate": _mean(rows, "ik_rate"),
        "pen_frames": sum(r["pen_frames"] for r in rows),
        "pen_frac": sum(r["pen_frames"] for r in rows) / n,
        "pen_max_cm": max([r["pen_max_cm"] for r in rows], default=0.0),
        "rho_mean": _mean(rows, "rho_mean"),
    }


def _cells(r):
    return (f"{r['ik_rate'] * 100:.1f}%", f"{r['pen_frames']}/{r['n_frames']}",
            f"{r['pen_max_cm']:.2f}", f"{r['rho_mean']:.3f}")


def write_table(res: dict, out: Path):
    L = ["# 根位姿两条路线对比（补画面级三列）：穿躯 / 穿透深度 / 臂展利用率", "",
         f"- 素材：`run_collcmp.sh` 的 {len(res)} 段官方片段 × {{neural, grid}}，"
         "两条碰撞过滤（`--arm_torso_collision --dual_hand_collision`）都开着，"
         "所以穿透是**残留**", "",
         "- `穿躯` = 碰撞过滤跑完后仍有真实网格穿透的帧数 / 总帧数，判据是 MuJoCo "
         "mesh contacts（**独立于**我方代理几何），手臂和手指都算，明细见 `results.json`",
         f"- `ρ̄` = 平均 `‖p_wrist − p_waist_pitch‖ / r_max`，r_max = {R_MAX_M7} m。"
         "Ego2Robot 的目标函数里有一项把 ρ 往 0.65 拽 —— ρ̄ 偏小就是手臂普遍折着", "",
         "| 片段 | 帧数 | neural ik | neural 穿躯 | neural 最深cm | neural ρ̄ "
         "| grid ik | grid 穿躯 | grid 最深cm | grid ρ̄ |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for clip in sorted(res):
        v = res[clip]
        tag = "" if clip in EXAMPLES else " ᴴᶠ"
        L.append(f"| `{clip}`{tag} | {v['neural']['n_frames']} | "
                 + " | ".join(_cells(v["neural"]) + _cells(v["grid"])) + " |")
    groups = [("官方 examples 5 段", [c for c in res if c in EXAMPLES]),
              ("HF 新拉 8 段", [c for c in res if c not in EXAMPLES]),
              ("全部", list(res))]
    for name, clips in groups:
        a = {s: _agg([res[c][s] for c in clips]) for s in SOLVERS}
        L.append(f"| **{name}均值** | {a['neural']['n_frames']} | "
                 + " | ".join(f"**{a[s]['ik_rate'] * 100:.1f}%** | "
                              f"**{a[s]['pen_frames']} ({a[s]['pen_frac'] * 100:.1f}%)** | "
                              f"**{a[s]['pen_max_cm']:.2f}** | **{a[s]['rho_mean']:.3f}**"
                              for s in SOLVERS) + " |")
    L += ["", "ᴴᶠ = 从官方 HF 数据集新拉的片段（不是 `examples/` 里那 5 段）。", "",
          "## 我方代理 vs MuJoCo 网格（同一批轨迹上的两个判据）", "",
          "代理几何是过滤器**用来做梯度**的那一套，网格是事后复核。两列差得多说明"
          "代理的保守度没覆盖住真实网格，那是过滤器的调参问题，不是检测漏了。", "",
          "| 片段 | solver | 我方代理 穿透帧/最深cm | MuJoCo网格 穿透帧/最深cm | "
          "其中手指参与 | ρ<0.4 帧占比 |", "|---|---|---|---|---|---|"]
    for clip in sorted(res):
        for s in SOLVERS:
            r = res[clip][s]
            L.append(f"| `{clip}` | {s} | {r['ours_pen_frames']}/{r['ours_pen_max_cm']:.2f} "
                     f"| {r['pen_frames']}/{r['pen_max_cm']:.2f} "
                     f"| {r['pen_frames_hand']} 帧, 最深 {r['pen_max_hand_cm']:.2f} cm "
                     f"| {r['rho_lt_04_frac'] * 100:.1f}% |")
    L += ["", "## 代理几何 vs 真实网格：逐帧混淆表（决定校准方向的那张）", "",
          "`漏` = 网格判穿而我方代理判没穿的帧数（**危险方向**：过滤器根本没被触发）；",
          "`误` = 我方判穿而网格判没穿的帧数（**浪费方向**：修了不该修的，还可能把姿态修歪）。",
          "",
          "`漏时差多少` = 那些漏掉的帧上我方代理的有符号距离（正数 = 还差这么多才报警），"
          "也就是「膨胀余量至少要加到这个数」；`误报多深` = 误报帧上我方报的穿透深度，"
          "也就是「放松余量 m 会把 |d| < m 的误报全放行」。两列的分位数是 p50 / p90 / 最大。",
          "",
          "| 片段 | solver | 都判 | 漏 | 误 | 漏时差多少 cm (p50/p90/max) | 误报多深 cm (p50/p90/max) |",
          "|---|---|---|---|---|---|---|"]
    for clip in sorted(res):
        for s in SOLVERS:
            r = res[clip][s]
            f3 = lambda a: "/".join("—" if np.isnan(x) else f"{x:.2f}" for x in a)  # noqa: E731
            L.append(f"| `{clip}` | {s} | {r['n_both']} | **{r['n_miss']}** | {r['n_false']} "
                     f"| {f3(r['miss_d_cm'])} | {f3(r['false_d_cm'])} |")
    for s in SOLVERS:
        rows = [res[c][s] for c in res]
        L.append(f"| **合计** | **{s}** | **{sum(r['n_both'] for r in rows)}** "
                 f"| **{sum(r['n_miss'] for r in rows)}** "
                 f"| **{sum(r['n_false'] for r in rows)}** | | |")
    (out / "table.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path,
                    default=P.data("outputs") / "retarget" / "collcmp",
                    help="run_collcmp.sh 的产物根目录")
    ap.add_argument("--out", type=Path, default=None,
                    help="默认 outputs/dev/collcmp_table/")
    ap.add_argument("--proxy", choices=("preset", "default"), default="preset",
                    help="漏/误两列拿哪把尺子算代理判据。preset（默认）= 每条路线用"
                         "它自己标定的盒子（和过滤器跑时用的那一个一致）；default = "
                         "两条路线都用类默认盒，用来复现标定前的旧表。"
                         "网格 contact 那几列与本开关无关。")
    args = ap.parse_args()
    if not args.root.is_dir():
        raise SystemExit(f"找不到 {args.root} —— 先跑 scripts/dev/run_collcmp.sh")
    pairs = discover(args.root)
    if not pairs:
        raise SystemExit(f"{args.root} 下没有成对的 <片段>_{{neural,grid}} 目录")
    out = P.check_output_dir(args.out or (P.data("outputs") / "dev" / "collcmp_table"))
    out.mkdir(parents=True, exist_ok=True)

    env = M7Env()
    env.reset()
    # 代理判据按路线取各自标定的盒子 —— grid 的过滤器跑的时候用的就是标定盒，
    # 拿默认盒去算它的漏/误等于"拿 A 尺子调 B 尺子"。
    cap = {s: M7CapsuleModel(
               env.model,
               torso_half=(arm_torso_preset(s).get("torso_half")
                           if args.proxy == "preset" else None))
           for s in SOLVERS}
    res, frames = {}, {}
    for clip, dirs in pairs.items():
        res[clip] = {s: measure(dirs[s], env, cap[s]) for s in SOLVERS}
        for s in SOLVERS:
            f = res[clip][s].pop("_frames")
            for k, v in f.items():
                frames[f"{clip}|{s}|{k}"] = v
        n = res[clip]["neural"]
        g = res[clip]["grid"]
        print(f"  {clip:32s} neural ik={n['ik_rate']*100:5.1f}% "
              f"穿躯 {n['pen_frames']:3d}/{n['n_frames']:3d} 最深 {n['pen_max_cm']:5.2f}cm "
              f"ρ̄={n['rho_mean']:.3f} 漏{n['n_miss']:3d}/误{n['n_false']:3d}"
              f"   |   grid ik={g['ik_rate']*100:5.1f}% "
              f"穿躯 {g['pen_frames']:3d}/{g['n_frames']:3d} 最深 {g['pen_max_cm']:5.2f}cm "
              f"ρ̄={g['rho_mean']:.3f} 漏{g['n_miss']:3d}/误{g['n_false']:3d}")
    np.savez_compressed(out / "per_frame.npz", **frames)
    (out / "results.json").write_text(
        json.dumps({"r_max": R_MAX_M7,
                    # 口径随表存下来：同一个 results.json 既可能是标定前的旧表也可能
                    # 是标定后的，事后靠数字分不出来
                    "root": str(args.root), "proxy": args.proxy,
                    "torso_half": {s: np.asarray(cap[s].TORSO_HALF).tolist()
                                   for s in SOLVERS},
                    "clips": res}, indent=2, ensure_ascii=False),
        encoding="utf-8")
    write_table(res, out)
    print(f"\n写出 {out / 'table.md'} 和 {out / 'results.json'}")


if __name__ == "__main__":
    main()
