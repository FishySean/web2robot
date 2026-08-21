"""校准 **grid 路线**的臂-躯碰撞代理余量 —— 只动几何余量和触发门槛，不动过滤器逻辑。

``neural`` 那条路线的参数一个都不碰：校准结果落 ``collision/presets.py``，
``neural`` 的预设留空，所以它的行为逐位不变。

## 为什么分两段跑

成本差三个数量级，混在一起扫就是浪费：

* **phase1 —— 代理盒半长 / 指尖半径**：纯几何，和过滤器无关。把每帧的骨段采样点和
  指尖在盒局部系里的坐标缓存下来，之后换一组半长只是几次 numpy 运算，所以上千组能
  穷举。判据是"和 MuJoCo 真实网格判决的一致率"。
* **phase2 —— ``enter_thresh`` / ``w_pen`` / ``max_iter``**：决定**修不修、修多狠**，
  必须真跑一遍过滤器再用网格判据复核，一组配置一段片段几十秒。

分开的真正理由不是省时间，是**别让两个错互相抵消**：盒子决定"看得准不准"，门槛决定
"看见了修不修"。一把梭地扫，"盒子放大 + 门槛提高"这种组合可能和原配置总分持平，但它
是两个方向相反的偏差凑出来的，换一批数据立刻崩。

## 口径

* 判据函数直接 import ``collcmp_table`` 的 ``scan`` / ``summarize`` —— 校准和验收
  **同一套实现**，否则调出来的参数没法拿那张对比表验。
* 素材是**不带碰撞过滤**的 run（`run_collcal_prefilter.sh` 的产物）。用带过滤的产物
  校准会套娃：那些帧已经被旧参数改过了。
* phase2 一定包含"默认参数"这一组，它就是验收要对比的"调参前"。
* ``ee_shift`` 是保真代价（修完之后手腕世界位置相对原轨迹挪了多远）。少一帧穿模但手
  飘了 5 cm 不是进步 —— 手腕目标来自人手，挪动就是失真，所以这一列和穿透列一起看。

## 跑法

    scripts/dev/m7_tool.sh sweep_arm_torso_params.py phase1
    scripts/dev/m7_tool.sh sweep_arm_torso_params.py phase2 --half A B C

产物落 ``outputs/dev/collcal/``。
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from collcmp_table import scan, summarize                    # noqa: E402
from web2robot.collision import ArmTorsoFilter, M7CapsuleModel   # noqa: E402
from web2robot.paths import P                                # noqa: E402
from web2robot.robots.m7.env import M7Env                    # noqa: E402

#: 躯干网格的真实 AABB 半长（waist_pitch_link 上，m7.xml 里量的）。
#: 现行代理是它的 [0.755, 0.794, 0.900] 倍 —— "保守缩小"是当初手挑的，没校准过。
MESH_HALF = np.array([0.139, 0.170, 0.239])
DEFAULT_HALF = M7CapsuleModel.TORSO_HALF.copy()
DEFAULT_TIP_R = M7CapsuleModel.TIP_RADIUS
SIDES = ("left", "right")
PREFILTER = P.repo_root / "outputs" / "dev" / "collcal" / "prefilter"
OUT = P.repo_root / "outputs" / "dev" / "collcal"


# ── 素材 ──────────────────────────────────────────────────────────────────────

def load_clip(run_dir: Path):
    tr = np.load(run_dir / "trajectory.npz", allow_pickle=True)
    ln = [str(x) for x in tr["left_finger_joint_names"]]
    rn = [str(x) for x in tr["right_finger_joint_names"]]
    bare_l = [n[len("left_"):] for n in ln]
    bare_r = [n[len("right_"):] for n in rn]
    # 过滤器的 process() 收的是**不带侧前缀**的关节名（它内部自己加 side_），
    # 而 npz 里存的是带前缀的。两侧脱掉前缀必须一模一样，否则说明我理解错了名字约定。
    assert bare_l == bare_r, (bare_l, bare_r)
    return {
        "name": run_dir.name,
        "ql": tr["q_left"].astype(np.float64), "qr": tr["q_right"].astype(np.float64),
        "fl": tr["q_left_fingers"].astype(np.float64),
        "fr": tr["q_right_fingers"].astype(np.float64),
        "ln": ln, "rn": rn, "bare": bare_l,
        "ik": float(np.load(run_dir / "metrics.npz", allow_pickle=True)["ik_rate"]),
    }


def _set_pose(env, c, ql, qr, t):
    env.set_arm_joints("left", ql[t])
    env.set_arm_joints("right", qr[t])
    env.set_finger_joints(c["fl"][t], c["ln"])
    env.set_finger_joints(c["fr"][t], c["rn"])       # 内部已 mj_forward


# ── phase1：盒半长 / 指尖半径的纯几何标定 ─────────────────────────────────────

def cache_local_points(env, cap, c, n_sample=10):
    """每帧把代理的所有采样点换算到**盒局部系**，缓存下来。

    换一组半长时盒的位姿和这些点的局部坐标都不变（半长不影响运动学），所以缓存之后
    任何一组半长都只是 ``|p| - half`` 几次数组运算 —— 这就是 phase1 能穷举的原因。

    返回 ``pts (T, 2, K, 3)``、``rad (2, K)``、``is_tip (2, K)``，K = 骨段采样点 + 指尖。
    """
    T = len(c["ql"])
    K = len(cap.bones["left"]) * n_sample + len(cap.tips["left"])
    pts = np.zeros((T, 2, K, 3))
    rad = np.zeros((2, K))
    is_tip = np.zeros((2, K), bool)
    ts = np.linspace(0.0, 1.0, n_sample)
    for t in range(T):
        _set_pose(env, c, c["ql"], c["qr"], t)
        center, R = cap._torso_frame(env.data)
        Rt = R.T
        for si, side in enumerate(SIDES):
            k = 0
            for pbid, seg, r in cap.bones[side]:
                a = env.data.xpos[pbid]
                b = a + env.data.xmat[pbid].reshape(3, 3) @ seg
                for u in ts:
                    pts[t, si, k] = Rt @ (a + u * (b - a) - center)
                    rad[si, k] = r
                    k += 1
            for tbid in cap.tips[side]:
                pts[t, si, k] = Rt @ (env.data.xpos[tbid] - center)
                is_tip[si, k] = True
                k += 1
    return pts, rad, is_tip


def proxy_sdf(pts, rad, is_tip, half, tip_r):
    """缓存的局部点 + 一组半长 → **逐帧**有符号距离 (T,)，负 = 穿透。

    和 ``capsule_collision`` 里的标量实现同一个公式（点-盒 SDF 减半径，胶囊按采样点
    取 min），这里只是向量化。**公式必须一致**，不然扫出来的参数拿回去用会变味。
    对两侧一起取 min，和 ``scan`` 里 ``min(p["left"], p["right"])`` 是同一个口径。
    """
    r = np.where(is_tip, tip_r, rad)                     # (2, K)
    d = np.abs(pts) - half                               # (T, 2, K, 3)
    outside = np.linalg.norm(np.maximum(d, 0.0), axis=-1)
    inside = np.minimum(0.0, d.max(axis=-1))
    return ((outside + inside) - r).reshape(len(pts), -1).min(axis=1)


def confuse(ours, mesh_bad, mesh_depth):
    """代理判决 vs 网格判决的混淆统计。``ours`` 是逐帧有符号距离 (T,)。"""
    ours_bad = ours < 0
    miss = mesh_bad & ~ours_bad
    false_alarm = ours_bad & ~mesh_bad
    return {
        "n": int(len(ours)),
        "n_both": int((mesh_bad & ours_bad).sum()),
        "n_miss": int(miss.sum()),
        "n_false": int(false_alarm.sum()),
        # 漏报的严重程度用**网格深度**量（我们瞎掉的那部分到底多深），
        # 不用代理自己的距离 —— 代理在漏报帧上的读数正是不可信的那个东西。
        "miss_mesh_cm": float(mesh_depth[miss].max() * 100) if miss.any() else 0.0,
        "false_d_cm": float(-ours[false_alarm].min() * 100) if false_alarm.any() else 0.0,
        "ours_bad": int(ours_bad.sum()),
        "mesh_bad": int(mesh_bad.sum()),
    }


def auc(score, label):
    """``score`` 把 ``label`` 排得多开（0.5 = 瞎猜，1.0 = 完全分得开）。

    为什么要这个数：漏/误的计数是在 "sdf < 0" 这个**特定阈值**上量的，而阈值本身正是
    phase2 要调的东西（``enter_thresh``）。只看计数会有一大片配置并列同分，破不了同分
    就等于随手挑一个 —— 网格搜根位姿那次的教训就是"同分怎么破"决定了 66.7% 还是 100%。
    AUC 与阈值无关，量的是"这个盒子作为探测器本身好不好"，正好对上两段划分：
    **盒子形状定探测器质量，门槛定工作点**。
    """
    pos, neg = score[label], score[~label]
    if not len(pos) or not len(neg):
        return float("nan")
    from scipy.stats import rankdata
    r = rankdata(np.concatenate([pos, neg]))
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2)
                 / (len(pos) * len(neg)))


def cmd_phase1(args):
    env = M7Env()
    cap = M7CapsuleModel(env.model)
    clips = [load_clip(PREFILTER / n) for n in args.clips]

    cache = []
    for c in clips:
        t0 = time.time()
        sc = scan(env, cap, c["ql"], c["qr"], c["fl"], c["fr"], c["ln"], c["rn"])
        mesh_depth = np.maximum(sc["mj_arm"], sc["mj_hand"])
        pts, rad, is_tip = cache_local_points(env, cap, c)
        cache.append({"name": c["name"], "pts": pts, "rad": rad, "is_tip": is_tip,
                      "mesh_depth": mesh_depth, "mesh_bad": mesh_depth > 0,
                      "mesh_arm": sc["mj_arm"] > 0, "mesh_hand": sc["mj_hand"] > 0})
        print(f"  {c['name']}: {len(c['ql'])} 帧，网格判穿 "
              f"{int((mesh_depth > 0).sum())} 帧（臂 {int((sc['mj_arm'] > 0).sum())} / "
              f"指 {int((sc['mj_hand'] > 0).sum())}），缓存 {time.time() - t0:.0f}s")

    # 逐段缓存好了之后，把所有片段的帧**拼成一条**再打分：AUC 是排序统计量，
    # 逐段算完再平均和拼起来算不是一回事，而且 sip_coffee 只有 2 帧真穿，单独算它的
    # AUC 全靠那 2 帧，噪声大。混淆计数也一起用拼接后的总量，口径统一。
    pool_mesh_bad = np.concatenate([k["mesh_bad"] for k in cache])
    pool_mesh_depth = np.concatenate([k["mesh_depth"] for k in cache])

    def pooled(half, tip_r, off=0.0):
        ours = np.concatenate([proxy_sdf(k["pts"], k["rad"], k["is_tip"], half, tip_r)
                               for k in cache]) + off
        out = confuse(ours, pool_mesh_bad, pool_mesh_depth)
        out["auc"] = auc(-ours, pool_mesh_bad)      # -ours：越大越"像穿透"
        # 真穿的那些帧上代理读数是多少 —— 这直接决定 phase2 的 enter_thresh 该设多少：
        # 门槛必须比这些读数**浅**，否则真穿的帧根本进不了修复分支。
        bad = ours[pool_mesh_bad]
        out["bad_ours_cm"] = [round(float(np.percentile(bad, q) * 100), 2)
                              for q in (10, 50, 90)] if len(bad) else []
        return out

    base = pooled(DEFAULT_HALF, DEFAULT_TIP_R)
    print("\n现行参数（调参前基线），逐段:")
    for k in cache:
        b = confuse(proxy_sdf(k["pts"], k["rad"], k["is_tip"], DEFAULT_HALF,
                              DEFAULT_TIP_R), k["mesh_bad"], k["mesh_depth"])
        # 纯手指穿的帧数单独列：如果漏报主要是这类，病根不在盒子大小，而在代理只有
        # 5 个指尖球、没有手掌和手腕 —— 那是覆盖缺口，放大盒子补不上。
        only_hand = int((k["mesh_hand"] & ~k["mesh_arm"]).sum())
        print(f"  {k['name']:28s} 网格 {b['mesh_bad']:4d} 代理 {b['ours_bad']:4d} "
              f"| 都判穿 {b['n_both']:4d} 漏 {b['n_miss']:3d} 误 {b['n_false']:4d} "
              f"| 漏的最深 {b['miss_mesh_cm']:.2f}cm | 纯手指穿 {only_hand} 帧")
    print(f"  合计 {base['n']} 帧：网格 {base['mesh_bad']} 代理 {base['ours_bad']}，"
          f"漏 {base['n_miss']} 误 {base['n_false']}，"
          f"分歧率 {(base['n_miss'] + base['n_false']) / base['n'] * 100:.1f}%，"
          f"AUC {base['auc']:.4f}")

    # 两条互斥的候选路线，一起扫、让数据挑：
    #   (a) 逐轴缩盒 —— 表达力强，但盒子就不再是"躯干网格的 AABB"了，物理含义变模糊；
    #   (b) 盒子不动，给读数加一个标定偏移 δ —— 只有一个数，含义清楚（"代理比网格早报
    #       δ cm"），而且它是 (a) 的一个子集（各向同性缩放≈常数偏移）。
    # AUC 已经近满，说明形状本来就够用、错的是零点，所以 (b) 很可能够。但这要用数字
    # 证，不是靠感觉 —— 所以两边都扫，最后对比。
    grid = np.round(np.arange(args.lo, args.hi + 1e-9, args.step), 3)
    tips = [DEFAULT_TIP_R] if args.no_tip_sweep else [0.012, 0.020, 0.030]
    rows = []
    for sx in grid:
        for sy in grid:
            for sz in grid:
                half = MESH_HALF * np.array([sx, sy, sz])
                for tr in tips:
                    r = pooled(half, tr)
                    r.update(scale=[float(sx), float(sy), float(sz)],
                             half=half.round(4).tolist(), tip_r=float(tr), offset=0.0,
                             disagree=(r["n_miss"] + r["n_false"]) / r["n"])
                    rows.append(r)
    off_rows = []
    for off in args.offsets:
        r = pooled(DEFAULT_HALF, DEFAULT_TIP_R, off)
        r.update(scale=None, half=DEFAULT_HALF.round(4).tolist(),
                 tip_r=DEFAULT_TIP_R, offset=float(off),
                 disagree=(r["n_miss"] + r["n_false"]) / r["n"])
        off_rows.append(r)
    # 主判据是加权错误数，**同分用 AUC 破**（阈值无关的探测器质量），
    # 再同分挑盒子更大的那个（宁可误报不可漏报，和保守策略一致）。
    rows.sort(key=lambda r: (r["n_miss"] * args.miss_weight + r["n_false"],
                             -r["auc"], -sum(r["scale"])))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "phase1.json").write_text(json.dumps(
        {"baseline": base, "default_half": DEFAULT_HALF.tolist(),
         "mesh_half": MESH_HALF.tolist(), "miss_weight": args.miss_weight,
         "clips": [k["name"] for k in cache],
         "best_auc": max(rows, key=lambda r: r["auc"]),
         "offset_rows": off_rows,
         "rows": rows[:200]}, ensure_ascii=False, indent=1))

    fmt = ("  {half:24s} {tip:.3f} {off:+.3f} {miss:3d} {false:4d} {both:5d} "
           "{dis:5.1f}% {auc:.4f}  {mcm:.2f}  {bad}")
    row_line = lambda r: fmt.format(  # noqa: E731
        half=str(r["half"]), tip=r["tip_r"], off=r["offset"], miss=r["n_miss"],
        false=r["n_false"], both=r["n_both"], dis=r["disagree"] * 100, auc=r["auc"],
        mcm=r["miss_mesh_cm"], bad=r["bad_ours_cm"])
    head = ("  半长[m]                  tip    δ     漏   误  都判穿 分歧率   AUC   "
            "漏最深cm  真穿帧上代理读数 p10/50/90 cm")
    print(f"\n(a) 逐轴缩盒：按 (漏×{args.miss_weight} + 误) 排序、同分按 AUC 破，前 12 组：")
    print(head)
    for r in rows[:12]:
        print(row_line(r))
    print("\n(b) 盒子不动、只加标定偏移 δ（δ 越大越宽松）：")
    print(head)
    for r in off_rows:
        print(row_line(r))
    ba = max(rows, key=lambda r: r["auc"])
    print(f"\nAUC 最高的一组（探测器本身最好，未必在 0 阈值上得分最高 —— 两者分开"
          f"正说明问题在零点不在形状）：{ba['half']} tip={ba['tip_r']:.3f} "
          f"AUC={ba['auc']:.4f} 漏 {ba['n_miss']} 误 {ba['n_false']}")
    print(f"完整前 200 组 → {OUT / 'phase1.json'}")


# ── phase2：门槛 / 权重，真跑过滤器 ───────────────────────────────────────────

def _hand_pos_traj(env, c, ql, qr):
    """逐帧两只手 hand_frame 的世界位置 (T, 2, 3) —— 保真代价的度量基准。"""
    import mujoco
    bid = [mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, f"{s}_hand_frame")
           for s in SIDES]
    out = np.zeros((len(ql), 2, 3))
    for t in range(len(ql)):
        _set_pose(env, c, ql, qr, t)
        out[t] = [env.data.xpos[b] for b in bid]
    return out


def cmd_phase2(args):
    from sim.robots import ROBOT_CONFIGS
    cfg = ROBOT_CONFIGS["m7"]
    env = M7Env()
    clips = [load_clip(PREFILTER / n) for n in args.clips]
    half = np.array(args.half) if args.half else None

    configs = [{"tag": "默认（调参前）"}]
    if half is not None and args.with_box_only:
        configs.append({"tag": "只换盒（门槛不动）", "torso_half": half})
    for et in args.enter_thresh:
        for mg in args.margin:
            for wp in args.w_pen:
                # 标签直接写两个**物理量**（多深才修 / 推出多少余量），而不是两个原始
                # 参数名 —— 报告里要能一眼看出这组配置在行为上意味着什么
                configs.append({"tag": f"修>{(et - mg) * 100:.1f}cm/推出{mg * 100:.1f}cm"
                                       + (f"/wpen{wp}" if wp != 20.0 else ""),
                                "torso_half": half, "enter_thresh": et,
                                "margin": mg, "w_pen": wp})

    results = []
    for cf in configs:
        kw = {k: v for k, v in cf.items() if k != "tag" and v is not None}
        rows = []
        for c in clips:
            t0 = time.time()
            # 判据用的代理必须和过滤器用的**同一组几何**，否则等于拿 A 尺子调 B 尺子
            cap = M7CapsuleModel(env.model, torso_half=kw.get("torso_half"))
            raw_hands = _hand_pos_traj(env, c, c["ql"], c["qr"])
            f = ArmTorsoFilter(cfg, verbose=False, **kw)
            ql, qr = f.process(c["ql"].copy(), c["qr"].copy(),
                               q_left_fingers=c["fl"], q_right_fingers=c["fr"],
                               finger_jnames=c["bare"])
            shift = np.linalg.norm(_hand_pos_traj(env, c, ql, qr) - raw_hands, axis=-1)
            sc = scan(env, cap, ql, qr, c["fl"], c["fr"], c["ln"], c["rn"])
            row = summarize(sc, {"ik_rate": c["ik"], "clip": c["name"],
                                 "ee_shift_mean_cm": float(shift.mean() * 100),
                                 "ee_shift_max_cm": float(shift.max() * 100),
                                 "secs": round(time.time() - t0, 1)})
            rows.append(row)
            print(f"  [{cf['tag']}] {c['name']:28s} 穿躯 {row['pen_frames']:4d}/"
                  f"{row['n_frames']:4d} 最深 {row['pen_max_cm']:5.2f}cm "
                  f"代理判穿 {row['ours_pen_frames']:4d} "
                  f"手腕挪动 均{row['ee_shift_mean_cm']:.2f}/最{row['ee_shift_max_cm']:.2f}cm "
                  f"({row['secs']:.0f}s)")
        n = sum(r["n_frames"] for r in rows)
        results.append({
            "tag": cf["tag"],
            "kwargs": {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                       for k, v in kw.items()},
            "n_frames": n,
            "pen_frames": sum(r["pen_frames"] for r in rows),
            "pen_frac": sum(r["pen_frames"] for r in rows) / n,
            "ours_pen_frames": sum(r["ours_pen_frames"] for r in rows),
            "gap_frames": abs(sum(r["ours_pen_frames"] for r in rows)
                              - sum(r["pen_frames"] for r in rows)),
            "n_miss": sum(r["n_miss"] for r in rows),
            "n_false": sum(r["n_false"] for r in rows),
            "pen_max_cm": max(r["pen_max_cm"] for r in rows),
            "ee_shift_mean_cm": float(np.mean([r["ee_shift_mean_cm"] for r in rows])),
            "ee_shift_max_cm": max(r["ee_shift_max_cm"] for r in rows),
            "rho_mean": float(np.mean([r["rho_mean"] for r in rows])),
            "per_clip": rows,
        })

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "phase2.json").write_text(json.dumps(results, ensure_ascii=False, indent=1))
    print("\n| 配置 | 穿躯帧 | 代理判穿 | 帧数差 | 漏/误 | 最深cm | 手腕挪动 均/最 cm | ρ̄ |")
    print("|---|---|---|---|---|---|---|---|")
    for r in results:
        print(f"| {r['tag']} | {r['pen_frames']}/{r['n_frames']} "
              f"({r['pen_frac'] * 100:.1f}%) | {r['ours_pen_frames']} | "
              f"{r['gap_frames']} | {r['n_miss']}/{r['n_false']} | "
              f"{r['pen_max_cm']:.2f} | {r['ee_shift_mean_cm']:.2f}/"
              f"{r['ee_shift_max_cm']:.2f} | {r['rho_mean']:.3f} |")
    print(f"\n→ {OUT / 'phase2.json'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    # --clips 挂在 parent 上而不是主 parser 上：nargs="+" 紧跟一个位置参数（子命令名）
    # 会把子命令名一起吞掉，写成 `phase1 --clips a b` 才不会歧义
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--clips", nargs="+", default=None,
                       help=f"{PREFILTER} 下的目录名；默认全部。"
                            "名字开头的 '-' 可以省掉（argparse 会当选项）")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("phase1", parents=[common],
                        help="盒半长/指尖半径的纯几何标定（穷举，秒级）")
    p1.add_argument("--lo", type=float, default=0.40, help="半长相对网格 AABB 的最小倍数")
    p1.add_argument("--hi", type=float, default=1.15)
    p1.add_argument("--step", type=float, default=0.05)
    p1.add_argument("--miss_weight", type=float, default=3.0,
                    help="排序时漏报相对误报的权重（漏报是安全问题，误报是保真问题）")
    p1.add_argument("--no_tip_sweep", action="store_true")
    p1.add_argument("--offsets", type=float, nargs="+",
                    default=[0.0, 0.005, 0.01, 0.015, 0.02, 0.03, 0.04],
                    help="路线 (b)：盒子不动，给代理读数加的标定偏移 [m]")
    p1.set_defaults(fn=cmd_phase1)

    p2 = sub.add_parser("phase2", parents=[common],
                        help="门槛/权重，真跑过滤器（分钟级）")
    p2.add_argument("--half", type=float, nargs=3, default=None,
                    help="phase1 选出来的盒半长 [m]；不给就只扫门槛")
    p2.add_argument("--enter_thresh", type=float, nargs="+", default=[0.04, 0.02, 0.01])
    p2.add_argument("--margin", type=float, nargs="+", default=[0.0],
                    help="修好之后要求的余量 [m]；配合 enter_thresh 决定"
                         "「深过 (enter_thresh - margin) 才修，推到富余 margin」")
    p2.add_argument("--with_box_only", action="store_true",
                    help="额外跑一组「只换盒、门槛不动」做消融")
    p2.add_argument("--w_pen", type=float, nargs="+", default=[20.0])
    p2.set_defaults(fn=cmd_phase2)

    args = ap.parse_args()
    have = sorted(d.name for d in PREFILTER.iterdir()
                  if (d / "trajectory.npz").exists())
    if args.clips is None:
        args.clips = have
    else:
        # 官方片段名多是 YouTube id，常以 '-' 开头（-2cNMO9Mm3Q_192.4_209.2），argparse
        # 会当成选项拒收。所以允许写去掉开头 '-' 的名字，这里按后缀补回来。
        args.clips = [n if n in have
                      else next(h for h in have if h.lstrip("-") == n.lstrip("-"))
                      for n in args.clips]
    print(f"素材（不带碰撞过滤）：{', '.join(args.clips)}")
    args.fn(args)


if __name__ == "__main__":
    main()
