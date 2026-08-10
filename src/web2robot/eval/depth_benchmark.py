"""HO-3D 手腕定位评测：从冻好的 npz 算出深度/平面误差与深度跟随。

这是论文里"单目深度瓶颈的解法"那张表的计算口径，从
`hand2robot/scripts/p4_hawor_routing/step16_hawor_vs_wilor.py` 迁过来的。原版把
"读 GT + 跑 HaWoR + 读 WiLoR + 算误差 + 画图"挤在一个 93 行的脚本里，要 GPU 才能跑；
现在感知那部分的结果已经冻进 `evidence/depth_benchmark_ho3d/data/`（见
`scripts/dev/freeze_depth_benchmark.py`），这里只剩**纯 numpy 的统计**，秒级、无依赖。

## 四个指标各自在说什么

- ``depth_cm`` —— 手腕深度绝对误差的**中位**。取中位不取均值，因为 HaWoR 会有个别
  离群帧（SMu41 实测深度 range 26 cm 而 GT 只变 1 cm），均值会被一两帧带跑。
- ``inplane_cm`` —— XY 平面误差中位。这一项 HaWoR 反而略逊于 WiLoR，如实报。
- ``depth_rel_pct`` —— 深度误差 / |GT 深度|，用来跨序列比较（绝对值受距离影响）。
- ``depth_r`` —— 估计深度与 GT 深度的相关系数。**这一项比误差更能说明问题**：
  r 为负意味着"物体靠近时它认为在远离"，这种估计做重定向会得到反向的 reach，
  而绝对误差再小也看不出这件事。ABF12 的 WiLoR+MoGe 就是 r=−0.64。

## 一个必须跟着数字一起报的告示

GT 深度本身的变化幅度（``gt_depth_range_cm``）小的时候，``depth_r`` 没有意义 ——
SMu41 的 GT 深度整段只变 1.0 cm，那个 r 是在噪声上算相关。所以它跟着结果一起返回，
而不是留给读表的人自己去查。
"""
from pathlib import Path

import numpy as np

METHODS = ("hawor", "wilor")
SEQUENCES = ("ABF12", "SMu41", "MC4")

#: 冻好的证据放哪 —— 从本文件往上四层是仓库根（src/web2robot/eval/ → 根）
EVIDENCE_DIR = Path(__file__).resolve().parents[3] / "evidence" / "depth_benchmark_ho3d" / "data"


def load_bench(seq, evidence_dir=None):
    """读一条序列冻好的三方手腕点。返回 dict，键见 `freeze_depth_benchmark.py`。"""
    d = Path(evidence_dir or EVIDENCE_DIR) / f"bench_{seq}.npz"
    if not d.is_file():
        raise FileNotFoundError(
            f"找不到 {d}。这是冻好的证据，应该在 git 里；"
            f"要重新生成得跑 scripts/dev/freeze_depth_benchmark.py（需要 GPU + HO-3D）")
    with np.load(d) as z:
        return {k: z[k] for k in z.files}


def align(bench, method):
    """按帧号取 GT 与某个方法的交集。

    和原版 ``errs()`` 一样是**逐方法各自和 GT 求交**，不是三方一起求交 —— 两个方法
    覆盖的帧不同时，强行三方对齐会把某个方法的可用帧白扔掉。所以 GT 深度的统计量
    （range）也是逐方法算的。
    """
    if method not in METHODS:
        raise ValueError(f"method 得是 {METHODS} 之一，给的是 {method!r}")
    gf, gw = bench["gt_frames"], bench["gt_wrist"]
    ef, ew = bench[f"{method}_frames"], bench[f"{method}_wrist"]
    if len(ef) == 0:
        return np.zeros(0, np.int64), np.zeros((0, 3)), np.zeros((0, 3))
    pos = {int(f): i for i, f in enumerate(ef)}
    idx = [(i, pos[int(f)]) for i, f in enumerate(gf) if int(f) in pos]
    gi = np.array([a for a, _ in idx], np.int64)
    ei = np.array([b for _, b in idx], np.int64)
    return gf[gi], gw[gi], ew[ei]


def evaluate(bench, method):
    """一条序列 × 一个方法 → 指标 dict。帧数不足 2 时相关系数给 nan 而不是报错。"""
    frames, gt, est = align(bench, method)
    if len(frames) == 0:
        return None
    dep = np.abs(est[:, 2] - gt[:, 2])
    inp = np.linalg.norm((est - gt)[:, :2], axis=1)
    r = (float(np.corrcoef(est[:, 2], gt[:, 2])[0, 1]) if len(frames) > 1 else float("nan"))
    return {
        "method": method,
        "n_frames": int(len(frames)),
        "depth_cm": float(np.median(dep) * 100),
        "inplane_cm": float(np.median(inp) * 100),
        "depth_rel_pct": float(np.median(dep / np.abs(gt[:, 2])) * 100),
        "depth_r": r,
        "gt_depth_range_cm": float((gt[:, 2].max() - gt[:, 2].min()) * 100),
        "est_depth_range_cm": float((est[:, 2].max() - est[:, 2].min()) * 100),
    }


def evaluate_all(sequences=SEQUENCES, evidence_dir=None):
    """所有序列 × 所有方法。返回 ``{seq: {method: metrics}}``，缺的方法不出现。"""
    out = {}
    for seq in sequences:
        b = load_bench(seq, evidence_dir)
        out[seq] = {m: r for m in METHODS if (r := evaluate(b, m)) is not None}
    return out


def format_table(results):
    """论文那张表的文本版。深度跟随不可信的序列会带一个显式的告示行。"""
    lines = ["序列        n   方法          深度err  平面err  深度相对  深度跟随r",
             "-" * 68]
    notes = []
    for seq, per in results.items():
        for m in METHODS:
            r = per.get(m)
            if r is None:
                continue
            lines.append(f"{seq:<10} {r['n_frames']:>3}  {m:<12} "
                         f"{r['depth_cm']:>6.1f}cm {r['inplane_cm']:>6.1f}cm "
                         f"{r['depth_rel_pct']:>7.0f}% {r['depth_r']:>+9.2f}")
        any_m = next(iter(per.values()), None)
        if any_m and any_m["gt_depth_range_cm"] < 2.0:
            notes.append(f"  ! {seq}: GT 深度整段只变 {any_m['gt_depth_range_cm']:.1f}cm，"
                         f"深度跟随 r 是在噪声上算相关，不可信")
    return "\n".join(lines + ([""] + notes if notes else []))
