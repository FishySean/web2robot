"""重画 HO-3D 深度评测的汇总图 —— 这次数字是**算出来的**，不是抄的。

原版 `step17_summary_3seq.py` 头一行就写着"硬编码自 step16 各序列输出"：六个深度误差、
六个相关系数、三个 GT 运动幅度，全是手抄进源码的。这在当时是最快的做法，但它意味着
**图和数据之间没有任何链接** —— 谁改了评测口径，图不会跟着变；谁抄错一位，也没人拦。
论文要用的图不能是这个状态。

现在两个面板的每个数都从 `evidence/depth_benchmark_ho3d/data/bench_*.npz` 现算
（`web2robot.eval.depth_benchmark`），顺便就把当初那 15 个手抄的数验了一遍。

    PYTHONPATH=src envs/rt_env/bin/python scripts/dev/render_depth_benchmark_fig.py
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from web2robot.eval.depth_benchmark import SEQUENCES, evaluate_all, format_table  # noqa: E402

# step17 里手抄的那 15 个数，留着当参照物核对，勿整理
_OLD_HARDCODED = {
    "wilor_dep": [11.0, 9.5, 0.7], "hawor_dep": [0.6, 3.5, 2.6],
    "wilor_r": [-0.64, 0.11, 0.91], "hawor_r": [0.61, 0.61, 0.87],
    "gt_motion": [3.6, 1.0, 7.1],
}


def check_against_hardcoded(res):
    """现算的值 vs step17 手抄的值。差超过显示精度就报出来。"""
    got = {
        "wilor_dep": [res[s]["wilor"]["depth_cm"] for s in SEQUENCES],
        "hawor_dep": [res[s]["hawor"]["depth_cm"] for s in SEQUENCES],
        "wilor_r": [res[s]["wilor"]["depth_r"] for s in SEQUENCES],
        "hawor_r": [res[s]["hawor"]["depth_r"] for s in SEQUENCES],
        "gt_motion": [res[s]["hawor"]["gt_depth_range_cm"] for s in SEQUENCES],
    }
    bad = 0
    for k, old in _OLD_HARDCODED.items():
        tol = 0.005 if k.endswith("_r") else 0.05
        for seq, o, g in zip(SEQUENCES, old, got[k]):
            if abs(o - g) > tol:
                print(f"  ✗ {k}[{seq}]: 手抄 {o} vs 现算 {g:.4f}")
                bad += 1
    print(f"  step17 手抄的 15 个数：{15 - bad}/15 与现算一致 {'✓' if not bad else '✗'}")
    return got, bad


def render(res, got, out):
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.8))
    x = np.arange(len(SEQUENCES))
    w = 0.36
    b1 = axs[0].bar(x - w / 2, got["wilor_dep"], w, label="WiLoR+MoGe", color="crimson")
    b2 = axs[0].bar(x + w / 2, got["hawor_dep"], w, label="HaWoR", color="seagreen")
    for b in list(b1) + list(b2):
        axs[0].text(b.get_x() + b.get_width() / 2, b.get_height(),
                    f"{b.get_height():.1f}", ha="center", va="bottom", fontsize=9)
    axs[0].set_xticks(x)
    axs[0].set_xticklabels(SEQUENCES)
    axs[0].set_ylabel("wrist depth error (cm)")
    axs[0].set_title("Depth error vs GT per sequence (lower=better)")
    axs[0].legend()
    ceiling = max(got["hawor_dep"])
    axs[0].axhline(ceiling, ls="--", c="gray", lw=1)
    axs[0].text(2.3, ceiling + 0.2, f"HaWoR ceiling ~{ceiling:.1f}cm",
                fontsize=8, color="gray")

    axs[1].bar(x - w / 2, got["wilor_r"], w, label="WiLoR+MoGe", color="crimson")
    axs[1].bar(x + w / 2, got["hawor_r"], w, label="HaWoR", color="seagreen")
    axs[1].axhline(0, c="k", lw=0.8)
    for i, (seq, g) in enumerate(zip(SEQUENCES, got["gt_motion"])):
        # GT 深度变化太小的序列，r 是在噪声上算相关 —— 图上直接标出来，别让读表的人自己去查
        note = f"GT motion\n{g:.1f}cm" + ("\n(r not meaningful)" if g < 2.0 else "")
        axs[1].text(i, -0.82, note, ha="center", fontsize=7,
                    color="firebrick" if g < 2.0 else "dimgray")
    axs[1].set_xticks(x)
    axs[1].set_xticklabels(SEQUENCES)
    axs[1].set_ylabel("depth tracking corr r (higher=better)")
    axs[1].set_ylim(-0.95, 1.05)
    axs[1].set_title("Depth-motion tracking vs GT")
    axs[1].legend(loc="upper left")
    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=100)
    print(f"WROTE {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path,
                    default=REPO / "evidence/depth_benchmark_ho3d/figures/FIG_SUMMARY_3seq.png")
    args = ap.parse_args()
    res = evaluate_all()
    print(format_table(res))
    print()
    got, bad = check_against_hardcoded(res)
    render(res, got, args.out)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
