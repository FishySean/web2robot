"""把两条深度策略产出的手形画出来看 —— 因为开合数字差了 10 倍，光看数不能判谁坏在哪。

`pointmap` 报 8~15 cm 开合，`global-scale` 报 1.4~1.5 cm。两个都不像真的（ABF12 里手一直
攥着一个罐子），但**坏法不同**，而这个区别只有看图才分得出来：

- pointmap 逐关节独立取深度 → 手形被深度噪声撕开，指尖乱飞，所以开合忽大忽小；
- global-scale 手形是 WiLoR 自洽的 → 形状对，但整段乘一个错的尺度 → 整只手偏小。

四宫格：RGB+2D 关键点 / MoGe 深度 / pointmap 手形 / global-scale 手形。

    CUDA_VISIBLE_DEVICES=7 PYTHONPATH=<repo>/src \\
      HF_HOME=$HOME/.cache/huggingface \\
      <repo>/envs/perception_env/bin/python scripts/dev/viz_wilor_depth_modes.py \\
      --clips outputs/clips/cli_smoke_abf12_pointmap outputs/clips/cli_smoke_abf12_globalscale \\
      --rgb /tmp/abf12_rgb --out outputs/viz/wilor_depth_modes.mp4

不吃 GPU（深度图那格从 clip 里的点反推不出来，所以要么现算要么省掉 —— 这里省掉，
用 clip 里已有的东西画三格，第四格画两条策略的手腕深度曲线）。
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

# MANO 21 点的骨架连线（手腕 0 → 四根手指 + 拇指）
BONES = [(0, 1), (1, 2), (2, 3), (3, 4),
         (0, 5), (5, 6), (6, 7), (7, 8),
         (0, 9), (9, 10), (10, 11), (11, 12),
         (0, 13), (13, 14), (14, 15), (15, 16),
         (0, 17), (17, 18), (18, 19), (19, 20)]


def load_clip(d):
    d = Path(d)
    meta = json.loads((d / "hand_meta.json").read_text())
    J = np.fromfile(d / "hand_joints.bin", dtype=np.float32).reshape(meta["joints_shape"])
    return J.astype(np.float64)


def bone_lengths(joints_1hand):
    """每帧的骨长 (T, 20)。真手的骨长是常数，所以这是"手形被撕开"的直接度量。"""
    return np.array([[np.linalg.norm(f[a] - f[b]) for a, b in BONES]
                     for f in joints_1hand])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clips", nargs=2, required=True, metavar=("POINTMAP", "GLOBALSCALE"))
    ap.add_argument("--rgb", required=True, help="原始图片目录")
    ap.add_argument("--out", required=True)
    ap.add_argument("--hand", type=int, default=1, help="0=左 1=右")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter
    import cv2

    Jp = load_clip(args.clips[0])[:, args.hand]
    Jg = load_clip(args.clips[1])[:, args.hand]
    imgs = sorted(Path(args.rgb).glob("*.jpg"))
    T = min(len(Jp), len(Jg), len(imgs))
    print(f"T={T}  pointmap{Jp.shape}  global-scale{Jg.shape}")

    bl_p, bl_g = bone_lengths(Jp[:T]), bone_lengths(Jg[:T])
    # 骨长的逐帧变动幅度：真手应当接近 0
    cv_p = np.nanstd(bl_p, 0) / np.nanmean(bl_p, 0)
    cv_g = np.nanstd(bl_g, 0) / np.nanmean(bl_g, 0)
    print(f"骨长变异系数（越小越像真手）: pointmap {np.nanmean(cv_p)*100:.1f}%  "
          f"global-scale {np.nanmean(cv_g)*100:.1f}%")
    print(f"骨长均值: pointmap {np.nanmean(bl_p)*100:.2f}cm  "
          f"global-scale {np.nanmean(bl_g)*100:.2f}cm  （真手指节 ~2-4cm）")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(13, 7))
    axi = fig.add_subplot(2, 2, 1)
    axp = fig.add_subplot(2, 2, 2, projection="3d")
    axg = fig.add_subplot(2, 2, 3, projection="3d")
    axb = fig.add_subplot(2, 2, 4)

    # 第四格是静态的：两条策略的骨长分布，一眼看出谁的手形在抖
    axb.boxplot([bl_p.ravel() * 100, bl_g.ravel() * 100],
                labels=["pointmap", "global-scale"])
    axb.axhspan(2, 4, color="green", alpha=0.15)
    axb.text(0.02, 0.96, "green band = real finger bone 2-4cm", transform=axb.transAxes,
             va="top", fontsize=9, color="darkgreen")
    axb.set_ylabel("bone length (cm)")
    axb.set_title("bone length: constant on a real hand; spread = shape torn apart",
                  fontsize=10)
    axb.set_yscale("log")

    # 两个 3D panel 用**同一个**半径，否则各自 autoscale 会把 6.5 倍的尺度差藏掉 ——
    # 而"手被缩小了"恰恰是 global-scale 的病，必须看得见
    VIEW_R = 0.10

    writer = FFMpegWriter(fps=10, codec="libx264",
                          extra_args=["-pix_fmt", "yuv420p", "-crf", "20"])
    with writer.saving(fig, str(out), dpi=110):
        for t in range(T):
            axi.clear(); axp.clear(); axg.clear()
            im = cv2.cvtColor(cv2.imread(str(imgs[t])), cv2.COLOR_BGR2RGB)
            axi.imshow(im); axi.set_title(f"RGB  f{t}"); axi.axis("off")

            for ax, J, name, cvv, bl in ((axp, Jp, "pointmap", cv_p, bl_p),
                                         (axg, Jg, "global-scale", cv_g, bl_g)):
                f = J[t]
                if np.isfinite(f).all():
                    ax.scatter(f[:, 0], f[:, 2], -f[:, 1], s=12, c="crimson")
                    for a, b in BONES:
                        ax.plot([f[a, 0], f[b, 0]], [f[a, 2], f[b, 2]],
                                [-f[a, 1], -f[b, 1]], c="steelblue", lw=1.2)
                    c = f.mean(0)
                    ax.set_xlim(c[0] - VIEW_R, c[0] + VIEW_R)
                    ax.set_ylim(c[2] - VIEW_R, c[2] + VIEW_R)
                    ax.set_zlim(-c[1] - VIEW_R, -c[1] + VIEW_R)
                ax.set_title(f"{name}   bone {np.nanmean(bl)*100:.2f}cm, "
                             f"wobble {np.nanmean(cvv)*100:.0f}%", fontsize=10)
                ax.set_xlabel("x (m)"); ax.set_ylabel("z = depth (m)")
            writer.grab_frame()
    plt.close(fig)
    print(f"WROTE {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
