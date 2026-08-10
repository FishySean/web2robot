"""把 HO-3D 深度评测的证据冻成"纯 numpy 就能复现"的形式，一次性脚本。

为什么不是直接把旧文件拷进 git：旧那份复现路径太脆。原版 `step16_hawor_vs_wilor.py`
每次都要 **GPU + hawor_env + HaWoR checkout + HO-3D 数据集** 才能算出那张表，而这四样里
有三样是随时会没的东西 ——

- HO-3D（`rgbd_val/`）是外部数据集，`.gitignore` 里明确排掉了；
- `HaWoR/example/ho3d_*/` 是第三方 checkout 里的产物，一次 `git clean -xdf` 就没；
- `hawor.ckpt` 3.27 GB，也不在库里（当初还下载截断过一次）。

所以这里把**三方的手腕 3D 点**（GT / HaWoR / WiLoR+MoGe）按帧对齐存成一个小 npz。
存的是原始 3D 点、不是算好的误差数 —— 这样论文里换个统计口径（比如改中位为均值、
换成 per-joint）还能重算，而不是只剩三个写死的数字。冻完总共几十 KB，进得了 git。

HaWoR 那一路**算两遍**：一遍是 `step16` 的内联写法逐字抄过来，一遍走迁移后的
`web2robot.perception.hawor`，然后断言逐位相同。等于顺手给 perception 模块多加了三条
真实序列的比对（原来只比过 ho3d_SMu41 一条）。

跑法（需要 GPU、hawor_env、HaWoR checkout、HO-3D）：

    scripts/dev/m7_tool.sh  # ← 不是这个，这个是 rt_env
    CUDA_VISIBLE_DEVICES=3 PYTHONPATH=/mnt/vlm/fanshaoheng/web2robot/src:/mnt/vlm/fanshaoheng/HaWoR \\
      /mnt/vlm/fanshaoheng/web2robot/envs/hawor_env/bin/python \\
      /mnt/vlm/fanshaoheng/web2robot/scripts/dev/freeze_depth_benchmark.py

冻好之后就再也不需要跑它了 —— 评测走 `web2robot.eval.depth_benchmark`，纯 numpy。
"""
import os
import pickle
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np

# HO-3D 的 handJoints3D 是 y/z 翻向的（官方 coordChangeMat），转过来才是 z 朝前的相机系
COORD_CHANGE = np.diag([1.0, -1.0, -1.0])

HO3D = "/mnt/vlm/fanshaoheng/hand_projects/hand2robot/rgbd_val/ho3d/train"
HAWOR_EX = "/mnt/vlm/fanshaoheng/HaWoR/example"
WILOR_NPZ = "/mnt/vlm/fanshaoheng/hand_projects/hand2robot/outputs/eval_hawor/data"
OUT = "/mnt/vlm/fanshaoheng/web2robot/evidence/depth_benchmark_ho3d/data"

# 序列名 → 当初跑 step16 时给的 NFR（SLAM 文件名里带这个数，猜不出来）
SEQS = {"ABF12": 88, "SMu41": 55, "MC4": 77}


def load_gt(seq, nfr):
    """HO-3D 真值手腕。只有部分帧有标注，缺帧的 meta 是 15 字节的假 pkl，要过滤。"""
    frames, wrists = [], []
    for k in range(nfr):
        p = f"{HO3D}/{seq}/meta/{k:04d}.pkl"
        if not os.path.exists(p):
            continue
        try:
            m = pickle.load(open(p, "rb"), encoding="latin1")
        except Exception:
            continue
        if m.get("handJoints3D") is None:
            continue
        J = np.array(m["handJoints3D"]).reshape(21, 3) @ COORD_CHANGE.T
        frames.append(k)
        wrists.append(J[0])
    return np.array(frames, np.int64), np.array(wrists, np.float64)


def hawor_inline(seq, nfr):
    """`step16_hawor_vs_wilor.py` 第 34-42 行逐字抄过来，当参照物用，勿整理。"""
    import joblib
    import torch
    from hawor.utils.process import run_mano
    from lib.eval_utils.custom_utils import load_slam_cam

    HAWOR = f"{HAWOR_EX}/ho3d_{seq}"
    r = joblib.load(f"{HAWOR}/world_space_res.pth")
    trans, rot, hpose, betas, valid = [torch.tensor(np.array(x)).float() for x in r]
    out = run_mano(trans[1:2].cuda(), rot[1:2].cuda(), hpose[1:2].cuda(),
                   betas=betas[1:2].cuda())          # 右手=idx1
    Jw = out["joints"][0].cpu().numpy()               # (T,21,3) world
    R_w2c, t_w2c, _, _ = load_slam_cam(f"{HAWOR}/SLAM/hawor_slam_w_scale_0_{nfr}.npz")
    R_w2c = R_w2c.numpy()
    t_w2c = t_w2c.numpy()
    Jc = np.einsum("tij,tkj->tki", R_w2c, Jw) + t_w2c[:, None, :]
    return Jc


def hawor_via_module(seq, nfr):
    """迁移后的模块走一遍，用来和上面那份逐位对照。"""
    from pathlib import Path

    from hawor.utils.process import run_mano, run_mano_left
    from lib.eval_utils.custom_utils import load_slam_cam

    from web2robot.perception.hawor import hawor_to_joints
    from web2robot.perception.to_clip import HAND_RIGHT

    J = hawor_to_joints(Path(f"{HAWOR_EX}/ho3d_{seq}"), nfr,
                        load_slam_cam=load_slam_cam, run_mano=run_mano,
                        run_mano_left=run_mano_left, hands=("right",))
    return J[:, HAND_RIGHT]


def main():
    os.makedirs(OUT, exist_ok=True)
    bad = 0
    for seq, nfr in SEQS.items():
        print(f"\n=== {seq} (NFR={nfr}) ===")
        gt_frames, gt_wrist = load_gt(seq, nfr)
        print(f"  GT 有标注的帧: {len(gt_frames)}")

        Jc_inline = hawor_inline(seq, nfr)
        Jc_module = hawor_via_module(seq, nfr)

        # 只在 GT 帧上比（评测也只用这些帧）；模块版会把 invalid 帧置 NaN，内联版不会
        keep = gt_frames[gt_frames < len(Jc_inline)]
        a = Jc_inline[keep, 0]
        b = Jc_module[keep, 0]
        finite = np.isfinite(b).all(axis=1)
        same = np.array_equal(a[finite], b[finite])
        print(f"  内联版 vs 模块版：{finite.sum()}/{len(keep)} 帧模块判为 valid，"
              f"逐位一致 {'✓' if same else '✗ !!'}")
        if not same:
            bad += 1
            d = np.abs(a[finite] - b[finite])
            print(f"    最大差 {d.max():.3e} m")
        if finite.sum() != len(keep):
            print(f"    注意：模块版多屏掉了 {len(keep) - finite.sum()} 帧"
                  f"（内联版会把这些帧的垃圾值算进统计）")

        wp = f"{WILOR_NPZ}/wilor_moge_{seq}.npz"
        if os.path.exists(wp):
            d = np.load(wp)
            wl_frames, wl_wrist = d["frames"].astype(np.int64), d["wrist"].astype(np.float64)
        else:
            print(f"  !! 没有 {wp}，WiLoR 这一路留空")
            wl_frames, wl_wrist = np.zeros(0, np.int64), np.zeros((0, 3))
        print(f"  WiLoR+MoGe 帧: {len(wl_frames)}")

        out = f"{OUT}/bench_{seq}.npz"
        np.savez(out,
                 gt_frames=gt_frames, gt_wrist=gt_wrist,
                 hawor_frames=np.arange(len(Jc_inline), dtype=np.int64),
                 hawor_wrist=Jc_inline[:, 0].astype(np.float64),
                 wilor_frames=wl_frames, wilor_wrist=wl_wrist,
                 nfr=np.int64(nfr))
        print(f"  WROTE {out}  ({os.path.getsize(out)/1024:.1f} KB)")

    print(f"\n{'全部序列内联版与模块版逐位一致 ✓' if bad == 0 else f'{bad} 个序列不一致 ✗'}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
