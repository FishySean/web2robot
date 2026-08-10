"""WiLoR+MoGe 迁移的真实数据比对：新模块能不能逐位复现 HO-3D 评测那份手腕点。

判据很硬：`evidence/depth_benchmark_ho3d/data/bench_*.npz` 里的 ``wilor_wrist`` 是
2026-07-14 用 `step16b_wilor_moge_abf12.py` 跑出来的，论文表里 "11.0 cm" 那个数就是
从它算的。新模块重跑一遍，**必须逐位相同** —— 不同就说明取深度或反投影的算法在迁移中
被改动了，而那会直接改动论文里的数字。

这条比对不能做成单测（要 GPU + wilor-mini + moge + HO-3D 数据集），所以放 dev。

    CUDA_VISIBLE_DEVICES=7 PYTHONPATH=/mnt/vlm/fanshaoheng/web2robot/src \\
      HF_HOME=/mnt/vlm/fanshaoheng/.cache/huggingface \\
      /mnt/vlm/fanshaoheng/web2robot/envs/perception_env/bin/python \\
      /mnt/vlm/fanshaoheng/web2robot/scripts/dev/check_wilor_vs_baseline.py

``HF_HOME`` 那行不是可选的：这台机器的 shell 里 ``HF_HOME=/mnt/vlm/common/cache``
（共享目录，我们**没有写权限**），而 MoGe 的权重实际缓存在自己家目录下。不覆盖就是
``PermissionError: '/mnt/vlm/common/cache/hub/models--Ruicheng--moge-2-vitl-normal'``
—— 看起来像权重没下载，其实是 hf_hub 想在只读目录里建缓存。

注意一处**照抄的可疑行为**：`step16b` 取的是 ``outs[0]``，也就是"第一个检测框"，
完全不看左右手。HO-3D 那三条序列画面里只有一只手，所以当时没出问题；但拿到双手片段上
这行会随机取到另一只手。这里为了逐位复现照抄，**不在比对脚本里"修"它** ——
生产路径 `wilor_to_joints` 走的是按左右手分槽（`hand_slot`），本来就没有这个问题。
"""
import argparse
import os
import pickle
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

HO3D = "/mnt/vlm/fanshaoheng/hand_projects/hand2robot/rgbd_val/ho3d/train"
SEQS = {"ABF12": 88, "SMu41": 55, "MC4": 77}


def _old_wrist(det, dep, K):
    """`step16b_wilor_moge_abf12.py` 第 23-27 行逐字抄过来，参照物，勿整理。"""
    H, W = dep.shape
    kp2d = np.array(det["wilor_preds"]["pred_keypoints_2d"]).reshape(21, 2)
    wu, wv = kp2d[0]
    xi, yi = int(np.clip(wu, 0, W - 1)), int(np.clip(wv, 0, H - 1))
    patch = dep[max(0, yi - 3):yi + 4, max(0, xi - 3):xi + 4]
    dm = np.nanmedian(patch[np.isfinite(patch)])
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    return np.array([(xi - cx) / fx * dm, (yi - cy) / fy * dm, dm])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seqs", nargs="+", default=list(SEQS))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    import cv2
    import torch
    from moge.model.v2 import MoGeModel
    from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import (
        WiLorHandPose3dEstimationPipeline,
    )

    from web2robot.eval.depth_benchmark import load_bench
    from web2robot.perception.wilor import joints_from_depth_and_K

    wp = WiLorHandPose3dEstimationPipeline(device=args.device, dtype=torch.float16)
    moge = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal").to(args.device).eval()

    bad = 0
    for seq in args.seqs:
        nfr = SEQS[seq]
        frozen = load_bench(seq)
        want_frames = frozen["wilor_frames"]
        want_wrist = frozen["wilor_wrist"]

        got_frames, got_new, got_old = [], [], []
        for k in range(nfr):
            mp = f"{HO3D}/{seq}/meta/{k:04d}.pkl"
            rp = f"{HO3D}/{seq}/rgb/{k:04d}.jpg"
            if not os.path.exists(mp) or not os.path.exists(rp):
                continue
            try:
                m = pickle.load(open(mp, "rb"), encoding="latin1")
            except Exception:
                continue
            if m.get("handJoints3D") is None:
                continue
            K = np.array(m["camMat"])
            rgb = cv2.cvtColor(cv2.imread(rp), cv2.COLOR_BGR2RGB)
            outs = wp.predict(rgb)
            if not outs:
                continue
            t = torch.tensor(rgb / 255., dtype=torch.float32,
                             device=args.device).permute(2, 0, 1)
            with torch.no_grad():
                dep = moge.infer(t)["depth"].cpu().numpy()
            got_frames.append(k)
            got_old.append(_old_wrist(outs[0], dep, K))
            # 新模块：只要手腕那一个关节，算法应当与上面逐位一致
            got_new.append(joints_from_depth_and_K(outs[0], dep, K, joint_indices=[0])[0])

        got_frames = np.array(got_frames, np.int64)
        got_new = np.array(got_new)
        got_old = np.array(got_old)

        print(f"\n=== {seq} ===")
        print(f"  帧号: 冻结 {len(want_frames)} / 本次 {len(got_frames)}  "
              f"{'一致 ✓' if np.array_equal(want_frames, got_frames) else '不一致 ✗'}")

        same_new_old = np.array_equal(got_new, got_old)
        print(f"  新模块 vs 内联参照物: {'逐位一致 ✓' if same_new_old else '不一致 ✗'}")
        if not same_new_old:
            bad += 1
            print(f"    最大差 {np.nanmax(np.abs(got_new - got_old)):.3e} m")

        if np.array_equal(want_frames, got_frames):
            same_frozen = np.array_equal(got_new, want_wrist)
            print(f"  新模块 vs 2026-07-14 冻结值: "
                  f"{'逐位一致 ✓' if same_frozen else '不一致 ✗'}")
            if not same_frozen:
                bad += 1
                d = np.abs(got_new - want_wrist)
                print(f"    最大差 {np.nanmax(d):.3e} m，"
                      f"深度维最大差 {np.nanmax(d[:, 2]):.3e} m")
                # 论文里的数字会不会因此变？直接算出来看
                dz_old = np.abs(want_wrist[:, 2] - frozen["gt_wrist"][:, 2])
                dz_new = np.abs(got_new[:, 2] - frozen["gt_wrist"][:, 2])
                print(f"    深度误差中位: 冻结 {np.median(dz_old)*100:.2f}cm → "
                      f"本次 {np.median(dz_new)*100:.2f}cm")
        else:
            bad += 1

    print(f"\n{'全部序列逐位一致 ✓' if bad == 0 else f'{bad} 处不一致 ✗'}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
