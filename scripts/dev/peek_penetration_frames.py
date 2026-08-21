"""某段片段的残留穿透，逐帧拆开看：深度分布 + 最坏帧号 + 代理当时怎么读的。

`collcmp_table.py` 出的是每段一行的汇总，够用来比两条路线，**不够用来判断一段到底
坏在哪**。「90/104 帧穿模」和「49/257 帧穿模」哪个更该修，光看帧数是反的：前者中位数
1.26 cm、抽帧看过去前臂贴着胸甲，肉眼看不出；后者最深 13.16 cm，整条小臂埋进躯干。

所以这个脚本回答三个问题：
  1. 那些穿模帧**有多深**（分位数）—— 决定要不要管；
  2. **哪几帧最坏** —— 直接给帧号，好拿 ffmpeg 抽出来看画面；
  3. 那几帧上**我方代理读数是多少** —— 正数 = 代理说"还没碰"（漏检，病在检测），
     负数 = 代理说"已经穿了"（报了但没修动，病在过滤器/源头坏帧）。
     这两种病的治法完全不同，混成一个就会一直调错地方。

    scripts/dev/m7_tool.sh peek_penetration_frames.py \
        outputs/retarget/collcmp_cal/-0RheyDV3a0_48.6_55.3_grid --route grid

`--route` 决定拿哪把尺子算代理读数：过滤器当时用的就是那条路线标定的盒子，
拿另一条的盒子去读等于"拿 A 尺子量 B 尺子"（同 `collcmp_table.py --proxy preset`）。
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np


def _load_collcmp():
    """借用 collcmp_table.py 的量法，保证和汇总表口径一致（不复制一份判据）。"""
    here = Path(__file__).resolve().parent
    sys.path.insert(0, str(here))            # audit_mujoco_contacts 在同目录
    spec = importlib.util.spec_from_file_location("_cct", here / "collcmp_table.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_cct"] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run_dir", type=Path, nargs="+", help="重定向产物目录（含 trajectory.npz）")
    ap.add_argument("--route", default="grid", choices=("grid", "neural"),
                    help="代理读数按哪条路线标定的盒子算（默认 grid）")
    ap.add_argument("--top", type=int, default=5, help="报最坏的几帧")
    args = ap.parse_args()

    cct = _load_collcmp()
    env = cct.M7Env()
    env.reset()
    cap = cct.M7CapsuleModel(
        env.model, torso_half=cct.arm_torso_preset(args.route).get("torso_half"))

    for d in args.run_dir:
        r = cct.measure(d, env, cap)
        mesh = r["_frames"]["mesh"] * 100.0      # cm，真实网格判据，臂/手取深的那个
        ours = r["_frames"]["ours"] * 100.0      # cm，我方代理的有符号距离
        pen = np.flatnonzero(mesh > 0)
        print(f"\n=== {d.name} ===  {len(pen)}/{len(mesh)} 帧穿  最深 {mesh.max():.2f} cm  "
              f"（臂 {r['pen_frames_arm']} / 手 {r['pen_frames_hand']}）")
        if not len(pen):
            continue
        print(f"  穿透深度 cm: p50={np.median(mesh[pen]):.2f} "
              f"p90={np.percentile(mesh[pen], 90):.2f} max={mesh.max():.2f}   "
              f"<1cm 占 {(mesh[pen] < 1).mean() * 100:.0f}%")
        worst = np.argsort(-mesh)[:args.top]
        print("  最坏帧: " + ", ".join(f"f{i}={mesh[i]:.2f}(代理{ours[i]:+.2f})" for i in worst))
        # 正/负一句话结论，免得每次自己回想符号约定
        pos = (ours[worst] > 0).sum()
        print(f"  → 最坏 {len(worst)} 帧里 {pos} 帧代理是正数（漏检，病在检测），"
              f"{len(worst) - pos} 帧是负数（报了没修动，病在过滤器或源头坏帧）")


if __name__ == "__main__":
    main()
