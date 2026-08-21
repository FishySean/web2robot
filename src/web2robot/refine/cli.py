"""``python -m web2robot.refine`` —— 事后重判，不用重跑 IK。

读一个重定向输出目录里的 ``object_poses.npz`` + ``hand_poses.npz``，换块长 / 预算 /
λ 再判一遍。调阈值的时候用得上：一次 IK 要几分钟，重判是毫秒级。

::

    envs/rt_env/bin/python -m web2robot.refine \\
        --run outputs/retarget/-1r9yl-P-Ao_86.3_90.8 --horizon 20 --per_frame_budget 0.05

``hand_poses.npz`` 是 ``test.py --action_refine …`` 落的。只有 ``object_poses.npz``
（即只开过 ``--object_tracking on``）的目录判不了 —— 缺了"执行后位姿"这一半，
误差无从算起，这里直接报错而不是拿参考当执行（那会得出恒为 0 的误差）。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from web2robot.refine.blocks import H_DEFAULT, RefineConfig
from web2robot.refine.modes import MODES
from web2robot.refine.run import refine_run
from web2robot.refine.score import ErrorWeights


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m web2robot.refine",
        description="动作分级精修的判决（EgoEngine §3.2.2），第一期只判不解")
    ap.add_argument("--run", required=True, help="重定向输出目录")
    ap.add_argument("--out", help="判决落哪（默认就写回 --run 那个目录）")
    ap.add_argument("--horizon", type=int, default=H_DEFAULT,
                    help=f"块长，论文 H={H_DEFAULT} 个控制步")
    ap.add_argument("--per_frame_budget", type=float, default=0.05,
                    help="每帧误差预算，累计上限 = 它 × 块长（论文没给数值）")
    ap.add_argument("--lam_p", type=float, default=1.0, help="e_t 里的 λp（论文没给）")
    ap.add_argument("--lam_R", type=float, default=1.0, help="e_t 里的 λR（论文没给）")
    ap.add_argument("--min_valid_frac", type=float, default=0.5,
                    help="有效帧比例低于此值的块判 unknown，不判 ok")
    ap.add_argument("--mode", choices=("none",) + MODES[1:], default="none",
                    help="等价于 --action_refine；mpc/rl 求解器未实现，只影响判决里"
                         "写的目标模式")
    args = ap.parse_args(argv)

    from web2robot.paths import P
    run = Path(args.run).resolve()
    obj = run / "object_poses.npz"
    hands = run / "hand_poses.npz"
    if not obj.exists():
        raise SystemExit(f"{obj} 不存在 —— 这个目录没开过 --object_tracking on")
    if not hands.exists():
        raise SystemExit(
            f"{hands} 不存在 —— 它是 test.py --action_refine 落的。"
            "只有 object_poses.npz 的目录判不了：缺执行后位姿这一半。")

    hp = np.load(hands)
    root_R = hp["root_R"] if "root_R" in hp.files else None
    root_t = hp["root_t"] if "root_t" in hp.files else None
    cfg = RefineConfig(horizon=args.horizon, per_frame_budget=args.per_frame_budget,
                       weights=ErrorWeights(args.lam_p, args.lam_R),
                       min_valid_frac=args.min_valid_frac)
    out_dir = P.check_output_dir(Path(args.out).resolve() if args.out else run)
    print(f"Run    : {run}")
    print(f"Out    : {out_dir}")
    refine_run(out_dir, obj, hp["hand_ref"], hp["hand_ach"], root_R, root_t,
               cfg=cfg, requested=args.mode, write_hand_poses=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
