"""第③步的命令行入口：感知前端产物 → EgoInfinity clip 目录。

    scripts/s3_to_clip.sh hawor external/HaWoR/example/ho3d_SMu41 --frames 55 \\
        --out outputs/clips/ho3d_SMu41 --fps 15 --hands right

前端各有各的环境（HaWoR 在 ``hawor_env``，WiLoR+MoGe 在 ``perception_env``），所以
子命令一个前端一个，薄壳按子命令挑解释器。逻辑全在
:mod:`web2robot.perception.hawor` 等模块里，这里只做参数解析和"把 HaWoR 的三个函数
import 进来传下去"这件接线。

``--frames`` 要给：HaWoR 的 SLAM 结果文件名里带帧数（``hawor_slam_w_scale_0_<N>.npz``），
猜不出来。给错了会当场报 FileNotFoundError 并把 SLAM 目录里实际有什么列出来 ——
比默认猜一个然后读到别的序列的位姿要好。
"""
import argparse
import sys
from pathlib import Path

from web2robot.paths import P
from web2robot.perception.hawor import aperture, hawor_to_joints, read_focal
from web2robot.perception.to_clip import (
    DEFAULT_FOCAL, HAND_RIGHT, valid_frame_counts, write_clip,
)


def _hawor(args) -> int:
    # HaWoR 仓库的函数：只在这里 import，模块层一律靠注入（见 perception.hawor 说明）
    from hawor.utils.process import run_mano, run_mano_left
    from lib.eval_utils.custom_utils import load_slam_cam

    src = Path(args.src)
    joints = hawor_to_joints(src, args.frames, load_slam_cam=load_slam_cam,
                             run_mano=run_mano, run_mano_left=run_mano_left,
                             hands=tuple(args.hands), device=args.device)
    focal = read_focal(src, default=DEFAULT_FOCAL)
    if focal == DEFAULT_FOCAL and not (src / "est_focal.txt").is_file():
        print(f"  est_focal.txt 缺失，focal 用兜底值 {DEFAULT_FOCAL}"
              f"（只影响可视化投影，重定向吃米制 3D 点）")

    out = P.check_output_dir(args.out)
    r = write_clip(out, joints, fps=args.fps, focal=focal, clip_id=args.id)
    n = valid_frame_counts(joints)
    print(f"WROTE {out}  T={r['meta']['n_frames']}  focal={focal}  fps={args.fps}")
    print(f"  有效帧 左{n['left']} 右{n['right']}")
    if "right" in args.hands:
        ap = aperture(joints[:, HAND_RIGHT])
        import numpy as np
        if np.isfinite(ap).any():
            print(f"  右手开合 {np.nanmin(ap)*100:.1f}~{np.nanmax(ap)*100:.1f}cm")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="web2robot.perception", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="frontend", required=True)

    h = sub.add_parser("hawor", help="HaWoR（相机运动的片段走这条）")
    h.add_argument("src", type=Path, help="HaWoR 产物目录（含 world_space_res.pth 和 SLAM/）")
    h.add_argument("--frames", type=int, required=True,
                   help="帧数；HaWoR 的 SLAM 文件名里带这个数")
    h.add_argument("--out", type=Path, required=True, help="clip 目录（落 outputs/ 下）")
    h.add_argument("--fps", type=float, default=15.0)
    h.add_argument("--hands", nargs="+", default=["left", "right"],
                   choices=["left", "right"],
                   help="要导哪几只手；单手序列只给一只，另一只整段留 NaN")
    h.add_argument("--id", default=None, help="片段 id，默认取 --out 的目录名")
    h.add_argument("--device", default="cuda")
    h.set_defaults(fn=_hawor)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
