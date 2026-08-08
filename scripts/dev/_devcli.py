"""scripts/dev/ 里出片脚本的公共入口参数。

这些脚本都要吃"一次 ``s4_retarget.sh`` 的输出目录"（里面有 ``trajectory.npz``）。
迁移前它们各自写死了 ``/mnt/vlm/fanshaoheng/phase1_repro/m7_test_out`` —— 重构前的
一次性目录，换机器或换一次实验就指错，而且**指错了不会报错，只会 render 出一张
似是而非的旧图**。所以统一改成必填参数。

放在这里而不是 ``src/web2robot/`` 下，是因为它只服务开发期出片脚本，不是流水线逻辑；
``scripts/dev/`` 会成为 ``sys.path[0]``（``m7_tool.sh`` 直接跑脚本文件），所以
``from _devcli import ...`` 就能 import，不需要把它做成包。
"""
import argparse
from pathlib import Path

import numpy as np


def parser(doc: str) -> argparse.ArgumentParser:
    """建一个带 ``run_dir`` / ``--out`` 的 parser；调用方再加自己的参数。"""
    p = argparse.ArgumentParser(description=doc,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir", type=Path,
                   help="s4_retarget.sh 的输出目录（里面要有 trajectory.npz）")
    p.add_argument("--out", type=Path, default=None,
                   help="产物落地目录，默认就写回 run_dir")
    return p


def load_traj(args):
    """校验目录、准备产物目录、加载 trajectory.npz。

    返回 ``(traj, out_dir)``。找不到 npz 就直接退出 —— 宁可当场报，也不要拿着
    一份别的实验的轨迹 render 出一张看起来"差不多对"的图。
    """
    npz = args.run_dir / "trajectory.npz"
    if not npz.is_file():
        raise SystemExit(f"找不到 {npz}\n"
                         f"run_dir 要指向 s4_retarget.sh 的输出目录，例如：\n"
                         f"  scripts/s4_retarget.sh examples/fill_jar --robot m7 "
                         f"--out /tmp/fill_jar ...")
    out = args.out or args.run_dir
    out.mkdir(parents=True, exist_ok=True)
    return np.load(npz, allow_pickle=True), out
