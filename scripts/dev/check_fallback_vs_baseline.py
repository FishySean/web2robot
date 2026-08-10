#!/usr/bin/env python3
"""在**真实片段**上比对 ``retarget/fallback.py`` 与迁移前内联实现（2026-08-10 基线）。

和 ``tests/test_retarget_modules.py`` 是一对，分工不同：

- 单测用合成输入，好处是能**刻意**打到三条填补分支、跑得快、不需要 ``external/``；
  代价是它覆盖不了真实数据的长相（四元数跳变的实际幅度、深度爆点的实际形状）。
- 这个脚本拿 ``external/`` 里所有 clip 的原始手腕轨迹跑同一套比对，覆盖真实长相；
  代价是它需要 ``external/`` 在位，所以进不了单测。

两边的参照实现是同一段照抄的旧代码。哪天 ``traj_cleanup`` 的判据要改（换感知前端
时会），先跑这个脚本，就知道改动波及了哪几段片段、波及多少帧 —— 比"端到端 md5
变了"这条信号早得多，也具体得多。

纯 CPU，不需要模型和 checkpoint，11 段片段十几秒跑完。

    scripts/dev/m7_tool.sh check_fallback_vs_baseline.py
"""
import argparse
import contextlib
import hashlib
import io
import sys
from pathlib import Path

import numpy as np

from utils.clip_io import SamplesSequence
from web2robot.paths import P
from web2robot.retarget.fallback import (
    apply_rest_fallback, clean_input_wrists, relax_fingers_on_rest, status_overlay_text,
)
from web2robot.robots.m7 import CONFIG as M7_CONFIG
from web2robot.trajectory.traj_cleanup import (
    FILL_REST, STATUS_NAMES, blend_to_rest, clean_wrist_trajectory, relax_fingers,
)

RAMP_SEC = 0.5
CLEAN_KW = dict(max_interp_sec=1.5, max_hold_sec=0.5, detect_bad=True)


# ── 迁移前的内联实现（照抄 2026-08-10 之前的 scripts/test.py，勿整理） ──────────

def _old_clean(raw_left, raw_right, fps):
    left, st_l, ca_l, _ = clean_wrist_trajectory(raw_left, fps, side="left", **CLEAN_KW)
    right, st_r, ca_r, _ = clean_wrist_trajectory(raw_right, fps, side="right", **CLEAN_KW)
    if np.isnan(left[:, 0]).all() or np.isnan(right[:, 0]).all():
        _miss = "left" if np.isnan(left[:, 0]).all() else "right"
        raise RuntimeError(f"{_miss} hand is never detected")
    return left, right, st_l, st_r, ca_l, ca_r


def _old_rest(q_left, q_right, st_l, st_r, rest, fps, T):
    w_l, w_r = np.zeros(T), np.zeros(T)
    if (st_l == FILL_REST).any():
        q_left, w_l = blend_to_rest(q_left, st_l, np.asarray(rest["left"], np.float64),
                                    fps, ramp_sec=RAMP_SEC)
    if (st_r == FILL_REST).any():
        q_right, w_r = blend_to_rest(q_right, st_r, np.asarray(rest["right"], np.float64),
                                     fps, ramp_sec=RAMP_SEC)
    return q_left, q_right, w_l, w_r


def _old_overlay(st_l, st_r, T):
    return "|".join(
        " ".join(f"{s}:{STATUS_NAMES[int(a[t])]}"
                 for s, a in (("L", st_l), ("R", st_r)) if int(a[t]) != 0)
        for t in range(T))


# ── 比对 ─────────────────────────────────────────────────────────────────────

def _quiet(fn):
    """跑 fn，吞掉它的进度输出；报错的话把错误类型也当成"结果"一起比。

    整段单手的片段两边都该抛 RuntimeError —— 这条行为和逐位一致同等重要，
    不能只比"跑通了的"那几段（第一版就漏了这条，结果假红了一次）。
    """
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            return "ok", fn()
    except RuntimeError as e:
        return "raise", str(e).split(" — ")[0].split(" hand is never")[0]


def _md5(a):
    return hashlib.md5(np.ascontiguousarray(a).tobytes()).hexdigest()[:12]


def check_clip(clip: Path) -> tuple[bool, str]:
    seq = SamplesSequence(clip)
    fps = seq.fps
    raw_left, raw_right = seq.raw_wrist_trajectories()
    T = len(raw_left)

    kind_new, new = _quiet(lambda: clean_input_wrists(raw_left, raw_right, fps,
                                                     log=lambda *_: None, **CLEAN_KW))
    kind_old, old = _quiet(lambda: _old_clean(raw_left, raw_right, fps))
    if "raise" in (kind_new, kind_old):
        same = kind_new == kind_old and new == old
        return same, (f"T={T:<4} 整段单手 → 两边都拒掉"
                      f"  {'✓ 行为一致' if same else f'✗ {kind_new}/{kind_old}'}")

    o_left, o_right, o_st_l, o_st_r, o_ca_l, o_ca_r = old

    # 关节角/手指角在这一步之前是 IK 出来的，这里不需要真值 —— 兜底只按权重混合，
    # 固定种子的随机数组照样能把"混合结果逐位一致"证出来，而且不用跑模型。
    rng = np.random.default_rng(3)
    q_l, q_r = rng.normal(0, .4, (T, 7)), rng.normal(0, .4, (T, 7))
    Q_lf, Q_rf = rng.uniform(0, 1.5, (T, 12)), rng.uniform(0, 1.5, (T, 12))

    _, n_rest = _quiet(lambda: apply_rest_fallback(
        q_l.copy(), q_r.copy(), new.status_left, new.status_right,
        M7_CONFIG["start_config"], fps, ramp_sec=RAMP_SEC, log=lambda *_: None))
    o_rest = _old_rest(q_l.copy(), q_r.copy(), o_st_l, o_st_r,
                       M7_CONFIG["start_config"], fps, T)

    n_lf, n_rf = relax_fingers_on_rest(Q_lf.copy(), Q_rf.copy(), n_rest[2], n_rest[3])
    o_lf = relax_fingers(Q_lf.copy(), o_rest[2]) if o_rest[2].any() else Q_lf.copy()
    o_rf = relax_fingers(Q_rf.copy(), o_rest[3]) if o_rest[3].any() else Q_rf.copy()

    pairs = [("left", new.left, o_left), ("right", new.right, o_right),
             ("status_left", new.status_left, o_st_l),
             ("status_right", new.status_right, o_st_r),
             ("cause_left", new.cause_left, o_ca_l),
             ("cause_right", new.cause_right, o_ca_r),
             ("q_left", n_rest[0], o_rest[0]), ("q_right", n_rest[1], o_rest[1]),
             ("w_left", n_rest[2], o_rest[2]), ("w_right", n_rest[3], o_rest[3]),
             ("Q_left_fingers", n_lf, o_lf), ("Q_right_fingers", n_rf, o_rf)]
    bad = [k for k, a, b in pairs if not np.array_equal(a, b, equal_nan=True)]
    overlay = "|".join(status_overlay_text(new.status_left, new.status_right, t)
                       for t in range(T))
    if overlay != _old_overlay(o_st_l, o_st_r, T):
        bad.append("overlay")

    n_fill = int((new.status_left != 0).sum() + (new.status_right != 0).sum())
    n_rest_fr = int((new.status_left == FILL_REST).sum()
                    + (new.status_right == FILL_REST).sum())
    return not bad, (f"T={T:<4} 填补帧={n_fill:<4} 静息帧={n_rest_fr:<4} "
                     f"md5(left)={_md5(new.left)} "
                     f"{'✓ 逐位一致' if not bad else f'✗ 不一致: {bad}'}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--examples", type=Path, default=None,
                    help="片段目录，默认 configs/paths.yaml 里 roots: egoinfinity_clips")
    args = ap.parse_args()
    root = args.examples or P.root("egoinfinity_clips")
    clips = sorted(p for p in root.iterdir() if (p / "hand_meta.json").exists())
    if not clips:
        print(f"{root} 下没有片段（要有 hand_meta.json）")
        return 2

    print(f"{len(clips)} 段真实片段，参照物＝迁移前 test.py 的内联实现\n")
    all_ok = True
    for clip in clips:
        ok, line = check_clip(clip)
        all_ok &= ok
        print(f"  {clip.name:<24} {line}")
    print("\n全部片段逐位一致 ✓" if all_ok else "\n有不一致 ✗")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
