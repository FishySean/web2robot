"""比对质检输出与回归基准。

用法::

    # 1) 重跑
    PYTHONPATH=src envs/rt_env/bin/python -m web2robot.quality \
        data/videos/ tests/regression/*.mp4 --out /tmp/re/qc.jsonl --viz /tmp/re/ev
    # 2) 比对
    envs/rt_env/bin/python scripts/dev/diff_quality_run.py /tmp/re/qc.jsonl

为什么不写成 unittest：跑一次要 ~70 秒并占一块 GPU，不该混在秒级的
``python -m unittest`` 里。这是"改了质检代码之后手动跑一次"的工具。


判什么、不判什么
----------------
**不要求数值逐位相同。** KeypointRCNN 在 GPU 上不是逐位确定的（选到哪块卡、
cuDNN 挑哪个算法都会变），实测一次重跑就有单帧检测翻转：``cup_cpvH8gzUTko``
的 ``torso_rate`` 0.4828→0.4655 —— n=58，差值 0.0172 正好是 1/58，就是一帧。
要求逐位相同的话，这个测试会在没人改过代码时随机报红，很快就没人看。

所以判两件事，都比"数值相同"更贴近真正关心的问题：

1. **判决字段逐字一致**（零容忍）—— 判决、原因码、路由标签、建议路线。
   这是验收线：迁移不许改变任何一段片段的结论。
2. **每个"参与判决的信号"没有越过它的阈值**（``GATES``）—— 并且报出
   余量。数值抖一点无所谓，抖到阈值另一侧才是问题。余量小的项会被点出来，
   那才是下次真需要收紧阈值时该看的地方。

只被"记录"、不参与判决的信号（forearm_frac_med、box_frac_med、
both_wrist_rate 之类）只报漂移量，不作为不通过的理由。
"""
import argparse
import json
from pathlib import Path

# 判决/标签类字段：一个字都不能变
DECISION_FIELDS = [
    "verdict", "reasons", "needs_human_review",
    "view_class", "camera_motion", "bg_texture",
    "suggested_route", "route_rationale",
    "stages_run", "stages_skipped", "error",
]

# 参与判决的信号 -> (QCConfig 里的阈值字段, 方向)
# 方向只影响提示语，判的是"有没有换边"，两侧都算。
# 逐条都能在 pipeline.py / pose_gate.classify_framing 里找到对应的比较。
GATES = [
    ("framing.body_frame_rate",    "body_frame_rate_min", ">="),
    ("framing.body_span_est_sec",  "min_usable_sec",      ">="),
    ("hands.both_hand_rate",       "both_hand_rate_min",  ">="),
    ("hands.hands_span_est_sec",   "min_usable_sec",      ">="),
    ("hands.any_hand_rate",        "min_hand_ratio",      ">="),
    ("hands.avg_hand_size",        "min_hand_size",       ">="),
    ("hands.avg_hand_size",        "max_hand_size",       "<="),
    ("hands.trunc_ratio",          "max_trunc_ratio",     "<="),
    ("camera_motion.bg_flow",      "max_bg_flow",         "<="),
    ("texture.corner_density",     "min_corner_density",  ">="),
    ("blur.hand_lapvar_med",       "min_hand_lapvar",     ">="),
    ("hygiene.duration",           "min_duration_sec",    ">="),
]


def load(p):
    return {json.loads(l)["clip_id"]: json.loads(l) for l in open(p)}


def dig(d, dotted):
    for part in dotted.split("."):
        if not isinstance(d, dict):
            return None
        d = d.get(part)
    return d


def numeric_drift(prefix, a, b, out):
    """递归收集所有数值字段的漂移量（不判对错，只报）。"""
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            numeric_drift(f"{prefix}.{k}", a.get(k), b.get(k), out)
    elif isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        for i, (x, y) in enumerate(zip(a, b)):
            numeric_drift(f"{prefix}[{i}]", x, y, out)
    elif (isinstance(a, (int, float)) and isinstance(b, (int, float))
            and not isinstance(a, bool) and not isinstance(b, bool)):
        if a != b:
            rel = abs(a - b) / max(abs(a), abs(b), 1e-12)
            out.append((prefix, a, b, abs(a - b), rel))
    elif a != b:
        out.append((prefix, a, b, None, None))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("new", help="新跑出来的 qc.jsonl")
    ap.add_argument("--baseline",
                    default=str(Path(__file__).resolve().parents[2]
                               / "tests" / "regression" / "qc.jsonl"))
    ap.add_argument("--margin-warn", type=float, default=0.15,
                    help="余量小于阈值的这个比例时点出来提醒（默认 0.15）")
    a = ap.parse_args(argv)

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from web2robot.quality.config import QCConfig
    cfg = QCConfig()

    old, new = load(a.baseline), load(a.new)
    print(f"基准 {a.baseline}\n新跑 {a.new}\n片段数 {len(old)} → {len(new)}")

    fail = False
    if set(old) != set(new):
        print("片段集合不同！")
        print("  只在基准:", sorted(set(old) - set(new)))
        print("  只在新跑:", sorted(set(new) - set(old)))
        fail = True
    shared = sorted(set(old) & set(new))

    # ---- 1. 判决字段：零容忍 ----
    bad = [(k, f, old[k].get(f), new[k].get(f))
           for k in shared for f in DECISION_FIELDS
           if old[k].get(f) != new[k].get(f)]
    print(f"\n[1] 判决字段 {len(DECISION_FIELDS)} 项 × {len(shared)} 段（零容忍）："
          + ("全部一致" if not bad else f"{len(bad)} 处不一致"))
    for k, f, o, n in bad:
        print(f"    {k}.{f}\n      基准={o}\n      新跑={n}")
    fail |= bool(bad)

    # usable_sec 一起算在判决里（它决定 trim 剪到哪儿）
    span_bad = []
    for k in shared:
        o, n = old[k].get("usable_sec"), new[k].get("usable_sec")
        if (o is None) != (n is None) or (o is not None and abs(o - n) > 0.05):
            span_bad.append((k, o, n))
    print(f"    usable_sec（容差 0.05s，采样步长本身是秒级）："
          + ("一致" if not span_bad else f"{len(span_bad)} 处超差"))
    for k, o, n in span_bad:
        print(f"      {k}: {o} → {n}")
    fail |= bool(span_bad)

    # ---- 2. 参与判决的信号：不许换边 ----
    crossed, tight = [], []
    for k in shared:
        for path, attr, direction in GATES:
            thr = getattr(cfg, attr)
            o, n = dig(old[k].get("signals") or {}, path), \
                   dig(new[k].get("signals") or {}, path)
            if o is None or n is None:
                continue
            if (o >= thr) != (n >= thr):
                crossed.append((k, path, attr, thr, o, n))
            else:
                m = min(abs(o - thr), abs(n - thr))
                if thr and m < a.margin_warn * abs(thr):
                    tight.append((k, path, attr, thr, o, n, m))
    print(f"\n[2] 参与判决的信号 {len(GATES)} 条（不许越过阈值）："
          + ("没有一条换边" if not crossed else f"{len(crossed)} 条换边"))
    for k, path, attr, thr, o, n in crossed:
        print(f"    {k} {path} 越过 {attr}={thr}: {o} → {n}")
    fail |= bool(crossed)
    if tight:
        print(f"    余量不足 {a.margin_warn:.0%} 的（不算不通过，但下次调阈值先看这些）:")
        for k, path, attr, thr, o, n, m in tight:
            print(f"      {k} {path}={n} vs {attr}={thr}（余量 {m:.4g}）")

    # ---- 3. 数值漂移：只报 ----
    drift = []
    for k in shared:
        numeric_drift(f"{k}.signals", old[k].get("signals"),
                      new[k].get("signals"), drift)
    nonnum = [d for d in drift if d[3] is None]
    num = [d for d in drift if d[3] is not None]
    print(f"\n[3] 数值漂移（只报，不作为不通过理由）：{len(num)} 处")
    for path, o, n, d, rel in sorted(num, key=lambda x: -x[4])[:10]:
        print(f"    {path}: {o} → {n}  (差 {d:.4g}, 相对 {rel:.2%})")
    if nonnum:
        print(f"    非数值字段变化 {len(nonnum)} 处（这个要看）：")
        for path, o, n, _, _ in nonnum:
            print(f"      {path}: {o} → {n}")
        fail = True

    print("\n" + ("不通过" if fail else
                  "通过：判决逐字一致，没有信号越过阈值，剩下的是 GPU 非确定性抖动"))
    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
