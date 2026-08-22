"""CLI: python -m web2robot.quality <videos-or-dirs> --out results.jsonl

Writes one JSON object per clip (machine-readable, for step 2) plus a markdown
summary (human-readable, for the spot-check the metric!=visual rule requires).
"""
import argparse
import glob
import json
import os
import sys
import time

from .config import QCConfig, GATE_MODES, ROUTING_MODES
from .pipeline import diagnose_many
from .schema import Verdict

VIDEO_EXT = (".mp4", ".mov", ".mkv", ".webm", ".avi", ".MP4", ".MOV")


def fingerprint(path: str, chunk: int = 1 << 20) -> str:
    """Cheap content fingerprint: size + first and last 1MB.

    Scraped sets are full of re-uploads of the same file under different names --
    measured on the 10 files in hand2robot/videos/, 3 are byte-identical copies
    of others (only 7 distinct clips). Path/realpath dedup cannot see that, and
    duplicates would silently triple the apparent sample size and any pass-rate
    computed from it.

    Head+tail rather than a full hash so it stays O(1) per file at scrape scale;
    two distinct videos sharing size AND both 1MB ends is not a realistic
    collision for container formats that carry per-file metadata up front.
    """
    import hashlib
    h = hashlib.md5()
    n = os.path.getsize(path)
    h.update(str(n).encode())
    with open(path, "rb") as fh:
        h.update(fh.read(chunk))
        if n > 2 * chunk:
            fh.seek(-chunk, os.SEEK_END)
            h.update(fh.read(chunk))
    return h.hexdigest()


def expand(inputs, dedup_content: bool = True):
    out = []
    for p in inputs:
        if os.path.isdir(p):
            for e in VIDEO_EXT:
                out += sorted(glob.glob(os.path.join(p, f"*{e}")))
        elif any(c in p for c in "*?["):
            out += sorted(glob.glob(p))
        else:
            out.append(p)
    seen, uniq = set(), []
    for p in out:
        rp = os.path.realpath(p)
        if rp not in seen:
            seen.add(rp)
            uniq.append(p)
    if not dedup_content:
        return uniq, {}
    by_fp, kept, dupes = {}, [], {}
    for p in uniq:
        try:
            fp = fingerprint(p)
        except OSError:
            kept.append(p)
            continue
        if fp in by_fp:
            dupes.setdefault(os.path.basename(by_fp[fp]), []).append(
                os.path.basename(p))
        else:
            by_fp[fp] = p
            kept.append(p)
    return kept, dupes


def write_markdown(reps, path, cfg, elapsed, dupes=None):
    order = {Verdict.ACCEPT.value: 0, Verdict.TRIM.value: 1, Verdict.DEFER.value: 2,
             Verdict.UNKNOWN.value: 3, Verdict.REJECT.value: 4,
             Verdict.SKIPPED.value: 5}
    reps = sorted(reps, key=lambda r: (order.get(r.verdict, 9), r.clip_id))
    n = len(reps)
    counts = {}
    for r in reps:
        counts[r.verdict] = counts.get(r.verdict, 0) + 1
    L = ["# Stage-1 video quality report", "",
         f"- clips: **{n}**  |  wall time: {elapsed:.1f}s "
         f"({elapsed / max(1, n):.1f}s per clip)",
         "- verdicts: " + ", ".join(f"**{k}** {v}" for k, v in sorted(counts.items())),
         "", "> DEFER is not a rejection: hands-only framing cannot be separated "
         "from egocentric footage by body pose alone, and first-person video is a "
         "valid route in step 2.", ""]
    if cfg.quality_gate == "skip":
        L += ["> **`--quality_gate skip`：这份报告没有任何测量。** 判决一律 "
              "`skipped`（不是 `accept` —— 没量过的东西不能算通过），下面的 "
              "framing / hand 明细表是空的。", ""]
    elif cfg.routing == "skip":
        L += ["> **`--routing skip`：质检照旧全跑，但没有算建议路线。** "
              "`route` 一列全是 `-`，signals 里的信号一条不少。", ""]
    if dupes:
        L += ["## Duplicates dropped", "",
              "Byte-identical re-uploads under different names. Reported rather "
              "than silently skipped -- they inflate the apparent sample size and "
              "every rate computed from it.", ""]
        for k, v in sorted(dupes.items()):
            L.append(f"- `{k}` == " + ", ".join(f"`{x}`" for x in v))
        L.append("")
    L += ["| clip | verdict | view | camera | texture | route | usable | reasons |",
          "|---|---|---|---|---|---|---|---|"]
    for r in reps:
        L.append(f"| {r.clip_id} | **{r.verdict}** | {r.view_class} | "
                 f"{r.camera_motion} | {r.bg_texture} | {r.suggested_route or '-'} | "
                 f"{('%.1fs' % r.usable_sec) if r.usable_sec else '-'} | "
                 f"{', '.join(r.reasons) or '-'} |")
    L += ["", "## Framing detail (body-pose gate)", "",
          "| clip | n | step | body_rate | body_span | torso | head | forearm | elbow@btm | wrist_rate† |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for r in reps:
        f = r.signals.get("framing")
        if not f:
            continue
        L.append(f"| {r.clip_id} | {f['n_sampled']} | "
                 f"{f['sample_step_sec']}s | "
                 f"**{f['body_frame_rate']:.2f}** | {f['body_span_est_sec']:.0f}s | "
                 f"{f['torso_rate']:.2f} | "
                 f"{f['head_rate']:.2f} | {f['forearm_frac_med']:.3f} | "
                 f"{f['elbow_at_bottom_rate']:.2f} | {f['both_wrist_rate']:.2f} |")
    L += ["", "`head` is reported but never gates: a clip whose camera missed the "
          "head is still usable.",
          "", "† `wrist_rate` (both wrist KEYPOINTS visible) is reported and "
          "**decides nothing**. Measured inverted on hands-only footage: paired "
          "controls scored 0.21 for two hands vs 0.25 for one hand at det 0.7, and "
          "the inversion held at 0.3/0.1/0.05. That boundary moved to the hand "
          "detector below.",
          "", "## Hand detail (hand gate — decides hands_only vs no_stable_hands)", "",
          "| clip | n | both_hand | hands_span | any_hand | mean_n | avg_size | trunc | dup |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in reps:
        h = r.signals.get("hands")
        if not h:
            continue
        if not h.get("available"):
            L.append(f"| {r.clip_id} | - | - | - | - | - | - | - | - |  <!-- detector "
                     f"unavailable -> unknown, not rejected -->")
            continue
        L.append(f"| {r.clip_id} | {h['n_sampled']} | "
                 f"**{h['both_hand_rate']:.2f}** | "
                 f"{(h['hands_span_est_sec'] or 0):.0f}s | "
                 f"{h['any_hand_rate']:.2f} | {h['mean_n_hands']:.2f} | "
                 f"{h['avg_hand_size']:.4f} | {h['trunc_ratio']:.2f} | "
                 f"{h['n_merged']} |")
    L += ["", "`any_hand` is the official `hand_ratio` (>=1 hand) kept at its "
          "official definition for comparability; it **cannot** tell one hand from "
          "two (0.58 vs 0.54 on the paired controls), which is why `both_hand` "
          "exists. `avg_size` / `trunc` are recorded, never fatal. `dup` counts "
          "duplicate boxes merged away (same hand detected twice) -- without that "
          "merge a single pair of hands can report as three.",
          "", "A class is reached by EITHER a high whole-file rate OR a long "
          "enough contiguous span. `*_span` is interpolated between samples "
          "(`step` apart), not measured frame by frame.", "",
          "## Config", "", "```json",
          json.dumps(_config_record(cfg), indent=2, default=str), "```"]
    with open(path, "w") as fh:
        fh.write("\n".join(L) + "\n")


def _config_record(cfg) -> dict:
    """配置快照。``hand_weights=None`` 时补上实际解析到的权重路径 ——
    报告是留档用的，"用了哪个权重"不能因为改成从 paths.yaml 查而丢掉。"""
    from . import hand_gate
    d = dict(cfg.to_dict())
    if d.get("hand_weights") is None:
        d["hand_weights"] = hand_gate.find_weights(None)
    return d


def main(argv=None):
    ap = argparse.ArgumentParser("web2robot.quality", description=__doc__)
    ap.add_argument("inputs", nargs="+", help="video files, globs, or directories")
    ap.add_argument("--out", default="qc_results.jsonl")
    ap.add_argument("--md", default=None, help="markdown summary (default: --out with .md)")
    ap.add_argument("--source", default="scraped",
                    choices=["scraped", "official", "selfcap"],
                    help="provenance tag; determines which criteria are evaluable")
    ap.add_argument("--n-frames", type=int, default=None)
    ap.add_argument("--hand-weights", default=None,
                    help="YOLO hand detector .pt (default: the on-disk WiLoR/"
                         "official detector.pt)")
    ap.add_argument("--device", default=None, help="cuda:N | cpu | auto")
    ap.add_argument("--no-early-exit", action="store_true",
                    help="compute every signal even on rejected clips (calibration)")
    ap.add_argument("--keep-duplicates", action="store_true",
                    help="do not drop byte-identical re-uploads")
    ap.add_argument("--per-frame", action="store_true", help="keep per-frame rows in JSONL")
    ap.add_argument("--viz", default=None, metavar="DIR",
                    help="write annotated evidence frames + a contact sheet here")
    # 下划线写法是主入口（retarget 那边的 CLI 全是下划线），连字符留成别名，
    # 因为本文件其余的选项都是连字符，两种都能敲比让人猜要好。
    ap.add_argument("--quality_gate", "--quality-gate", dest="quality_gate",
                    default="builtin", choices=list(GATE_MODES),
                    help="builtin=走本模块的三档判定（默认，行为不变）；"
                         "skip=一个 stage 都不跑，片段原样往下传")
    ap.add_argument("--routing", default="builtin", choices=list(ROUTING_MODES),
                    help="builtin=照旧给出 suggested_route（默认）；"
                         "skip=不算路由标签，质检信号照样全跑")
    a = ap.parse_args(argv)

    cfg = QCConfig(quality_gate=a.quality_gate, routing=a.routing)
    if a.n_frames:
        cfg.n_frames = a.n_frames
    if a.hand_weights:
        cfg.hand_weights = a.hand_weights
    if a.device:
        cfg.device = a.device
    if a.no_early_exit:
        cfg.early_exit = False

    paths, dupes = expand(a.inputs, dedup_content=not a.keep_duplicates)
    if not paths:
        print("no videos found", file=sys.stderr)
        return 2
    for k, v in sorted(dupes.items()):
        print(f"[quality] duplicate of {k}: {', '.join(v)} -- skipped", flush=True)

    # Say out loud whether the hand detector loaded. Without it the hands-only
    # boundary is unmeasurable and every such clip comes out UNKNOWN -- that must
    # be visible up front, not inferred afterwards from a wall of 'unknown'.
    # quality_gate=skip 时连模型都不加载：跳过就该是零 GPU、零权重依赖，否则
    # "跳过"只省了推理时间，在没装权重的机器上照样报警。
    if cfg.quality_gate == "skip":
        print("[quality] --quality_gate skip：不跑任何 stage，"
              f"{len(paths)} 段原样往下传（判决 skipped，不是 accept）", flush=True)
    else:
        from . import hand_gate
        hm, hdev = hand_gate.get_hand_model(cfg.hand_weights, cfg.device)
        if hm is None:
            print("[quality] WARNING hand detector unavailable "
                  f"(weights={hand_gate.find_weights(cfg.hand_weights)}) -- "
                  "hands-only clips will be reported UNKNOWN, not rejected",
                  file=sys.stderr)
        else:
            print(f"[quality] hand detector on {hdev}: "
                  f"{hand_gate.find_weights(cfg.hand_weights)}", flush=True)
    print(f"[quality] {len(paths)} clips, n_frames={cfg.n_frames}, "
          f"early_exit={cfg.early_exit}, quality_gate={cfg.quality_gate}, "
          f"routing={cfg.routing}", flush=True)

    t0 = time.time()

    def prog(i, n, p):
        print(f"  [{i + 1}/{n}] {os.path.basename(p)}", flush=True)

    viz = None
    if a.viz:
        from .viz import Visualizer
        viz = Visualizer(a.viz, cfg)

    reps = diagnose_many(paths, cfg, a.source, a.per_frame, prog,
                         on_pose=(viz.on_pose if viz else None))
    elapsed = time.time() - t0

    with open(a.out, "w") as fh:
        for r in reps:
            fh.write(json.dumps(r.to_dict(with_frames=a.per_frame),
                                ensure_ascii=False) + "\n")
    md = a.md or os.path.splitext(a.out)[0] + ".md"
    write_markdown(reps, md, cfg, elapsed, dupes)

    print(f"\n[quality] {elapsed:.1f}s -> {a.out}  /  {md}")
    if viz:
        sheet = viz.save()
        if sheet:
            print(f"[quality] evidence: {sheet}")
    for r in reps:
        print(f"  {r.verdict:7s} {r.view_class:19s} {r.clip_id}"
              f"   {', '.join(r.reasons) or ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
