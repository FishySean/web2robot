"""Stage-1 orchestration.

Order and early exit
--------------------
The pose gate runs first, and when it rejects, the remaining stages are skipped.
This is the cheap-early-exit the pipeline needs: optical flow over 24 frame pairs
costs more than the pose pass, and there is no point measuring the background
texture of a clip that shows one hand poking in from an edge.

That is not in conflict with the earlier "compute all signals, do not
short-circuit" decision. That decision was about ATTRIBUTION -- when a clip is
rejected, list every failing check rather than the first one, because the
framing and view criteria overlap and reporting only the first misattributes the
cause. So: within a stage, evaluate everything; across stages, exit early once
the verdict can no longer change. `QCConfig.early_exit=False` disables the skip
for calibration runs.

Verdicts
--------
  ACCEPT  third-person body framing, no fatal cuts   -> routed now
  TRIM    ditto, but a cut splits it                 -> routed, cropped to span
  DEFER   hands-only framing                         -> step 2 classifies the view
  REJECT  no stable hands, or nothing left after cuts
  UNKNOWN a signal needed for the verdict failed     -> human look

DEFER is not a pass. It is the honest statement that body framing alone cannot
separate egocentric footage from an overhead third-person close-up, and that
first-person clips must survive stage 1 because they are a valid step-2 route.

两个开关（2026-08-21）
--------------------
``cfg.quality_gate`` / ``cfg.routing``，取值见 ``config.GATE_MODES`` /
``ROUTING_MODES``，默认都是 ``builtin`` = 上面描述的现状。``skip`` 只是不做，
不改任何一条判定规则：

  quality_gate=skip  一个 stage 都不跑（连 ffprobe 都不开），判决 ``skipped``，
                     片段原样往下传。routing 也就无从可算 —— 它的三个输入
                     (view_class / camera_motion / bg_texture) 全部没有测量。
  routing=skip       质检照旧全跑，只是不调 ``labels.suggest``。signals 里的
                     信号一条不少，下游要自己判还有料。

两个开关独立：公司可能只替换其中一个。
"""
from typing import List, Optional, Tuple
import os

from .config import QCConfig
from .schema import ClipReport, Verdict, ViewClass, CameraMotion
from ..common import video_io
from ..routing import labels
from . import pose_gate, hand_gate, motion, appearance

ALL_STAGES = ("hygiene", "pose_gate", "hand_gate", "shot_cuts",
              "camera_motion", "texture", "blur")
"""每个 stage 的名字，只用于 quality_gate=skip 时如实写出"哪些没跑"。
顺序跟 diagnose_clip 里 stages_run.append 的顺序一致。"""


def _skipped_report(rep: ClipReport, cfg: QCConfig) -> ClipReport:
    """quality_gate=skip 的返回值：说清楚"什么都没做"，不假装任何结论。"""
    rep.verdict = Verdict.SKIPPED.value
    rep.add_reason("quality_gate_skipped")
    rep.stages_skipped += list(ALL_STAGES)
    # needs_human_review 保持 False：这不是"要人看一眼"，是"这一步整个交给别人"。
    # usable_span 留空 = 不裁剪，下游拿整段（原样往下传的字面意思）。
    rep.suggested_route = None
    rep.route_rationale = [
        "quality_gate=skip：路由的三个输入（view_class / camera_motion / "
        "bg_texture）都没有测量，给不出建议路线"]
    if cfg.routing == "skip":
        rep.route_rationale.append("routing=skip：路由本身也关了")
    return rep


def _route(cfg: QCConfig, view_class: str, camera_motion: str,
           bg_texture: str, duration_ok: bool) -> Tuple[Optional[str], List[str]]:
    """路由开关的唯一落点。

    ``labels.suggest`` 在本文件里只允许出现在这里 —— 之前有两个调用点（提前
    退出那条和正常判决那条），开关必须两条都管住，漏一条就会出现"跳过了路由
    但被拒的片段还是带着路线"这种自相矛盾的输出。单测钉住了这个唯一性。
    """
    if cfg.routing == "skip":
        return None, ["routing=skip：没有计算建议路线（质检信号照样在 signals 里）"]
    return labels.suggest(view_class, camera_motion, bg_texture, duration_ok)


def diagnose_clip(path: str, cfg: Optional[QCConfig] = None,
                  source: str = "scraped", keep_frames: bool = False,
                  clip_id: Optional[str] = None, on_pose=None) -> ClipReport:
    """Diagnose one clip.

    on_pose: optional callback(report, frames, pose_frames) invoked right after
    the pose stage, so the visualiser can reuse the decoded frames and pose
    results instead of paying for a second forward pass.
    """
    cfg = cfg or QCConfig()
    rep = ClipReport(clip_id=clip_id or os.path.basename(path),
                     path=os.path.abspath(path), source=source)

    if cfg.quality_gate == "skip":
        return _skipped_report(rep, cfg)
    # routing=skip 故意**不往 reasons 里写标记**：reasons 的契约是"所有没通过的
    # 检查，最要紧的在前"，塞一条记账用的码进去会把真正的原因挤到第二位（实测
    # 过一版：`['routing_skipped', 'no_person']`，报告里第一眼看到的是无关的那条）。
    # 关了路由这件事写在 route_rationale 里 —— 路由的解释本来就该在那儿。

    # ---------------- stage 0: hygiene ----------------
    info = video_io.probe(path)
    rep.stages_run.append("hygiene")
    if not info.ok:
        rep.verdict = Verdict.REJECT.value
        rep.add_reason("decode_error")
        rep.error = info.error
        return rep
    rep.signals["hygiene"] = dict(width=info.width, height=info.height,
                                  fps=round(info.fps, 3), n_frames=info.n_frames,
                                  duration=round(info.duration, 2))
    duration_ok = info.duration >= cfg.min_duration_sec
    if not duration_ok:
        rep.add_reason("too_short")
    if min(info.width, info.height) < cfg.min_side_px:
        rep.add_reason("too_small")

    # ---------------- stage 1: framing (pose) ----------------
    n_sample = video_io.plan_n_frames(info.duration, cfg)
    frames = video_io.sample_frames(path, n_sample, cfg.sample_lo, cfg.sample_hi)
    if not frames:
        rep.verdict = Verdict.UNKNOWN.value
        rep.add_reason("decode_error")
        rep.needs_human_review = True
        return rep
    pfs = pose_gate.run_pose(frames, cfg)
    agg = pose_gate.aggregate(pfs, info.fps, cfg)
    rep.stages_run.append("pose_gate")
    rep.signals["framing"] = agg

    # ---------------- stage 1b: hands ----------------
    # Same decoded frames, second model. The body gate owns THIRD_PERSON_BODY;
    # only a hand detector can draw the hands-only / unusable boundary (its
    # wrist keypoints measured INVERTED on hands-only footage -- see
    # hand_gate's module docstring).
    hfs = hand_gate.detect_hands(frames, cfg)
    hands = hand_gate.aggregate_hands(hfs, info.fps, cfg)
    rep.signals["hands"] = hands
    if hands["available"]:
        rep.stages_run.append("hand_gate")
    else:
        rep.stages_skipped.append("hand_gate")
        rep.needs_human_review = True

    view_class, framing_reasons = pose_gate.classify_framing(agg, cfg, hands)
    rep.view_class = view_class
    for r in framing_reasons:
        rep.add_reason(r)
    # Size checks are RECORDED, never fatal: the official max_hand_size=0.40 is a
    # 100M-scale coarse filter, ~7.6x looser than our own measurement wanted.
    if hands["available"] and hands["avg_hand_size"]:
        if hands["avg_hand_size"] < cfg.min_hand_size:
            rep.add_reason("hand_too_small")
        elif hands["avg_hand_size"] > cfg.max_hand_size:
            rep.add_reason("hand_too_large")
    if keep_frames:
        by_idx = {h.frame_idx: h for h in (hfs or [])}
        rep.per_frame = [dict(frame=p.frame_idx, found=p.found,
                              det=round(p.det, 3), n_wrist=p.n_wrist,
                              n_elbow=p.n_elbow, n_torso=p.n_torso,
                              n_head=p.n_head, framing_ok=p.framing_ok,
                              n_hands=(by_idx[p.frame_idx].n_hands
                                       if p.frame_idx in by_idx else None),
                              why=p.why()) for p in pfs]

    fatal_framing = view_class == ViewClass.NO_STABLE_HANDS.value
    if on_pose is not None:
        on_pose(rep, frames, pfs, hfs)
    if fatal_framing and cfg.early_exit:
        rep.verdict = Verdict.REJECT.value
        rep.stages_skipped += ["shot_cuts", "camera_motion", "texture", "blur"]
        rep.suggested_route, rep.route_rationale = _route(
            cfg, view_class, CameraMotion.UNKNOWN.value, "unknown", duration_ok)
        return rep

    # ---------------- stage 2: shot cuts ----------------
    cuts = motion.detect_shot_cuts(path, cfg.scene_threshold,
                                   ignore_before=cfg.cut_ignore_before_sec)
    rep.stages_run.append("shot_cuts")
    if cuts is None:
        rep.signals["cuts"] = dict(n_cuts=None, cut_times=None)
        rep.needs_human_review = True
        span = (0.0, info.duration)
    else:
        span = motion.longest_subseg(0.0, info.duration, cuts)
        rep.signals["cuts"] = dict(n_cuts=len(cuts),
                                   cut_times=[round(t, 2) for t in cuts[:64]],
                                   longest_span=[round(span[0], 2), round(span[1], 2)])
    if cuts is not None and (span[1] - span[0]) < cfg.min_subseg_sec:
        rep.add_reason("cuts_too_short")

    # Trim to where the usable footage actually is: the cut-free span says where
    # the editing is continuous, the pose/hand span says where the hands are. Only
    # their intersection satisfies both, so that is what downstream gets.
    # For hands-only clips the span comes from the HAND detector, matching the
    # instrument that decided the class in the first place.
    pose_span = (agg["body_span_est"] if view_class == ViewClass.THIRD_PERSON_BODY.value
                 else hands["hands_span_est"])
    final = span
    if pose_span:
        lo, hi = max(span[0], pose_span[0]), min(span[1], pose_span[1])
        if hi - lo >= min(cfg.min_usable_sec, agg["sample_step_sec"] or 0.0):
            final = (lo, hi)
        else:
            # the hands are outside the continuous shot -- neither trim is right
            rep.add_reason("span_conflict")
            rep.needs_human_review = True
    rep.usable_span = [round(final[0], 2), round(final[1], 2)]
    rep.usable_sec = round(final[1] - final[0], 2)
    rep.signals["cuts"]["pose_span"] = pose_span

    # ---------------- stage 3: camera motion ----------------
    pairs = video_io.sample_pairs(path, cfg.flow_max_pairs, cfg.sample_lo, cfg.sample_hi)
    cm = motion.camera_motion(pairs, cfg)
    rep.stages_run.append("camera_motion")
    rep.signals["camera_motion"] = cm
    if cm["bg_flow"] is None:
        rep.camera_motion = CameraMotion.UNKNOWN.value
        rep.needs_human_review = True
    else:
        rep.camera_motion = (CameraMotion.MOVING.value
                             if cm["bg_flow"] > cfg.max_bg_flow
                             else CameraMotion.STATIC.value)

    # ---------------- stage 4: appearance ----------------
    tex = appearance.background_texture(frames, [p.box for p in pfs])
    blur = appearance.hand_blur(frames, pfs)
    rep.stages_run += ["texture", "blur"]
    rep.signals["texture"] = tex
    rep.signals["blur"] = blur
    if tex["corner_density"] is None:
        rep.bg_texture = "unknown"
        rep.needs_human_review = True
    else:
        rep.bg_texture = ("rich" if tex["corner_density"] >= cfg.min_corner_density
                          else "poor")
        if rep.bg_texture == "poor":
            rep.add_reason("texture_poor")     # routing input, not fatal
    if blur["hand_lapvar_med"] is not None and \
            blur["hand_lapvar_med"] < cfg.min_hand_lapvar:
        rep.add_reason("blurry")

    # ---------------- verdict ----------------
    rep.suggested_route, rep.route_rationale = _route(
        cfg, view_class, rep.camera_motion, rep.bg_texture, duration_ok)

    fatal = {"no_stable_hands", "no_person", "cuts_too_short", "decode_error"}
    if fatal & set(rep.reasons):
        rep.verdict = Verdict.REJECT.value
    elif view_class == ViewClass.UNKNOWN.value:
        # the hands-only boundary could not be measured at all -- do not guess it
        # with the falsified wrist statistic, and do not reject (decision 3)
        rep.verdict = Verdict.UNKNOWN.value
        rep.needs_human_review = True
    elif view_class == ViewClass.HANDS_ONLY.value:
        rep.verdict = Verdict.DEFER.value
    elif cuts is not None and len(cuts) > 0:
        rep.verdict = Verdict.TRIM.value
    elif rep.needs_human_review:
        rep.verdict = Verdict.UNKNOWN.value
    else:
        rep.verdict = Verdict.ACCEPT.value
    return rep


def diagnose_many(paths: List[str], cfg: Optional[QCConfig] = None,
                  source: str = "scraped", keep_frames: bool = False,
                  progress=None, on_pose=None) -> List[ClipReport]:
    cfg = cfg or QCConfig()
    reps = []
    for i, p in enumerate(paths):
        if progress:
            progress(i, len(paths), p)
        try:
            reps.append(diagnose_clip(p, cfg, source, keep_frames, on_pose=on_pose))
        except Exception as e:                                  # noqa: BLE001
            r = ClipReport(clip_id=os.path.basename(p), path=os.path.abspath(p),
                           source=source, verdict=Verdict.UNKNOWN.value,
                           needs_human_review=True, error=f"{type(e).__name__}: {e}")
            reps.append(r)
    return reps
