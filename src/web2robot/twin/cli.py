"""``python -m web2robot.twin`` —— 单独跑物体位姿跟踪，不用启动整套重定向。

用途有两个：给模块二准备输入（``object_poses.npz``），和出一段能用人眼看的可视化
（位姿三轴画在片段画面上）。重定向流水线里那条路走的是 ``test.py
--object_tracking on``，调的是同一个 :func:`web2robot.twin.track_objects`，
两边不会走岔。

::

    envs/rt_env/bin/python -m web2robot.twin \\
        --clip data/clips_official/-20k07PjLTA_48.0_52.4 \\
        --out outputs/twin/-20k07PjLTA_48.0_52.4 --viz
"""
from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from web2robot.twin.object_pose import save_object_poses
from web2robot.twin.sources import SOURCES, track_objects
from web2robot.twin.viz import overlay_object_poses


def summarize(poses) -> dict:
    """一段片段的物体位姿摘要 —— 验收表就打印这几列。"""
    rows = []
    for tr in poses.tracks:
        rows.append({
            "oid": tr.oid,
            "is_task": tr.oid == poses.task_object_id,
            "trust_frac": round(float(tr.valid.mean()), 4),
            "grasped_frac": round(tr.grasped_frac, 4),
            "travel_m": round(tr.travel, 4),
            "has_obb": tr.obb is not None,
            "has_mesh": tr.mesh_path is not None,
            "mesh_scale": round(float(tr.scale), 6),
        })
    return {"clip": poses.clip, "source": poses.source, "frame": poses.frame,
            "n_frames": poses.n_frames, "fps": poses.fps,
            "n_objects": len(poses.tracks), "task_object_id": poses.task_object_id,
            "notes": poses.notes, "objects": rows}


def _write_h264(path: Path, frames, fps: float) -> None:
    """cv2 写 mp4v，再用 ffmpeg 转 h264/yuv420p —— mpeg4 在 VSCode 里放不出来。"""
    import cv2
    if not frames:
        return
    h, w = frames[0].shape[:2]
    with tempfile.TemporaryDirectory() as td:
        tmp = str(Path(td) / "raw.mp4")
        vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), max(fps, 1.0), (w, h))
        for f in frames:
            vw.write(np.ascontiguousarray(f))
        vw.release()
        path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", tmp,
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
                        str(path)], check=True)


def _read_frames(video: Path):
    import cv2
    if not video.exists():
        return []
    cap = cv2.VideoCapture(str(video))
    out = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        out.append(f)
    cap.release()
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m web2robot.twin",
        description="物体 6D 位姿跟踪（EgoEngine §3.1 数字孪生）")
    ap.add_argument("--clip", required=True, help="片段目录")
    ap.add_argument("--out", help="输出目录（默认 outputs/twin/<clip>）")
    ap.add_argument("--source", default="official", choices=sorted(SOURCES),
                    help="位姿来源 backend（默认 official，读片段自带的孪生）")
    ap.add_argument("--viz", action="store_true",
                    help="额外出一段 object_viz.mp4，位姿画在 depth.mp4 画面上")
    ap.add_argument("--axis_len", type=float, default=0.08, help="三轴长度（米）")
    ap.add_argument("--no_others", action="store_true", help="只画任务物体")
    args = ap.parse_args(argv)

    from web2robot.paths import P
    clip = Path(args.clip).resolve()
    out_dir = (Path(args.out).resolve() if args.out
               else P.repo_root / "outputs" / "twin" / clip.name)
    out_dir = P.check_output_dir(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    poses = track_objects(clip, source=args.source)
    npz = save_object_poses(out_dir / "object_poses.npz", poses)
    info = summarize(poses)
    (out_dir / "object_poses.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2))

    print(f"Clip   : {clip}")
    print(f"Out    : {out_dir}")
    print(f"物体数 : {info['n_objects']}  任务物体 = obj{info['task_object_id']}  "
          f"帧数 = {info['n_frames']}")
    for r in info["objects"]:
        print(f"   {'*' if r['is_task'] else ' '} obj{r['oid']:<3} "
              f"trust={r['trust_frac']:.2f} grasped={r['grasped_frac']:.2f} "
              f"travel={r['travel_m']:.3f}m obb={int(r['has_obb'])} mesh={int(r['has_mesh'])}")
    for nt in info["notes"]:
        print(f"   note: {nt}")
    print(f"→ {npz.name}")

    if args.viz:
        frames = _read_frames(clip / "depth.mp4")
        if not frames:
            print("   （没有 depth.mp4，画不了可视化）")
        else:
            vid = out_dir / "object_viz.mp4"
            _write_h264(vid, overlay_object_poses(
                frames, poses, axis_len=args.axis_len,
                show_others=not args.no_others), poses.fps)
            print(f"→ {vid.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
