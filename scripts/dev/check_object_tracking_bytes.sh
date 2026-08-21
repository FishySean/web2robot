#!/usr/bin/env bash
# --object_tracking 的"默认关闭 ⇒ 产物逐字节不变"验证：同一段片段跑三遍
#   base = 不传参数    off = --object_tracking off    on = --object_tracking on
# 期望 base/off/on 的 trajectory.npz / root_frames.npz / metrics.npz /
# robot_sim.mp4 / input_viz.mp4 md5 全同，on 只多出一个 object_poses.npz。
#   bash scripts/dev/check_object_tracking_bytes.sh > outputs/dev/objtrack_bytecheck.log 2>&1
set -x
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
CK=runs/m7/taskspace_v2/checkpoints/final.pt
CLIP="$ROOT/data/clips_official/-1r9yl-P-Ao_86.3_90.8"
run () {  # run <输出子目录> [额外参数…]
  CUDA_VISIBLE_DEVICES=3 nice -n 12 scripts/s4_retarget.sh "$CLIP" --robot m7 --ckpt $CK \
    --seed 0 --root_solver neural --out "outputs/dev/objtrack_bytecheck/$1" "${@:2}"
}
run base
run off --object_tracking off
run on  --object_tracking on
echo ALL_DONE
