#!/usr/bin/env bash
# --action_refine 的"默认关闭 ⇒ 产物逐字节不变"验证。四遍同一段片段、同一个 seed：
#   base   不传新参数
#   none   --action_refine none
#   ton    --object_tracking on --action_refine none   （开了孪生但不精修）
#   mpc    --object_tracking on --action_refine mpc    （出判决，求解器未实现）
# 期望：四份的 trajectory/root_frames/metrics/robot_sim/input_viz md5 全同；
#       ton 只多 object_poses.npz；mpc 再多 action_refine.{json,npz} + hand_poses.npz。
#   bash scripts/dev/check_action_refine_bytes.sh > outputs/dev/refine_bytecheck.log 2>&1
set -x
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
CK=runs/m7/taskspace_v2/checkpoints/final.pt
CLIP="$ROOT/data/clips_official/-1r9yl-P-Ao_86.3_90.8"
run () {  # run <输出子目录> [额外参数…]
  CUDA_VISIBLE_DEVICES=3 nice -n 12 scripts/s4_retarget.sh "$CLIP" --robot m7 --ckpt $CK \
    --seed 0 --root_solver neural --out "outputs/dev/refine_bytecheck/$1" "${@:2}"
}
run base
run none --action_refine none
run ton  --object_tracking on --action_refine none
run mpc  --object_tracking on --action_refine mpc
echo ALL_DONE
