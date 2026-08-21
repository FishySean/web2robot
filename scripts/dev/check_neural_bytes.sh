#!/usr/bin/env bash
# 标定后的「neural 路线逐字节不变」验证。
#
# 为什么要真跑一遍而不是看代码：presets.NEURAL 是空字典，理论上 ArmTorsoFilter 的
# 构造参数一个也没变；但"理论上没变"和"产物没变"是两件事（默认值、参数顺序、
# 预设注入点任何一处写错都会悄悄改轨迹）。neural 路线上的 13 段表、demo 视频、
# README 素材全是旧参数出的，所以这条得有字节级凭据。
#
# base = 不传预设参数（旧行为）  auto = --atf_preset auto（走 presets.NEURAL）
# 期望 trajectory.npz / metrics.npz / robot_sim.mp4 的 md5 全同。
#   bash scripts/dev/check_neural_bytes.sh > outputs/dev/neural_bytecheck.log 2>&1
set -x
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
CK=runs/m7/taskspace_v2/checkpoints/final.pt
CLIP="$ROOT/data/clips_official/-1r9yl-P-Ao_86.3_90.8"
COLL="--arm_torso_collision --dual_hand_collision"
run () {  # run <输出子目录> [额外参数…]
  CUDA_VISIBLE_DEVICES=3 nice -n 12 scripts/s4_retarget.sh "$CLIP" --robot m7 --ckpt $CK \
    --seed 0 --root_solver neural $COLL --out "outputs/dev/neural_bytecheck/$1" "${@:2}"
}
run base
run auto --atf_preset auto
set +x
cd outputs/dev/neural_bytecheck
md5sum base/trajectory.npz auto/trajectory.npz \
       base/metrics.npz    auto/metrics.npz \
       base/robot_sim.mp4  auto/robot_sim.mp4
for f in trajectory.npz metrics.npz robot_sim.mp4; do
  cmp -s "base/$f" "auto/$f" && echo "SAME $f" || echo "DIFF $f  <-- neural 路线被动到了"
done
echo ALL_DONE
