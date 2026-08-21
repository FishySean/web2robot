#!/usr/bin/env bash
# 根位姿求解器对照跑：13 段 × {grid, neural}，都开臂-躯 + 双手碰撞过滤。
# 已经有 robot_sim.mp4 的目录直接跳过，可以中断后重跑接着来。
#   nohup bash scripts/dev/run_collcmp.sh > outputs/dev/collcmp_run.log 2>&1 &
set -x
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
CK=runs/m7/taskspace_v2/checkpoints/final.pt
COLL="--arm_torso_collision --dual_hand_collision"
OFFICIAL="fill_jar serve_cake sip_coffee squeeze_soap -QALmP1nHtM_678.2_682.2"
NEW="--oo8_XIuOM_799.5_809.8 --oo8_XIuOM_900.3_917.4 -0RheyDV3a0_48.6_55.3 \
-1r9yl-P-Ao_231.8_241.5 -1r9yl-P-Ao_60.4_68.4 -1r9yl-P-Ao_86.3_90.8 \
-20k07PjLTA_48.0_52.4 -2cNMO9Mm3Q_192.4_209.2"

run () {   # run <clip路径> <短名> <solver>
  out="outputs/retarget/collcmp/$2_$3"
  [ -f "$out/robot_sim.mp4" ] && { echo "SKIP $out"; return; }
  CUDA_VISIBLE_DEVICES=0 nice -n 12 scripts/s4_retarget.sh "$1" --robot m7 --ckpt $CK --seed 0 \
    --root_solver "$3" $COLL --out "$out" || echo "FAILED $2 $3"
}
# 官方自带的 5 段（examples/ 下，上游相对路径）
for c in $OFFICIAL; do
  for s in grid neural; do run "examples/$c" "$c" "$s"; done
done
# 从官方 HF 数据集新拉的 8 段（data/clips_official/，相对仓库根）
for c in $NEW; do
  for s in grid neural; do run "$ROOT/data/clips_official/$c" "$c" "$s"; done
done
echo ALL_DONE
