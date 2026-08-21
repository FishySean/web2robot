#!/usr/bin/env bash
# 臂-躯碰撞过滤「调参前 / 调参后」验收对照：13 段 grid 路线各重跑一遍。
#
# 为什么只重跑 grid：标定只动了 grid 那一组余量（web2robot/collision/presets.py 里
# neural 是空字典），neural 的轨迹按构造与旧跑逐位相同 —— 所以 <片段>_neural 目录
# 直接软链到旧跑，省下 13 段生成模型 + IK 的时间。链接前会先核对旧跑存在。
# 这个"逐位相同"不是口头承诺，scripts/dev/check_neural_bytes.sh 会真去比字节。
#
# 底座求解是确定性的（--seed 0，grid 是穷举），所以两次跑的根位姿/IK 完全一致，
# 差别只有碰撞过滤那一步 —— 这才是干净的 A/B。
#
#   nohup bash scripts/dev/run_collcal_ab.sh > outputs/dev/collcal_ab.log 2>&1 &
# 完事后出表：
#   scripts/dev/m7_tool.sh collcmp_table.py --root outputs/retarget/collcmp_cal \
#     --out outputs/dev/collcal_ab_table
set -x
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
CK=runs/m7/taskspace_v2/checkpoints/final.pt
COLL="--arm_torso_collision --dual_hand_collision"
OLD=outputs/retarget/collcmp          # 调参前（legacy 参数）的 13×2 跑
NEW=outputs/retarget/collcmp_cal      # 调参后（grid 预设）
OFFICIAL="fill_jar serve_cake sip_coffee squeeze_soap -QALmP1nHtM_678.2_682.2"
NEWCLIPS="--oo8_XIuOM_799.5_809.8 --oo8_XIuOM_900.3_917.4 -0RheyDV3a0_48.6_55.3 \
-1r9yl-P-Ao_231.8_241.5 -1r9yl-P-Ao_60.4_68.4 -1r9yl-P-Ao_86.3_90.8 \
-20k07PjLTA_48.0_52.4 -2cNMO9Mm3Q_192.4_209.2"

mkdir -p "$NEW"
run () {   # run <clip路径> <短名>
  out="$NEW/$2_grid"
  [ -f "$out/robot_sim.mp4" ] || \
    CUDA_VISIBLE_DEVICES=0 nice -n 12 scripts/s4_retarget.sh "$1" --robot m7 --ckpt $CK \
      --seed 0 --root_solver grid --atf_preset auto $COLL --out "$out" \
      || echo "FAILED $2"
  # neural 侧：复用旧跑（collcmp_table.py 要成对的 _neural/_grid 目录才认）
  [ -d "$OLD/$2_neural" ] || { echo "MISSING $OLD/$2_neural"; return; }
  [ -e "$NEW/$2_neural" ] || ln -s "$ROOT/$OLD/$2_neural" "$NEW/$2_neural"
}
for c in $OFFICIAL;  do run "examples/$c" "$c"; done
for c in $NEWCLIPS;  do run "$ROOT/data/clips_official/$c" "$c"; done
echo ALL_DONE
