#!/usr/bin/env bash
# 碰撞校准用的**过滤前**轨迹：三段代表片段 × grid，不开任何碰撞过滤。
#
# 为什么要单独跑这一批：ArmTorsoFilter 是 IK 之后的纯后处理（输入 = q + 手指），
# 所以只要拿到"过滤前的 q"，扫参数就能**离线**做，不必每换一组参数就重跑一次
# 15~24 分钟的 grid 根位姿搜索。`outputs/retarget/collcmp/` 里那批是**过滤后**的，
# 拿它当输入等于在已经修过的姿态上再修一遍，会得出错的结论。
#
# 选这三段的理由（见 outputs/dev/collcmp_table/table.md 的混淆表）：
#   sip_coffee            代理报 169 帧、网格只 2 帧，且代理最深 2.40cm < enter_thresh
#                         → "过滤器全程没动"的极端样本
#   fill_jar              21 帧漏报里有 5 帧在这段，代理/网格都深（5.93 / 6.07cm）
#                         → 唯一同时有漏报和深穿的样本
#   -2cNMO9Mm3Q_192.4_209.2  漏 6 / 误 38，ρ<0.4 占 79%
#                         → 贴身姿态最多的样本
#
#   nohup bash scripts/dev/run_collcal_prefilter.sh > outputs/dev/collcal_prefilter.log 2>&1 &
set -x
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
CK=runs/m7/taskspace_v2/checkpoints/final.pt

run () {   # run <clip路径> <短名>
  out="outputs/dev/collcal/prefilter/$2"
  [ -f "$out/trajectory.npz" ] && { echo "SKIP $out"; return; }
  CUDA_VISIBLE_DEVICES=3 nice -n 12 scripts/s4_retarget.sh "$1" --robot m7 --ckpt $CK \
    --seed 0 --root_solver grid --out "$out" || echo "FAILED $2"
}
run "examples/sip_coffee" sip_coffee
run "examples/fill_jar"   fill_jar
run "$ROOT/data/clips_official/-2cNMO9Mm3Q_192.4_209.2" -2cNMO9Mm3Q_192.4_209.2
echo ALL_DONE
