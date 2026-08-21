#!/usr/bin/env bash
# 「机器人参数搬进 yaml + 新增两层坏帧粒度 ⇒ 产物逐字节不变」验证。
#
# 为什么必须真跑：这两件事在代码上都"应该"不改行为 —— yaml 里的数和原来写死的数
# 逐位相同，新增两层默认不开。但"应该"和"没变"是两件事：
#   * 参数搬家踩过的坑是**类型**（yaml 出 list、原来是 tuple/ndarray；1e-3 被 PyYAML
#     读成字符串），以及**求值时机**（构造签名的默认值在 import 时算，改成读文件就
#     多了一次 I/O 和一次 import，import 有副作用就可能动到随机数流）；
#   * 新增两层动了 scripts/test.py 的 argparse 和调用点，argparse 顺序变了都可能
#     影响随机数流（见 docs/VERIFICATION.md 那条"只有比 md5 才发现"）。
#
# 三遍：
#   base  = 什么新参数都不传（= 用户今天的跑法）
#   tiers = --bad_frame_tiers episode,segment,frame（三层全开）
#   grid  = --root_solver grid --atf_preset auto（让 yaml 里那组标定参数真的送进 MuJoCo）
#
# 判据：
#   1. base 和 **改动之前**留下的参照物 outputs/dev/neural_bytecheck/base/ 逐字节相同
#      （那份是 2026-08-20 A1 标定验证跑的，时间戳早于本次全部改动）
#   2. tiers 和 base 的原有产物逐字节相同 —— 新增两层"只看不动"，只许多出
#      bad_frame_tiers.json 一个文件
#   3. base **不该**多出 bad_frame_tiers.json（多一个文件也算产物变了）
#   4. grid 的日志里，ArmTorsoFilter 打印的盒/门槛/余量就是 yaml 里那组标定值
#
#   bash scripts/dev/check_tiers_yaml_bytes.sh > outputs/dev/tiers_yaml_bytecheck.log 2>&1
set -x
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
CK=runs/m7/taskspace_v2/checkpoints/final.pt
CLIP="$ROOT/data/clips_official/-1r9yl-P-Ao_86.3_90.8"
OUT="outputs/dev/tiers_yaml_bytecheck"
REF="outputs/dev/neural_bytecheck/base"      # 本次改动之前留下的参照物
COLL="--arm_torso_collision --dual_hand_collision"
mkdir -p "$OUT"
run () {  # run <输出子目录> [额外参数…]
  CUDA_VISIBLE_DEVICES=3 nice -n 12 scripts/s4_retarget.sh "$CLIP" --robot m7 --ckpt $CK \
    --seed 0 --root_solver neural $COLL --out "$OUT/$1" "${@:2}" 2>&1 | tee "$OUT/$1.log"
}
run base
run tiers --bad_frame_tiers episode,segment,frame
# grid 路线单独一条命令：它要换 --root_solver，不走上面那个 neural 的模板
CUDA_VISIBLE_DEVICES=3 nice -n 12 scripts/s4_retarget.sh "$CLIP" --robot m7 --ckpt $CK \
  --seed 0 --root_solver grid $COLL --atf_preset auto --out "$OUT/grid" 2>&1 | tee "$OUT/grid.log"
set +x

echo "=== 1) base vs 改动前的参照物（$REF）"
for f in trajectory.npz metrics.npz robot_sim.mp4; do
  cmp -s "$REF/$f" "$OUT/base/$f" \
    && echo "SAME $f" \
    || echo "DIFF $f  <-- 搬 yaml 或加 tiers 动到了默认跑法的产物"
done
md5sum "$REF"/trajectory.npz "$OUT/base/trajectory.npz" \
       "$REF"/metrics.npz    "$OUT/base/metrics.npz" \
       "$REF"/robot_sim.mp4  "$OUT/base/robot_sim.mp4"

echo "=== 2) tiers（三层全开）vs base：原有产物必须全同"
for f in trajectory.npz root_frames.npz metrics.npz robot_sim.mp4 input_viz.mp4; do
  cmp -s "$OUT/base/$f" "$OUT/tiers/$f" \
    && echo "SAME $f" \
    || echo "DIFF $f  <-- 新增两层不是'只看不动'"
done

echo "=== 3) 报告文件：默认不写，开了才写"
ls "$OUT/base/bad_frame_tiers.json" >/dev/null 2>&1 \
  && echo "BAD  默认跑法多写了 bad_frame_tiers.json" \
  || echo "OK   默认跑法没有多出文件"
ls "$OUT/tiers/bad_frame_tiers.json" >/dev/null 2>&1 \
  && echo "OK   三层全开时写出了 bad_frame_tiers.json" \
  || echo "BAD  开了 episode,segment 却没写报告"
echo "--- 报告内容（整段级只警告 / 轨迹段级只标记）"
cat "$OUT/tiers/bad_frame_tiers.json"
echo "--- 默认跑法的日志里不该出现 tier 打印"
grep -c "\[tier:" "$OUT/base.log" || true
echo "--- 三层全开的日志"
grep "\[tier:" "$OUT/tiers.log" || true

echo "=== 4) grid 路线：yaml 里那组标定参数真的送进了 MuJoCo"
grep "\[ArmTorsoFilter\]" "$OUT/grid.log" || true
echo "期望 torso_half=[0.0695, 0.119, 0.239] enter_thresh=0.020 margin=0.020"
echo ALL_DONE
