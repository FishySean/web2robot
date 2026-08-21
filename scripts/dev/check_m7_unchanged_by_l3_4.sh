#!/usr/bin/env bash
# 「加了第二台机器人 L3.4，M7 的产物一个字节都没变」的验证。
#
# 为什么要真跑一遍而不是看代码：L3.4 那三处注册**看起来**都是隔离的
# （注册表加一个独立键、`RobotIKConfig.l3_4` 是个新方法、工厂里加一个 if 分支），
# 但 `sim/robots/__init__.py` 里那两行 import 是**模块顶层**的 ——
# 跑 M7 也会执行它们（加载 L3.4 的 MJCF 路径、构造 RobotConfig）。
# "顶层 import 应该没有副作用"和"产物没变"是两件事：改 import 顺序影响随机数流、
# numpy 全局状态、mujoco 全局注册，都只有比 md5 才发现。
#
# 参照物是 A1 那次标定验证留下的 base/ 产物（2026-08-20 19:15，**早于** L3.4 的改动）：
#   trajectory.npz 9ef35b4eed590c543ae4af9c9b89e5c9
#   metrics.npz    33c049ac5b26fd848cdbcfa93321fae8
#   robot_sim.mp4  205d96dba4a701e4be19a88ff1ec0483   ← 这个数也写在 patches/README.md 里
# 所以这里只需要跑**一遍**（现在这份代码），和那份比。
# 参照物被删了就先重跑一次 scripts/dev/check_neural_bytes.sh 把 base/ 造回来。
#
# 片段/机器人/seed/路线/开关必须和参照物那次逐字相同 —— 差一个参数这条比对就没意义。
#   bash scripts/dev/check_m7_unchanged_by_l3_4.sh > outputs/dev/l34_m7_unchanged.log 2>&1
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
REF=outputs/dev/neural_bytecheck/base
OUT=outputs/dev/l34_m7_unchanged/after
if [[ ! -f "$REF/robot_sim.mp4" ]]; then
  echo "参照物不在：$REF —— 先跑 scripts/dev/check_neural_bytes.sh"; exit 1
fi

set -x
CUDA_VISIBLE_DEVICES=6 nice -n 12 scripts/s4_retarget.sh \
  "$ROOT/data/clips_official/-1r9yl-P-Ao_86.3_90.8" \
  --robot m7 --ckpt runs/m7/taskspace_v2/checkpoints/final.pt \
  --seed 0 --root_solver neural --arm_torso_collision --dual_hand_collision \
  --out "$OUT"
set +x

md5sum "$REF"/{trajectory.npz,metrics.npz,robot_sim.mp4} \
       "$OUT"/{trajectory.npz,metrics.npz,robot_sim.mp4}
rc=0
for f in trajectory.npz metrics.npz robot_sim.mp4; do
  if cmp -s "$REF/$f" "$OUT/$f"; then echo "SAME $f"
  else echo "DIFF $f  <-- 加 L3.4 动到了 M7 的产物"; rc=1; fi
done
echo "ALL_DONE rc=$rc"
exit $rc
