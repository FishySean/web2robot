#!/usr/bin/env bash
# 流水线第 1 步：取景质检 + 路由打标。
#
# 这是个薄壳，只做两件事：用对的解释器（这台机器 conda activate 不生效，
# 必须绝对路径）、把 src/ 放进 PYTHONPATH（没 pip install -e 也能跑）。
# 逻辑全在 src/web2robot/quality/，参数直接透传给它的 CLI。
#
#   scripts/s1_quality_gate.sh data/videos/ --out outputs/qc.jsonl --viz outputs/ev/
#   scripts/s1_quality_gate.sh --help
#
# 想整档跳过（2026-08-21：公司已有质检体系，这两步降级为可选）：
#   scripts/s1_quality_gate.sh data/videos/ --out outputs/qc.jsonl --quality_gate skip
#   scripts/s1_quality_gate.sh data/videos/ --out outputs/qc.jsonl --routing skip
# 默认 builtin = 行为不变。skip 档不加载任何模型，判决写 skipped（不是 accept）。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec env PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}" \
     "$ROOT/envs/rt_env/bin/python" -m web2robot.quality "$@"
