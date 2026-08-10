#!/usr/bin/env bash
# 流水线第③步：感知前端产物 → EgoInfinity clip 目录。
#
# 薄壳，只做三件事：
#   1. 按子命令挑解释器 —— 两条路线在**两个不同的 venv** 里（HaWoR 要
#      hawor_env，WiLoR+MoGe 要 perception_env），装在一起会打架。
#   2. src/ 进 PYTHONPATH（没 pip install -e 也能跑），前端仓库根目录也进去 ——
#      HaWoR 的 `from hawor.utils.process import ...` 是相对它自己的根算的。
#   3. cd 到前端仓库根 —— HaWoR 加载 MANO 模型用的是相对路径（_DATA/...），
#      不 cd 进去就 FileNotFoundError。所以 `--src`/`--out` 在 cd 之前先转绝对路径，
#      不然相对路径会跑到第三方 checkout 里去（和 s4 一样的坑）。
#
# 逻辑全在 src/web2robot/perception/，参数透传给它的 CLI。
#
#   scripts/s3_to_clip.sh hawor external/HaWoR/example/ho3d_SMu41 --frames 55 \
#       --out outputs/clips/ho3d_SMu41 --fps 15 --hands right
#   scripts/s3_to_clip.sh hawor --help
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CALLER_CWD="$PWD"

[[ $# -ge 1 ]] || { echo "用法: $0 <hawor> [参数...]   （见脚本头部注释）" >&2; exit 2; }
frontend="$1"; shift

case "$frontend" in
  hawor) PY="$ROOT/envs/hawor_env/bin/python"; REPO="$ROOT/external/HaWoR" ;;
  *) echo "还没有 '$frontend' 这条前端（现在只有 hawor；WiLoR+MoGe 待迁移）" >&2; exit 2 ;;
esac

# 路径参数在 cd 之前转绝对路径
args=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --out|--src)
      p="$2"; [[ "$p" = /* ]] || p="$CALLER_CWD/$p"
      args+=("$1" "$p"); shift 2 ;;
    --out=*|--src=*)
      k="${1%%=*}"; p="${1#*=}"; [[ "$p" = /* ]] || p="$CALLER_CWD/$p"
      args+=("$k" "$p"); shift ;;
    -*) args+=("$1"); shift ;;
    *)  # 位置参数（src）同样要转
        p="$1"; [[ "$p" = /* || ! -e "$CALLER_CWD/$p" ]] || p="$CALLER_CWD/$p"
        args+=("$p"); shift ;;
  esac
done

cd "$REPO"
exec env PYTHONUNBUFFERED=1 \
     PYTHONPATH="$ROOT/src:$REPO${PYTHONPATH:+:$PYTHONPATH}" \
     "$PY" -m web2robot.perception "$frontend" "${args[@]}"
