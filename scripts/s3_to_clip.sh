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
#   scripts/s3_to_clip.sh wilor data/webvid/xxx.mp4 --out outputs/clips/xxx --fps 15
#   scripts/s3_to_clip.sh hawor --help
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CALLER_CWD="$PWD"

[[ $# -ge 1 ]] || { echo "用法: $0 <hawor|wilor> [参数...]   （见脚本头部注释）" >&2; exit 2; }
frontend="$1"; shift

case "$frontend" in
  hawor) PY="$ROOT/envs/hawor_env/bin/python"; REPO="$ROOT/external/HaWoR" ;;
  # WiLoR 和 MoGe 都是 pip 包（wilor_mini / moge），没有要 cd 进去的第三方 checkout，
  # 所以 REPO 就是仓库根 —— 相对路径按用户预期解析，不会跑到 external/ 里去。
  wilor) PY="$ROOT/envs/perception_env/bin/python"; REPO="$ROOT" ;;
  *) echo "还没有 '$frontend' 这条前端（现在有 hawor、wilor）" >&2; exit 2 ;;
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

# HF_HOME 必须覆盖：这台机器的 shell 把它指向一个**共享且我们没写权限**的缓存目录，
# 而 MoGe 权重实际缓存在自己家目录下。不覆盖就是 PermissionError，而且报得像"权重没下载"
# （完整现象和那个共享路径记在 README「WiLoR+MoGe 这条前端」一节）。
# 用 $HOME 而不是写死绝对路径：换机器/换用户照样对，也过得了 test_no_hardcoded_paths。
cd "$REPO"
exec env PYTHONUNBUFFERED=1 \
     HF_HOME="${WEB2ROBOT_HF_HOME:-$HOME/.cache/huggingface}" \
     PYTHONPATH="$ROOT/src:$REPO${PYTHONPATH:+:$PYTHONPATH}" \
     "$PY" -m web2robot.perception "$frontend" "${args[@]}"
