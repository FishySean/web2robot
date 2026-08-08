#!/usr/bin/env bash
# 跑 scripts/dev/ 下的 M7 工具脚本（验证 / 出片 / 资产生成）。
#
# 薄壳，不含逻辑，只做四件事：
#   1. 用对的解释器（这台机器 conda activate 不生效，必须绝对路径）
#   2. PYTHONPATH = web2robot/src + 上游 retarget/
#      —— 我方 m7 定义在前者，少数脚本还要上游的 JaxVecEnv / wrist_ik / clip_io
#   3. cd 到上游 retarget/ —— 有两个脚本引用 examples/ 和 runs/ 里的片段与权重，
#      那些路径是相对它算的。M7 自己的资产已经不依赖 cwd 了（走 paths.yaml）
#   4. MUJOCO_GL=osmesa —— 这台机器没有 X11，默认 GLFW 后端直接报
#      "could not initialize GLFW"；egl 在这套 driver 上清理时抛 EGLError，
#      osmesa（CPU 软渲染）实测可用
#
#   scripts/dev/m7_tool.sh verify_m7_mjx_fk.py            # MJX FK 与 m7.xml 是否一致
#   scripts/dev/m7_tool.sh check_handframe_convention.py  # hand_frame 约定，两只手都验
#   scripts/dev/m7_tool.sh render_m7_hand_frames.py       # 画出末端帧三轴
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
UPSTREAM="$ROOT/external/EgoInfinity/retarget"
SCRIPT="$1"; shift
[[ -f "$ROOT/scripts/dev/$SCRIPT" ]] || { echo "找不到 scripts/dev/$SCRIPT"; exit 1; }
cd "$UPSTREAM"
exec env PYTHONPATH="$ROOT/src:$UPSTREAM${PYTHONPATH:+:$PYTHONPATH}" \
     MUJOCO_GL="${MUJOCO_GL:-osmesa}" PYTHONUNBUFFERED=1 \
     "$ROOT/envs/rt_env/bin/python" "$ROOT/scripts/dev/$SCRIPT" "$@"
