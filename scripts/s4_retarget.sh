#!/usr/bin/env bash
# 流水线第④+⑤步：重定向 + 自碰撞纠正。
#
# 这是个薄壳，只做四件事：
#   1. 用对的解释器（这台机器 conda activate 不生效，必须绝对路径）
#   2. 把 web2robot/src 放进 PYTHONPATH —— 上游 test.py 里的碰撞/清洗
#      import 已经改成 web2robot.collision.* / web2robot.trajectory.*
#   3. cd 到上游 retarget/ —— 它的机器人 config 和 checkpoint 路径都是相对它算的
#   4. 设 MUJOCO_GL=osmesa —— 这台机器没有 X11（DISPLAY 不存在），默认的
#      GLFW 后端直接 "could not initialize GLFW"，轨迹算完了但出不了片；
#      egl 在这套 driver 上初始化后清理时抛 EGLError，osmesa（CPU 软渲染）
#      实测可用。渲染慢一点，但这一步本来就是离线出片。
#   5. 把 --out 摆平（2026-08-10 加）——**上游 test.py 的 --out 默认值是
#      `<clip_parent>/<robot>/`，也就是把产物写在输入素材旁边**。再叠上第 3 条的
#      cd，结果就是产物全落进 external/（实测攒了 408 MB、243 个 mp4/npz，其中
#      上游 git 只跟踪 1 个）。external/ 是第三方 checkout，一次 git clean 就没了。
#      所以这里：给了 --out 就按**调用方原来的 cwd** 解析（cd 之前先转绝对路径，
#      不然相对路径会跑到上游目录里去）；没给就顶掉上游默认值，落
#      outputs/retarget/<片段名>/。落点是否合法由 web2robot.paths.check_output_dir
#      判（拒绝 external/ 内的路径），这里只负责不把人往坑里带。
#   6. 补 --no-preview（除非调用方自己给了）+ PYTHONUNBUFFERED=1。
#      上游 test.py 跑完会拉一个交互式 GLFW 预览窗口，无头机器上它在 C 层
#      直接 abort —— **python 的 stdout 缓冲区来不及 flush，整份日志凭空消失**
#      （踩过：robot_sim.mp4 都写出来了，日志里却只剩一行 GLFW 报错，
#      ArmTorsoFilter 的统计全丢）。两条一起上：不开预览，且不缓冲。
#
# 逻辑不在这里：碰撞纠正在 src/web2robot/collision/，轨迹清洗在
# src/web2robot/trajectory/，上游只剩重定向主流程和参数接线（见
# external/patches/）。
#
# 复现性注意：根锚点是从**随机先验**积 ODE 得来的，不给 --seed 就每次都不一样
# （上游 test.py 第 241 区块自己写了这件事）。要跟别人的结果对比就必须
# --seed 固定 + --n_samples 相同。
#
#   scripts/s4_retarget.sh examples/fill_jar --robot m7 --out outputs/retarget/fill_jar \
#       --ckpt runs/m7/taskspace_v2/checkpoints/final.pt --seed 0 --n_samples 5 \
#       --arm_torso_collision --dual_hand_collision
#   scripts/s4_retarget.sh --help
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UPSTREAM="$ROOT/external/EgoInfinity/retarget"
CALLER_CWD="$PWD"

args=()
clip=""; have_out=0; is_help=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --help|-h) is_help=1; args+=("$1"); shift ;;
    --out)
      # 相对路径按调用方的 cwd 解析 —— 下面要 cd 到上游，不转成绝对路径就会落进去
      have_out=1
      out="$2"; [[ "$out" = /* ]] || out="$CALLER_CWD/$out"
      args+=(--out "$out"); shift 2 ;;
    --out=*)
      have_out=1
      out="${1#--out=}"; [[ "$out" = /* ]] || out="$CALLER_CWD/$out"
      args+=(--out "$out"); shift ;;
    -*) args+=("$1"); shift ;;
    *)  [[ -n "$clip" ]] || clip="$1"; args+=("$1"); shift ;;
  esac
done

# 没给 --out 就顶掉上游那个"写在输入素材旁边"的默认值
if [[ $is_help -eq 0 && $have_out -eq 0 && -n "$clip" ]]; then
  args+=(--out "$ROOT/outputs/retarget/$(basename "${clip%/}")")
fi
case " ${args[*]} " in *" --no-preview "*|*" --help "*|*" -h "*) ;; *) args+=(--no-preview) ;; esac
cd "$UPSTREAM"
exec env PYTHONPATH="$ROOT/src:$UPSTREAM${PYTHONPATH:+:$PYTHONPATH}" \
     MUJOCO_GL="${MUJOCO_GL:-osmesa}" PYTHONUNBUFFERED=1 \
     "$ROOT/envs/rt_env/bin/python" "$UPSTREAM/scripts/test.py" "${args[@]}"
