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
#   5. 补 --no-preview（除非调用方自己给了）+ PYTHONUNBUFFERED=1。
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
#   scripts/s4_retarget.sh examples/fill_jar --robot m7 --out /tmp/out \
#       --ckpt runs/m7/taskspace_v2/checkpoints/final.pt --seed 0 --n_samples 5 \
#       --arm_torso_collision --dual_hand_collision
#   scripts/s4_retarget.sh --help
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UPSTREAM="$ROOT/external/EgoInfinity/retarget"
args=("$@")
case " ${args[*]} " in *" --no-preview "*|*" --help "*|*" -h "*) ;; *) args+=(--no-preview) ;; esac
cd "$UPSTREAM"
exec env PYTHONPATH="$ROOT/src:$UPSTREAM${PYTHONPATH:+:$PYTHONPATH}" \
     MUJOCO_GL="${MUJOCO_GL:-osmesa}" PYTHONUNBUFFERED=1 \
     "$ROOT/envs/rt_env/bin/python" "$UPSTREAM/scripts/test.py" "${args[@]}"
