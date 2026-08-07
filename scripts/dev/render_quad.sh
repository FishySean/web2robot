#!/usr/bin/env bash
# 四宫格对比视频：把 4 段等长视频拼成 2x2 并打标签。
#
# 用途是"人眼确认"这一关 —— 指标对了不等于数据可信（穿模、抖动、抓握被破坏
# 这些只有画面能说清）。所以每次改碰撞/清洗逻辑都要出一版四宫格。
#
# 用法:
#   scripts/dev/render_quad.sh 输出.mp4 视频1 标签1 视频2 标签2 视频3 标签3 视频4 标签4
#
# 编码固定 h264 / yuv420p:mpeg4 在 VSCode 里放不出来（踩过）。
set -euo pipefail
[ $# -eq 9 ] || { sed -n '2,12p' "$0"; exit 2; }
OUT="$1"; shift
# 必须用带中日韩字形的字体：Lato 之类只有拉丁字形，中文标签会画成一串豆腐块
# （踩过一次，四宫格出来标签全是方框）。
FONT=/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc
PANEL_W=720; PANEL_H=540

filters=""
for i in 0 1 2 3; do
  v="$1"; l="$2"; shift 2
  [ -f "$v" ] || { echo "找不到 $v" >&2; exit 1; }
  inputs="${inputs:-} -i $v"
  # 标签画在左上角半透明黑底上,保证浅色/深色画面都读得清
  filters="${filters}[$i:v]scale=${PANEL_W}:${PANEL_H}:force_original_aspect_ratio=decrease,"
  filters="${filters}pad=${PANEL_W}:${PANEL_H}:(ow-iw)/2:(oh-ih)/2:color=white,"
  filters="${filters}drawtext=fontfile=${FONT}:text='${l}':x=12:y=10:fontsize=30:"
  filters="${filters}fontcolor=black:box=1:boxcolor=white@0.75:boxborderw=8[v$i];"
done

# shellcheck disable=SC2086
ffmpeg -hide_banner -v warning -y $inputs \
  -filter_complex "${filters}[v0][v1][v2][v3]xstack=inputs=4:layout=0_0|w0_0|0_h0|w0_h0[out]" \
  -map "[out]" -c:v libx264 -pix_fmt yuv420p -crf 20 "$OUT"
echo "写出 $OUT"
ffprobe -v error -show_entries stream=width,height,nb_frames,codec_name -of default=nk=1:nw=1 "$OUT"
