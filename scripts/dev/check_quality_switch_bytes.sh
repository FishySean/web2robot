#!/usr/bin/env bash
# 「质检/路由三选一开关 ⇒ 默认跑法结论不变」验证。
#
# 为什么这里不能像 check_tiers_yaml_bytes.sh 那样直接比 md5：
#   质检的默认档要跑 KeypointRCNN，**它在 GPU 上不是逐位确定的**（选到哪块卡、
#   cuDNN 挑哪个算法都会变）。实测重跑一次 cup_cpvH8gzUTko 的 torso_rate 就从
#   0.4828 变 0.4655 —— n=58，差 1/58，一帧翻转（tests/regression/README.md）。
#   qc.md 里还写了 wall time，本来就不可能逐字节相同。
#   所以 builtin 档的判据是 diff_quality_run.py：**判决字段零容忍**、参与判决的
#   信号不许越过阈值、剩下的数值只报漂移。这比 md5 更贴近"结论没变"。
#   md5 仍然照打，只是当"参考"读，不当判据。
#
#   skip 档反过来：它不碰任何模型，是纯确定的，所以那一档**要求**逐字节相同。
#
# 五遍（前三遍各占一块 GPU 约 40 秒，后两遍 skip 档不到 0.1 秒）：
#   base = 一个新参数都不传（= 用户今天的跑法）
#   bi   = --quality_gate builtin --routing builtin（显式写出默认值）
#   nort = --routing skip（质检照旧全跑，只是不算路由）
#   skip = --quality_gate skip，跑两遍
#
# 判据：
#   1. base 对 2026-08-05 的回归基准：判决字段逐字一致
#   2. bi（显式默认值）对 base：判决字段逐字一致 —— 加了开关不等于改了默认档
#   3. nort 对 base：**只允许** suggested_route / route_rationale 两项变，
#      verdict / reasons 一字不许变
#   4. skip：两遍 jsonl 逐字节相同、判决全是 skipped（不是 accept）、日志里没有
#      加载手部检测器、耗时降到 1 秒以内
#
#   bash scripts/dev/check_quality_switch_bytes.sh > outputs/dev/quality_switch_bytecheck.log 2>&1
set -x
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
OUT="outputs/dev/quality_switch_bytecheck"
PY=envs/rt_env/bin/python
IN=(data/videos/ tests/regression/pos_twohands.mp4 tests/regression/neg_onehand.mp4 \
    tests/regression/neg_noperson.mp4)
mkdir -p "$OUT"
run () {  # run <名字> [额外参数…]
  CUDA_VISIBLE_DEVICES=3 nice -n 12 env PYTHONPATH=src $PY -m web2robot.quality \
    "${IN[@]}" --out "$OUT/$1.jsonl" "${@:2}" 2>&1 | tee "$OUT/$1.log"
}
run base
run bi   --quality_gate builtin --routing builtin
run nort --routing skip
run skip1 --quality_gate skip
run skip2 --quality_gate skip
set +x

echo "=== 0) 参考用的 md5（builtin 档本来就不保证逐位相同，只当参考读）"
md5sum tests/regression/qc.jsonl "$OUT"/base.jsonl "$OUT"/bi.jsonl "$OUT"/nort.jsonl \
       "$OUT"/skip1.jsonl "$OUT"/skip2.jsonl

echo "=== 1) base vs 2026-08-05 回归基准（判决零容忍）"
$PY scripts/dev/diff_quality_run.py "$OUT/base.jsonl"

echo "=== 2) bi（显式默认值）vs base（判决零容忍）"
$PY scripts/dev/diff_quality_run.py "$OUT/bi.jsonl" --baseline "$OUT/base.jsonl"

echo "=== 3) nort（--routing skip）vs base：只许路由字段变"
$PY - "$OUT/base.jsonl" "$OUT/nort.jsonl" <<'EOF'
import json, sys
ALLOWED = {"suggested_route", "route_rationale"}
load = lambda p: {json.loads(l)["clip_id"]: json.loads(l)
                  for l in open(p) if l.strip()}
old, new = load(sys.argv[1]), load(sys.argv[2])
assert set(old) == set(new), (sorted(old), sorted(new))
bad = []
for k in sorted(old):
    o, n = old[k], new[k]
    # signals 里的数值抖动是 GPU 非确定性，这里不判（diff_quality_run.py 判它）
    differ = {f for f in set(o) | set(n)
              if f not in ("signals", "per_frame") and o.get(f) != n.get(f)}
    extra = differ - ALLOWED
    print(f"{k:32s} verdict {o['verdict']} -> {n['verdict']}  变了: {sorted(differ)}")
    print(f"{'':32s} route  {o['suggested_route']} -> {n['suggested_route']}")
    if extra:
        bad.append((k, sorted(extra), "路由开关动到了不该动的字段"))
    if o["verdict"] != n["verdict"]:
        bad.append((k, ["verdict"], "判决变了"))
    if o["reasons"] != n["reasons"]:
        # cand2_ZKCmHESpYgM 的 any_hand_rate 恰好压在 min_hand_ratio=0.75 上，
        # low_hand_ratio 会随单帧翻转出现/消失（回归 README 记着的已知脆弱点）——
        # 真被它绊到时先看是不是这段，再看是不是开关的锅
        bad.append((k, [f"reasons {o['reasons']} -> {n['reasons']}"],
                    "判决理由变了（看是不是那段压线样本）"))
    if n["suggested_route"] is not None:
        bad.append((k, ["suggested_route"], "说了 skip 还给出了路线"))
    if not any("routing=skip" in w for w in n["route_rationale"]):
        bad.append((k, ["route_rationale"], "没写明是因为关了路由才没路线"))
print("\n判据 3：" + ("通过" if not bad else "不通过"))
for k, f, why in bad:
    print(f"  {k}: {f} <- {why}")
sys.exit(1 if bad else 0)
EOF
echo "判据 3 退出码 $?"

echo "=== 4) skip 档：确定性 + 真的没跑"
cmp -s "$OUT/skip1.jsonl" "$OUT/skip2.jsonl" \
  && echo "SAME skip 两遍 jsonl 逐字节相同（这一档不碰模型，所以要求逐位相同）" \
  || echo "DIFF skip 档居然不确定 <-- 有隐藏状态"
$PY - "$OUT/skip1.jsonl" <<'EOF'
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
v = sorted({r["verdict"] for r in rows})
ran = sorted({s for r in rows for s in r["stages_run"]})
route = {r["suggested_route"] for r in rows}
print(f"片段 {len(rows)} 段，判决集合 {v}，跑过的 stage {ran}，路线集合 {route}")
ok = v == ["skipped"] and ran == [] and route == {None}
print("判据 4a：" + ("通过（全是 skipped，不是 accept；一个 stage 都没跑）"
                     if ok else "不通过"))
sys.exit(0 if ok else 1)
EOF
echo "判据 4a 退出码 $?"
echo "--- skip 档不该加载手部检测器 / builtin 档该加载"
grep -c "hand detector on" "$OUT/skip1.log" || true
grep -c "hand detector on" "$OUT/base.log" || true
echo "--- 耗时对比（builtin 实测约 40s，skip 应该 <1s）"
grep -h "^\[quality\] [0-9.]*s ->" "$OUT/base.log" "$OUT/nort.log" "$OUT/skip1.log"
echo "--- 开关状态在日志和配置快照里都留了痕"
grep -h "quality_gate=" "$OUT"/base.log "$OUT"/nort.log "$OUT"/skip1.log
grep -h '"quality_gate"\|"routing"' "$OUT"/skip1.md "$OUT"/nort.md
echo ALL_DONE
