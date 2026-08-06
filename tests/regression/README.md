# 参考运行 + 对照样本(回归基准)

`qc.md` / `qc.jsonl` / `contact_sheet.png` 是 2026-08-05 验证通过的一次完整运行
(7 个去重后的抓取片段 + 3 个人造对照)。改了阈值或判据之后重跑一遍对比这份,判决翻转了就得
说清为什么。

```bash
cd <仓库根>            # web2robot/
# 1) 重跑(~70s,占一块 GPU)
PYTHONPATH=src envs/rt_env/bin/python -m web2robot.quality \
    data/videos/ tests/regression/*.mp4 --out /tmp/re/qc.jsonl --viz /tmp/re/ev
# 2) 比对(判决零容忍,数值只报漂移)
envs/rt_env/bin/python scripts/dev/diff_quality_run.py /tmp/re/qc.jsonl
```

`diff_quality_run.py` 判两件事,都不是"数值逐位相同":判决字段逐字一致,以及**每个参与判决的
信号没有越过它的阈值**。KeypointRCNN 在 GPU 上不是逐位确定的 —— 实测重跑一次
`cup_cpvH8gzUTko` 的 `torso_rate` 就从 0.4828 变 0.4655(n=58,差值正好 1/58,一帧翻转)。
要求逐位相同的测试会在没人改代码时随机报红,很快就没人看了。

**已知脆弱点(工具自己报出来的):** `cand2_ZKCmHESpYgM` 的 `any_hand_rate` 恰好等于
`min_hand_ratio=0.75`,余量为 0。一帧翻转就会让 `low_hand_ratio` 这条 reason 出现/消失,
把判决字段的零容忍检查带红。`low_hand_ratio` 本身只记录不否决,所以这是测试的脆弱性而不是
判据的问题 —— 真被它绊到时,该做的是给这段样本换个不压线的片段,不是放宽判决的容忍度。

## 三个对照必须是这个结果

| 片段 | 真值 | `both_hand` | 判决 |
|---|---|---|---|
| `pos_twohands.mp4` | 双手 | 0.42 | **defer** (hands_only) |
| `neg_onehand.mp4` | 单手 | 0.00 | **reject** (no_stable_hands) |
| `neg_noperson.mp4` | 无人无手 | 0.00 | **reject** (no_person) |

## 这三个样本是"造"出来的,不是挑出来的

前两次做对照都失败了:想裁一段"没拍到头"的,裁完发现头还在;想裁一段"没有人"的,裁完发现
里面有两只手。**从真实素材里挑反例,挑到的内容和意图对不上**。所以改成按构造法生成 ——
`neg_onehand` 与 `pos_twohands` 除了右半边被涂黑之外**是同一段素材同一批帧**,任何指标差异
只能来自"手的数量",不能来自光照、背景、动作、编码。

```bash
# 无人无手:合成测试图
ffmpeg -f lavfi -i testsrc2=size=640x360:rate=25 -t 12 \
       -c:v libx264 -pix_fmt yuv420p neg_noperson.mp4
# 单手:同一段素材涂黑右半边(双手素材的配对负例)
ffmpeg -i <双手素材> -vf "drawbox=x=iw/2:y=0:w=iw/2:h=ih:color=black@1:t=fill" \
       -c:v libx264 -pix_fmt yuv420p neg_onehand.mp4
# 双手:同一段素材原样(配对正例)
```

`neg_noperson` 用 `testsrc2` 而不是黑屏:纯黑帧对检测器太容易了,证明不了什么;彩条有大量
边缘和色块,是个更硬的负例。
