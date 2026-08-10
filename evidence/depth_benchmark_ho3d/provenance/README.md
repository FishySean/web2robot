# HaWoR 那三次运行的出处

`../data/bench_*.npz` 里 `hawor_wrist` 的来源。2026-07-14 用 `external/HaWoR` 跑的，
原始 stdout 压在这个目录里（`hawor_run_*.log.gz`，一共 22 KB）。

留着的理由只有一个：**HaWoR 的度量尺度是它自己每段现估的，而且逐段差很多。**
没有这三个数，"0.6 cm" 就没法判断是不是可复现的。

| 序列 | NFR | HaWoR 估的尺度 | focal |
|---|---|---|---|
| ABF12 | 88 | 0.1902507320046425 | 600（默认，未提供） |
| SMu41 | 55 | 3.923701286315918 | 600（默认，未提供） |
| MC4 | 77 | 2.3423545360565186 | 600（默认，未提供） |

**尺度在三条序列之间差了 20 倍**（0.19 / 2.34 / 3.92）。这不是笔误 —— HaWoR 的
`estimated scale` 是从 SLAM 的相机轨迹和手的运动一起拟出来的，同一份权重、同一套参数，
换一段视频就是另一个数。所以：

- 重跑这三段而拿到不同的尺度，就说明深度误差表也会变，**先查尺度再怀疑别的**；
- "HaWoR 深度准" 这句话的适用范围就是"尺度估对了的片段"。三段里 SMu41 的
  深度误差 3.5 cm 明显比 ABF12 的 0.6 cm 差，尺度估计的质量是首要嫌疑；
- `focal` 三段都走了默认 600（日志里 `No focal length provided, use default 600`），
  也就是说这份评测里 HaWoR 用的**不是**HO-3D 的真内参，而 WiLoR 那条用了真
  `camMat`。这对 WiLoR 有利，而 WiLoR 仍然差一个量级 —— 结论的方向因此更稳，
  但引用这张表时得把这句话一起写上。

取出上表那几行：

```bash
zcat evidence/depth_benchmark_ho3d/provenance/hawor_run_abf12.log.gz \
  | grep -E "No focal|estimated scale|Loading cameras" | sort -u
```

日志里剩下的绝大部分是 `torch.cuda.amp.autocast` 的弃用告警和 ffmpeg 版本横幅，没有信息量，
压着放就行，不用读。

其余 31 个 `.log`（装环境、下权重、调试的那些）2026-08-10 删了 —— 它们记的是
"当时怎么把环境装起来的"，而那件事该由 `envs/` 和 `docs/` 说清楚，不该靠翻 build log。
