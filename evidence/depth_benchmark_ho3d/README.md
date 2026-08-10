# HO-3D 手腕定位评测 —— "单目深度是硬瓶颈，HaWoR 是解法"的证据

论文里要引的那张表。原始评测 2026-07-14，2026-08-10 从
`hand_projects/hand2robot/outputs/eval_hawor/` 捞出来冻进版本库。

## 结论

| 序列 | n | GT 深度运动 | WiLoR+MoGe 深度误差 | HaWoR 深度误差 | 深度跟随 r（W→H） |
|---|---|---|---|---|---|
| ABF12 | 74 | 3.6 cm | **11.0 cm（22%）** | **0.6 cm（1%）** | −0.64 → +0.61 |
| SMu41 | 46 | 1.0 cm | 9.5 cm（17%） | 3.5 cm（6%） | +0.11 → +0.61 ⚠ |
| MC4 | 66 | 7.1 cm | **0.7 cm（1%）** | 2.6 cm（5%） | +0.91 → +0.87 |

平面（XY）误差方向相反：HaWoR 2.0~2.6 cm，WiLoR+MoGe 0.4~1.7 cm。HaWoR 用自估焦距 600
＋ SLAM 坐标系，WiLoR 用真 camMat ＋ 2D 直接反投影，所以平面上它更准。

⚠ SMu41 的 GT 深度整段只变 1.0 cm，那条 `r` 是在噪声上算相关，**不可信**。图和
`format_table()` 都会自己把这句话标出来，不靠读表的人去查。

**诚实版结论：HaWoR 不是每条都碾压，而是"稳定有界"。** 它三条全 ≤3.5 cm；
WiLoR+MoGe 是"抽奖"—— MoGe 场景猜测碰对时极准（MC4 0.7 cm），猜错时灾难
（ABF12 11 cm，且深度**反相关** r=−0.64）。

r 为负这一项比误差更要紧：物体靠近时它认为在远离，拿这种深度做重定向会得到反向的
reach，而只看绝对误差看不出来。

## 怎么复现

**秒级，纯 numpy，不需要 GPU / HO-3D 数据集 / HaWoR checkout：**

```bash
envs/rt_env/bin/python -m unittest tests.test_depth_benchmark -v   # 15 个用例
envs/rt_env/bin/python scripts/dev/render_depth_benchmark_fig.py   # 重画汇总图
```

评测口径在 [`src/web2robot/eval/depth_benchmark.py`](../../src/web2robot/eval/depth_benchmark.py)。

## `data/bench_<SEQ>.npz` 里是什么

三方的**手腕 3D 点**，按帧号对齐前的原始形态（相机系，米，z 朝前）：

| 键 | 形状 | 来源 |
|---|---|---|
| `gt_frames` / `gt_wrist` | (n,) / (n,3) | HO-3D `meta/*.pkl` 的 `handJoints3D[0]`，乘过官方 `coordChangeMat=diag(1,−1,−1)` |
| `hawor_frames` / `hawor_wrist` | (T,) / (T,3) | HaWoR `world_space_res.pth` → `run_mano`（右手 idx1）→ SLAM 位姿转相机系 |
| `wilor_frames` / `wilor_wrist` | (m,) / (m,3) | WiLoR 2D 关键点 + MoGe 逐帧场景深度反投影（`step16b`） |
| `nfr` | 标量 | 当初跑 HaWoR 时的帧数，SLAM 文件名里带这个数 |

**存 3D 点而不是存算好的误差数**：论文里换统计口径（中位改均值、改 per-joint、改
只统计某段）还能重算，而不是只剩三个写死的数字。三个文件加起来 24 KB。

对齐是**逐方法各自和 GT 求交**（`align()`），不是三方一起求交 —— 两个方法覆盖的帧数
不同时，强行三方对齐会白扔掉某个方法的可用帧。所以 GT 深度的统计量也是逐方法算的。

## 为什么要冻，而不是把旧文件拷过来

旧那份复现路径要 **GPU + hawor_env + HaWoR checkout + HO-3D 数据集** 四样齐备，
而其中三样随时会没：

- HO-3D（`rgbd_val/`）是外部数据集，`.gitignore` 里明确排掉（提交 `4ea043c`）；
- `HaWoR/example/ho3d_*/` 是第三方 checkout 里的产物，一次 `git clean -xdf` 就没；
- `hawor.ckpt` 3.27 GB 不在库里（当初还下载截断过一次，1.37 GB 的坏包）。

**更要紧的是：这份证据本来就已经不在版本库里了。** `hand_projects/.gitignore` 第 32 行
`hand2robot/outputs/eval_hawor/*` 把整个目录的内容排掉（合理 —— 那里有 840 MB 逐帧
转储和 60 MB 视频），负号规则只放行 `*.md`。于是**核心的 npz 和图全部处于"随时会消失
且不会有人发现"的状态**，唯一进了库的是 `MODEL_ROUTING_RESULTS.md`。

`tests/test_depth_benchmark.py::TestEvidenceIsPresent` 现在就是这件事的守卫：证据文件
不在了、少了某一方、单位从米变成毫米，测试当场变红。

## 冻的过程

[`scripts/dev/freeze_depth_benchmark.py`](../../scripts/dev/freeze_depth_benchmark.py)，
一次性脚本（还需要那四样东西，所以只跑了一次）。它顺手多做一件事：HaWoR 那一路
**算两遍** —— 一遍是 `step16` 内联写法逐字抄的，一遍走迁移后的
`web2robot.perception.hawor`，然后断言逐位相同。

实测三条序列 74/74、46/46、66/66 帧全部逐位一致 ✓，等于给 perception 模块又加了三条
真实序列的比对（迁移当天只比过 `ho3d_SMu41` 一条）。

存进 npz 的是**内联版**的结果，因为那是当初出表用的；模块版会把 HaWoR 标 invalid
或含 inf 的帧置 NaN，这三条序列上恰好一帧都没屏掉，所以两者等价。

## `figures/`

- `FIG_SUMMARY_3seq.png` —— 现在的版本，每个数从 `bench_*.npz` **现算**。
- `original_2026-07-14/` —— 当初的原始产物，留作出处对照。

原版 `step17_summary_3seq.py` 第一行写着"硬编码自 step16 各序列输出"：15 个数全是手抄
进源码的，图和数据之间没有任何链接。新脚本把那 15 个数和现算值逐个核对
（**15/15 一致 ✓**），并且在 SMu41 那一栏加了 "(r not meaningful)" 的标注。

## 还没捞过来的

`hand_projects/hand2robot/outputs/eval_hawor/` 里还有别的东西（模型路由的 9 段视频实验
`MODEL_ROUTING_RESULTS.md`、M7 重定向对照片、per-finger 重定向的图）。那些也在同一条
`gitignore` 规则下面，但**不是这次要保的这份证据**，等各自要用的时候按同样的办法冻。
