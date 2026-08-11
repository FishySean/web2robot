# 工程结构总览 —— 一切有迹可循

这份文档的用途：**不翻文件夹就能定位到想找的东西。** 新接手这个工程的人（包括半年后的
自己）只看这一份，应该就知道每样东西该在哪、为什么在那。

[`../README.md`](../README.md) 是给新人看的"这个项目是什么、怎么启动"；这份是"什么东西
放在哪"。规矩看 [`CONVENTIONS.md`](CONVENTIONS.md)，坑看 [`PITFALLS.md`](PITFALLS.md)，
验收看 [`VERIFICATION.md`](VERIFICATION.md)。四份不重复。

---

## 0. 一句话导航

| 我要找… | 去哪 |
|---|---|
| 某个流水线环节的代码 | `src/web2robot/<环节名>/` —— 包名就是环节名 |
| 怎么跑一条命令 | `scripts/s*.sh`（薄壳，一个环节一个） |
| 论文要引的实验数字 | `evidence/` —— **只有这里的产物进 git** |
| 跑出来的视频 / npz / clip | `outputs/`，按"谁写的"分子目录（见 §4） |
| 原始视频素材 | `data/` |
| 机器人 MJCF / mesh | `assets/robots/m7/` |
| 绝对路径、checkpoint 位置 | `configs/paths.yaml` —— **全工程唯一允许写绝对路径的文件** |
| 我们对第三方仓库改了什么 | `external/patches/README.md` |
| 某个决定当时为什么那么定 | `docs/` + 各模块 `__init__.py` 的文档字符串 |
| 重要参考资料的确切路径 | **本文 §3** |

---

## 1. 目录树 + 流水线环节对应

```
web2robot/
│
├── src/web2robot/            ← 全部逻辑。一个流水线环节一个包
│   ├── paths.py                 路径解析总入口（P.weights() / P.check_output_dir()）
│   ├── common/                  跨环节共用（video_io：解码抽帧）
│   ├── quality/     ══════ ①取景质检     拍全了没 / 稳不稳 / 背景合不合适
│   ├── routing/     ══════ ②视角与运动分类  第一或第三人称、相机动不动 → 选技术路线
│   ├── perception/  ══════ ③感知前端
│   │   ├── hawor.py             相机运动的片段走这条（SLAM，深度准，条件不满足整段失败）
│   │   ├── wilor.py + moge.py   相机固定的片段走这条（逐帧，从不崩溃，深度差 → 见 §3）
│   │   └── to_clip.py           下游输入契约（EgoInfinity clip 目录），与用哪个前端无关
│   ├── retarget/    ══════ ④重定向        坏帧兜底 + best-of-N 根锚点采样
│   ├── robots/m7/               机器人定义（IK 链、hand_frame 约定、采样配置）
│   │                            ——**不 import 任何重定向框架**，换框架时不用改
│   ├── collision/   ══════ ⑤碰撞检测      臂-躯 / 双手 / 手指胶囊过滤
│   ├── trajectory/  ══════ ⑤轨迹处理      坏帧三级检测 + 长度感知填补
│   └── eval/                    评测代码（给 evidence/ 算表用，纯 numpy、秒级）
│
├── scripts/                 ← 薄壳：只负责"用对的解释器 + 设好 PYTHONPATH"，不含逻辑
│   ├── s1_quality_gate.sh       ①
│   ├── s3_to_clip.sh            ③（子命令 hawor / wilor，各自的 venv）
│   ├── s4_retarget.sh           ④＋⑤（调上游主流程，碰撞/清洗走我方包）
│   └── dev/                     开发期工具：check_* 回归比对、render_*/viz_* 出片
│
├── tests/                   ← stdlib unittest，秒级，119/119
│   └── regression/              回归基准片段 + 期望判决（qc.jsonl / contact_sheet.png）
│
├── configs/paths.yaml       ← 唯一允许写绝对路径的地方。换机器只改这一个文件
│
├── assets/                  ← 我们产出的资产，进 git
│   ├── robots/m7/               MJCF / URDF / mesh / MJX（103 个文件）
│   └── weights/                 第三方权重的落地点（gitignore，`.gitkeep` 占位）
│
├── evidence/                ← 论文要引的证据。**进 git**（详见 §2 的三方边界）
│   └── depth_benchmark_ho3d/    深度误差 11cm → 0.6cm 那份，见 §3.1
│
├── data/                    ← 原始素材，不进 git（只有 README/MANIFEST 进）
│   ├── videos/                  质检用的候选视频（symlink 到旧目录）
│   └── webvid/raw/              手工挑的 7 段原片 + MANIFEST.md5（重抓不回来）
│
├── outputs/                 ← 全部产物，不进 git。**产物只许落这里**（见 §4）
│   ├── clips/                   ③的产物：EgoInfinity clip 目录
│   ├── retarget/                ④⑤的产物：trajectory.npz / robot_sim.mp4 / input_viz.mp4
│   ├── viz/                     给人看的结论片（四宫格、对比图）← §3.2
│   ├── dev/                     scripts/dev/ 出的片
│   ├── migration_check/         迁移期的新旧对比 run
│   ├── legacy_runs/             2026-08-10 从 external/ 搬回来的 316 MB 存量
│   └── archive/                 阶段性封存（`<主题>_<年-月>/`）
│
├── external/                ← 第三方仓库的 **symlink**，里面不改代码
│   ├── EgoInfinity -> ../../EgoInfinity
│   ├── HaWoR       -> ../../HaWoR
│   └── patches/                 我们对上游的改动全部记在这里
│
├── envs/                    ← 三个 venv 的 symlink + requirements-*.txt
├── docs/                    ← 决策记录、优先级、待办（本文也在这）
└── archive/                 ← 空占位；重构前的旧目录在 configs/paths.yaml 里注册为只读
```

流水线图和目录的对应关系是**一对一的**，这是故意的：看到图上某个框，包名就是框上的字。
唯一的例外是⑤，它是两个包 —— `collision/`（空间上的对不对）和 `trajectory/`（时间上的
连不连），因为它们的失效方式不同、验证方式也不同。

---

## 2. 三个最容易混的目录：`data/` vs `outputs/` vs `evidence/`

分界线只有一条问题：**丢了以后能不能拿回来。**

| | 丢了怎么办 | 进 git 吗 | 放什么 |
|---|---|---|---|
| `data/` | **拿不回来**（手工挑的素材，没有下载脚本） | 素材不进，**说明和 md5 清单进** | 原始视频 |
| `outputs/` | 重跑一遍流水线 | 不进 | clip、轨迹、视频、日志 |
| `evidence/` | **可能再也算不出来** | **进** | 论文要引的原始测量值 |

`evidence/` 之所以单列，是因为它的复现路径很脆：外部数据集会被清、第三方 checkout 会被
`git clean`、3 GB checkpoint 不在库里、机器会换。所以它的规矩比 `outputs/` 严
（见 [`evidence/README.md`](../evidence/README.md)）：**小 / 存原始测量值不存结论数字 /
秒级可复核零重依赖 / 每个数都有测试钉着**。

`external/` 的性质要单说：它是**别人家的目录**，我们只有读权限的心态。产物落进去的危害
不是乱，是一次 `git clean -xdf` 就全没 —— 实测攒过 408 MB 我们的产物在里面，而上游 git
只跟踪其中 1 个。这条判据写成了代码（`P.check_output_dir()`，违反就 `SystemExit`）而不是
写在文档里，理由见 [`CONVENTIONS.md` 第 1 条](CONVENTIONS.md)。

---

## 3. 重要参考资料 —— 逐条记住位置

### 3.1 深度误差对比实验（11 cm → 0.6 cm）★ 论文核心材料

整条链路"单目深度是硬瓶颈、HaWoR 是解法"的证据。**这是目前最不可替代的一份材料。**

| 路径 | 是什么 |
|---|---|
| [`evidence/depth_benchmark_ho3d/README.md`](../evidence/depth_benchmark_ho3d/README.md) | 结论、口径、怎么复核。**先看这个** |
| `evidence/depth_benchmark_ho3d/data/bench_{ABF12,SMu41,MC4}.npz` | 冻结的原始 3D 手腕点（24 KB）。存的是**测量值不是"11.0 cm"** |
| `evidence/depth_benchmark_ho3d/figures/FIG_SUMMARY_3seq.png` | 汇总图（可由脚本重画） |
| `evidence/depth_benchmark_ho3d/figures/original_2026-07-14/` | 2026-07-14 首次跑出来的原始四张图，**不重画，留档** |
| `evidence/depth_benchmark_ho3d/provenance/` | 那三次运行的 stdout（gz，22 KB）—— 溯源，见下 |
| `src/web2robot/eval/depth_benchmark.py` | 算表的代码，纯 numpy |
| `tests/test_depth_benchmark.py` | 19 个用例 0.3 秒，把论文里的每个数钉住 |

**引用这张表时有两句话必须一起写上**（钉在 `tests/test_depth_benchmark.py::TestProvenance`）：

1. **HaWoR 的度量尺度是它每段现估的** —— 三条序列量到 0.19 / 2.34 / 3.92，**差 20 倍**。
   重跑拿到别的尺度，整张表都会变，所以出现异常先查尺度再怀疑别的。
2. **HaWoR 跑的是默认 focal 600，WiLoR 那条用了 HO-3D 的真 `camMat`** —— 也就是这份对比
   **对 WiLoR 有利**，而 WiLoR 仍差一个量级。方向因此更稳，但不能不提。

### 3.2 "两种深度估计策略各错一半"——新发现的对比视频

| 路径 | 是什么 |
|---|---|
| **`outputs/viz/wilor_depth_modes.mp4`** | 四宫格对比片（h264，1430×770）。左右两个 3D 面板**共用同一个视野半径** —— 各自 autoscale 会把 6.5 倍的尺度差藏起来 |
| `scripts/dev/viz_wilor_depth_modes.py` | 出这个片的脚本（可重跑，命令在 [`VERIFICATION.md`](VERIFICATION.md) 的③感知小节） |
| `src/web2robot/perception/wilor.py` 文件头 | 那张骨长表 + 为什么两条策略的"开合"数字不可比 |
| `outputs/clips/cli_smoke_abf12_{pointmap,globalscale,K}/` | 三条深度路径各自的 clip 产物 |

结论（ABF12 前 30 帧实测，判据是骨长 —— 真手 MANO 骨长 2~4 cm 且逐帧近似常数）：

| | 骨长均值 | 骨长逐帧变异 | 病在哪 |
|---|---|---|---|
| `pointmap` | 2.94 cm ✓ | **5.7%** ✗ | 尺度对，手形被深度噪声撕开 |
| `global-scale` | **0.45 cm** ✗ | 0.5% ✓ | 手形对，整只手缩小约 6.5 倍 |

**这是一个待评估的新方向，不是已完成的工作**：取长补短（WiLoR 手形 + MoGe 逐帧手腕深度锚）
是新设计，要单独立项、单独量。6.5 倍不是常数（= 场景深度中位 / WiLoR 手腕深度中位），
换视频就变，所以 `global-scale` 出来的**绝对尺寸整段不可信**，只有形状和相对变化可信。

### 3.3 四宫格验证视频 —— 都在哪

四宫格是这个工程的标准验收形式（"指标 ≠ 画面"那条规矩的落地）。**规律是按"谁出的片"分**：

| 位置 | 哪个环节 / 哪次验收 |
|---|---|
| **`outputs/viz/<主题>.mp4`** | **给人看的结论片**（不是调试）。目前：`wilor_depth_modes.mp4` |
| `outputs/dev/compare_grid/fill_jar_grid_h264.mp4` | ⑤碰撞迁移：源 / 不开碰撞 / 新代码 / 旧代码 |
| `outputs/dev/compare_grid_retarget/fill_jar_grid_h264.mp4` | ④重定向迁移的同一组对比 |
| `outputs/dev/fill_jar/robot_sim_axes_h264.mp4` | `hand_frame` 轴向验收（带坐标轴叠加） |
| `outputs/migration_check/fill_jar_migration_quad.mp4` | 碰撞迁移期那次四宫格 |
| `outputs/legacy_runs/examples/_compare/{fill_jar,serve_cake}_badframe_quad.mp4` | 坏帧兜底机制的前后对比 |
| `outputs/legacy_runs/runs/m7/validation/fill_jar/robot_sim_axes_h264.mp4` | M7 资产迁移后的逐帧 hand_frame 验收 |
| `outputs/retarget/<片段名>/robot_sim.mp4` + `input_viz.mp4` | ④⑤每次正式跑的产物（不是四宫格，是单画面） |

**约定（往后请照这个放）**：

- 想让别人看结论的片 → `outputs/viz/<主题>.mp4`，一个主题一个文件，别套目录。
- 只为自己排查的片 → `outputs/dev/<run 名>/`，由 `scripts/dev/_devcli.py` 自动落位。
- 视频**一律 h264 / yuv420p**，否则 VSCode 里放不出来（mpeg4 踩过）。
- `outputs/migration_check/` 是迁移期的历史目录，**已封存不再往里写**。

### 3.4 其它需要记住位置的

| 路径 | 是什么 |
|---|---|
| [`../README.md`](../README.md) | 项目介绍：做什么、有哪些环节、每个环节用什么技术、怎么启动。**给新人的第一份** |
| [`CONVENTIONS.md`](CONVENTIONS.md) | 9 条必须遵守的工程规矩 + 每条由哪个测试钉着。**动手写代码之前看** |
| [`VERIFICATION.md`](VERIFICATION.md) | 一个模块一套验收判据 + 迁移的五步方法论。**改完之后看** |
| [`PITFALLS.md`](PITFALLS.md) | 17 个踩过的坑，现象 → 真因 → 怎么防。**报错方向不对时看** |
| [`external/patches/README.md`](../external/patches/README.md) | 我们对上游改了什么、为什么，以及每次迁移的处置记录。**动上游之前必读** |
| `external/patches/egoinfinity-modified.patch` | 唯一一份上游 diff（233 insertions）。**它变小是迁移做对了，变大就是有人往上游写逻辑** |
| `outputs/legacy_runs/MANIFEST.tsv` | 从 `external/` 搬回来的 316 MB 存量的逐文件清单（保持原相对路径，没重命名） |
| `data/webvid/README.md` + `raw/MANIFEST.md5` | 7 段手工挑的原片是什么、`md5sum -c` 怎么复核。**注意：这批是挑过的，不能当质检评测集**（选择偏差正好抵消掉质检要测的东西） |
| `tests/regression/` | 质检的回归基准：3 段片 + 期望判决 + contact sheet |
| [`docs/PRIORITY_2026-08-07.md`](PRIORITY_2026-08-07.md) | 当前优先级：质检/路由暂停自研，重定向第一 |
| [`docs/TODO22_FRONTEND_CONSOLE.md`](TODO22_FRONTEND_CONSOLE.md) | 前端控制台的设计要求 |
| [`docs/SYNC_2026-08-07.md`](SYNC_2026-08-07.md) | 阶段性同步记录 |
| `envs/requirements-{rt,hawor,perception}.txt` | 三个环境的精确版本。**共享机器，不要 pip install** |

---

## 4. 产物落点的规律：按"谁写的"分

`outputs/` 不是随手建目录 —— 每个写入口都有固定落点，看到路径就知道是谁跑的：

| 写入口 | 落点 |
|---|---|
| `scripts/s3_to_clip.sh` | `outputs/clips/<片段名>/`（3~4 个 clip 契约文件） |
| `scripts/s4_retarget.sh` | `outputs/retarget/<片段名>/`（顶掉上游"写在素材旁边"的默认值） |
| `scripts/dev/_devcli.py`（7 个出片脚本共用） | `outputs/dev/<run 名>/` |
| `scripts/dev/render_compare_grid.py` | 自带 `--out`，习惯落 `outputs/dev/compare_grid*/` |
| 人工封存 | `outputs/archive/<主题>_<年-月>/` |

四个写入口都过 `P.check_output_dir()` 这道闸，落点在 `external/` 里就直接 `SystemExit`。

---

## 5. 怎么防这份文档过期

文档会烂，所以它有测试钉着：`tests/test_docs_layout.py`

- **新建一个顶层目录但没在本文说明 → 测试变红**（这是最容易烂的地方）。
- 本文提到的、**应该进 git 的**路径（`src/` `evidence/` `configs/` `docs/` 之类）
  必须真实存在。
- `outputs/` `data/` 下的路径不做存在性断言 —— 它们不进 git，新克隆本来就没有；
  但它们的**父目录约定**要在本文 §4 的表里出现。

```bash
envs/rt_env/bin/python -m unittest tests.test_docs_layout -v
```
