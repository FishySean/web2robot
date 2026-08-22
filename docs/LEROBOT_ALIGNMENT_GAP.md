# 产出格式对齐 LeRobot v3.0：差距分析

2026-08-22。对应 2026-08-21 方向调整的任务A。

**这份文档只是分析。** 没有写任何转换代码，没有动过一行现有输出逻辑，
`trajectory.npz` 的字段和字节和 8/21 之前完全一样。下面每个数字都是这次实测的，
命令附在文末，可以逐条重跑。

参考样例：`/mnt/vlm/common/datasets/ABC-130k_lerobot_v30_repair_filter_qf094`
（59 MB，整棵树是软链接壳；真身在 `ABC-130k_lerobot_v30/`，一个任务一个 shard，
单任务 11 GB）。以下"参考"一律指实测这份数据，不是指 LeRobot 上游文档 —— 两者
不一定一致，而我们要对齐的是公司内部真正在吃的这份。

---

## 1. 结论先说

| 一句话 | 依据 |
|---|---|
| **数值部分（`action` / `observation.state` / 时间轴 / 索引 / 任务文本）能对齐**，工作量在写 parquet，不在改数据 | 我们有 (T,38) float32 关节角、有每段的自然语言描述、有帧数和 fps |
| **画面部分对不齐，而且不是"精度不够"是"没有"** | 参考三路特征都是 `dtype: video`；我们一路机器人相机画面都产不出（B3/B4） |
| **两处必须张勃拍板才能动**：38 维怎么摆、fps 不统一怎么办 | 都是"改数据本身"，发出去就是别人训练的口径（B5、新增 B9） |
| **一个纯工程缺口**：没有任何 env 装了 `pyarrow`，写 parquet 得先解决依赖 | 三个 env 全试过（新增 B10） |

> **2026-08-22 更新（写完这份文档之后）**：上表后两行里的"必须拍板"和"缺依赖"都已经
> 有答案了 —— 维度不降（38 维 + 临时 `robot_type`）、fps 不重采样（名义 fps + 自定义列）、
> `pyarrow` 已装。逐条口径见 [§6](#6-要拍板的--2026-08-22-全部拍板了)。
> **格式口径整套是临时占位**，等正式规范文档来了再调。前两行的结论不变。

---

## 2. 目标格式实测

### 2.1 目录布局

```
<dataset>/{train,val}/<task_name>/shard-0000-of-0001/
├── meta/
│   ├── info.json                       ← 全部 schema 都在这里
│   ├── tasks.parquet                   (task_index int64, task large_string)
│   ├── stats.json                      数据集级统计，10 个特征 × 10 项
│   ├── episodes/chunk-000/file-000.parquet   一行一个 episode
│   ├── subtask_labels.jsonl            ← 公司扩展，不是 LeRobot 标准
│   └── tenkh_episode_index.jsonl       ← 公司扩展：每个 episode 的源文件绝对路径
├── data/chunk-000/file-{000,001,002}.parquet
└── videos/observation.images.{cam_high,cam_left_wrist,cam_right_wrist}/chunk-000/file-*.mp4
```

`_repair_report.json`（`schema: repair_report/v1`, `tool: filter_tree`）在数据集根，
记录 442 个 healthy（直接软链接）、1 个 repaired、249/250 episode 恢复率 99.6%。
**这就是"公司已有质检体系"的实物**：树级别的完好性修复 + 过滤，和我们 `quality/`
那套"这段视频能不能用"完全不是一层东西（我们的是内容判决，它的是数据完整性）。

### 2.2 `info.json` 的关键字段

```
codebase_version  "v3.0"          robot_type  "yam_bimanual"     fps  30
total_episodes    585             total_frames  2784973          total_tasks  1
chunks_size       1000            data_files_size_in_mb  100     video_files_size_in_mb  200
data_path   "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
video_path  "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
splits      {"train": "0:585"}
action_convention  {"convention": "absolute", "source": "pipeline-detect",
                    "confidence": 0.9968, "at": "2026-07-14T09:38:42"}
```

`features` 一共 10 项：3 路 video + `observation.state` + `action` + `timestamp` +
`frame_index` + `episode_index` + `index` + `task_index`。

- 三路 video：`dtype: "video"`, `shape [3,480,640]`, `names ["channels","height","width"]`,
  `info` 里 `video.codec "h264"` / `video.pix_fmt "yuv420p"` / `video.fps 30.0` /
  `video.is_depth_map false` / `has_audio false`。实测 mp4 与之一致
  （ffprobe：h264 / 640×480 / yuv420p / 30 fps / 243662 帧）。
- `observation.state` 和 `action` 都是 `float32[14]`，`names.motors` 逐个列名：
  `left_joint_1..6` + `left_gripper` + `right_joint_1..6` + `right_gripper`。
  **一只手一个夹爪标量。**

### 2.3 data parquet 的行 schema

```
observation.state  fixed_size_list<float>[14]
action             fixed_size_list<float>[14]
timestamp          float          (float32，= frame_index / fps，实测间隔 0.03333334)
frame_index        int64          (episode 内从 0 连续)
episode_index      int64
index              int64          (**全数据集全局连续**，episodes parquet 里有 dataset_from_index/to_index)
task_index         int64
```

### 2.4 `action` 是怎么来的（这条最有用）

实测 episode 0（3848 帧）：

```
|action[t] - state[t]|    平均 0.009341   最大 0.32378
|action[t] - state[t+1]|  平均 0.0        最大 0.0        ← 逐位相同
|action[-1] - state[-1]|  0.0                             ← 最后一帧复制
state 逐帧变化平均         0.009341
```

**`action[t] ≡ state[t+1]`，最后一帧复制自身。** 也就是说这份数据集的 `action` 不是
真实下发的指令，是把观测状态整体前移一帧生成的（`info.json` 自己写了
`source: "pipeline-detect"`, `confidence: 0.9968`，是**检测出来的约定**而不是录下来的）。

这对我们是好消息：我们手上只有一条运动学解出来的关节轨迹，没有"指令 vs 观测"之分。
按同一个约定填 `state = q[t]` / `action = q[t+1]`（末帧复制），**和参考数据集的口径
一模一样**，不用编任何东西，也不会造出一个 `state == action` 的恒等映射让模型白学。

### 2.5 episodes parquet（一行一个 episode）

585 行。列分四类：

1. 索引：`episode_index` / `tasks list<string>` / `length` / `data/chunk_index` /
   `data/file_index` / `dataset_from_index` / `dataset_to_index` / `meta/episodes/*`
2. 每路视频：`videos/<key>/{chunk_index,file_index,from_timestamp,to_timestamp}`
   —— **v3.0 把多个 episode 打包进同一个 mp4**，靠这两个时间戳定位（实测 episode 0
   是 `0.0 → 128.26666666666668`）
3. 每个特征 × 10 项统计：`stats/<feature>/{min,max,mean,std,count,q01,q10,q50,q90,q99}`。
   一维特征是 `list<double>`；图像特征是 `list<list<list<double>>>`（逐通道，
   实测形状 `[[[v]],[[v]],[[v]]]`，值域已归一到 0–1，`count` = 73881600 = 像素数）
4. 公司扩展：`dense_subtask_names` / `_zh` / `_start_times` / `_end_times` /
   `_start_frames` / `_end_frames`（实测 12 段子任务，英文有值中文为空）

### 2.6 打包规则（实测，不是猜）

| | 规则 | 实测 |
|---|---|---|
| data parquet | 攒到 `data_files_size_in_mb: 100` 换文件 | 105.0 / 105.0 / 34.4 MB，装 250 / 249 / 86 个 episode |
| video mp4 | 攒到 `video_files_size_in_mb: 200` 换文件 | 12 个文件 205–210 MB + 尾巴 56 MB，每个装 50–54 个 episode |
| chunk 目录 | `chunks_size: 1000` 个文件一个 chunk | 全部落在 `chunk-000` |
| episode 长度 | — | min 1493 / max 10283 / 均值 4760.6 帧（30 fps → 50–343 秒） |

---

## 3. 我们现在的产出实测

一次重定向的产物目录（`outputs/retarget/collcmp/-1r9yl-P-Ao_86.3_90.8_grid/`）：

```
trajectory.npz   15 KB    关节角 + 关节名 + fps + robot + clip_id
root_frames.npz   4 KB    R_per_frame (T,3,3) / t_per_frame (T,3) / R_anchor / t_anchor
metrics.npz       9 KB    ik_rate / pos_err / ori_err / jlm / manipulability / roughness …
input_viz.mp4     2 MB    源画面叠手部标注，854×480，**mpeg4**
robot_sim.mp4     2 MB    MuJoCo 渲染的机器人，1440×1080，**mpeg4**
```

`trajectory.npz`（写在 `external/EgoInfinity/retarget/utils/clip_io.py::save_trajectory`）：

| 键 | dtype | shape | 备注 |
|---|---|---|---|
| `q_left` / `q_right` | float32 | (T,7) | 肩 pitch/roll、arm yaw、肘 pitch/yaw、腕 pitch/roll |
| `q_left_fingers` / `q_right_fingers` | float32 | (T,12) | thumb bend/rota1/rota2 + index abd/mcp/pip + middle/ring/pinky 各 mcp/pip |
| `*_joint_names` | object | (7,) / (12,) | 已经是全称，可直接当 `names.motors` |
| `fps` | float32 | () | 实测 15.401785850524902 |
| `robot` / `clip_id` | `<U` | () | `'m7'` / `'-1r9yl-P-Ao_86.3_90.8'` |
| `frame_status_{left,right}` | int8 | (T,) | **只有开 `--traj_cleanup` 才写**；0=ok 1=interp 2=hold 3=rest |
| `frame_cause_{left,right}` | int8 | (T,) | 同上；0=ok 1=missing 2=bad |

L3.4 的 `trajectory.npz` 维度**完全相同**（7+12 ×2），只有 `robot` 字段不同 ——
所以"38 维"这件事对我们现有两台机器人是统一的，不是 M7 特例。

现存 31 个产物目录里，`frame_status_left` 一个都没有（`--traj_cleanup` 默认关）。

### 3.1 我们有几段、多长、多少 fps

| clip_id | 帧数 | 时长 s | fps | `action_brief` |
|---|---|---|---|---|
| `--oo8_XIuOM_799.5_809.8` | 155 | 10.28 | 15.0778 | Insert stopper into cylinder |
| `--oo8_XIuOM_900.3_917.4` | 257 | 17.08 | 15.0468 | Insert coil spring |
| `-0RheyDV3a0_474.8_487.3` | 189 | 12.48 | 15.1442 | Serve cake slice |
| `-0RheyDV3a0_48.6_55.3` | 104 | 6.72 | 15.4762 | Show zucchini |
| `-1bQTExN1Ts_230.8_242.8` | 180 | 12.00 | 15.0000 | Sip coffee |
| `-1r9yl-P-Ao_231.8_241.5` | 146 | 7.93 | **18.4041** | Blow compressed air into fan |
| `-1r9yl-P-Ao_60.4_68.4` | 120 | 8.00 | 15.0000 | Show tool kit |
| `-1r9yl-P-Ao_86.3_90.8` | 69 | 4.48 | 15.4018 | Hold up metal case |
| `-20k07PjLTA_48.0_52.4` | 67 | 4.41 | 15.1927 | Add onions |
| `-2cNMO9Mm3Q_192.4_209.2` | 146 | 9.73 | 15.0000 | Present cake and pipe icing |

合计 1433 帧 ≈ 93 秒。参考数据集单个 episode 的均值就是 4760 帧 —— 我们目前十段
加起来还不到人家一个 episode 的三分之一。规模问题记在 B8。

**fps 逐段不同（15.0000 – 18.4041）**，来源是 `scene.json.fps`（= 切片帧数/时长）。
这是新发现，B5 只记了"15.4 不是 30"，没记"段间还不一致"。见新增的 B9。

### 3.2 任务文本有现成来源（好消息）

`data/clips_official/<clip>/scene.json` → `action100m_metadata`：

```json
"action_brief":    "Hold up metal case",
"action_detailed": "She lifts the removed metal bottom case, steadies it with one hand,
                    and points at it with a pen while presenting it to the camera.",
"actor":           "The presenter, a woman wearing glasses and a pink shirt",
"summary":         "…"
```

`action_brief` 直接就是 `tasks.parquet` 的 `task` 需要的东西。两点格式差异：
参考里是全小写无句点（`"roll the towels"`），我们的是首字母大写祈使句
（`"Hold up metal case"`）；参考一个 shard 一个 task（`total_tasks: 1`），
我们十段十个不同 brief —— 要么一段一个 task（`total_tasks: 10`），要么按 brief 聚类。

另外 `scene.json.video_source` 有 `youtube_id` / `start_seconds` / `end_seconds` / `url`,
正好对应参考里 `tenkh_episode_index.jsonl` 那个"源文件在哪"的位置 —— **溯源字段有先例可循**，
不用自己发明一个。

---

## 4. 逐字段对照

### 4.1 直接可映射（不改数值）

| 目标字段 | 我们的来源 | 说明 |
|---|---|---|
| `frame_index` | `arange(T)` | T = `q_left.shape[0]` |
| `episode_index` | 一个 clip 一个 episode | 我们的 `clip_id` 天然是 episode 粒度 |
| `index` | 全局累加 | 需要一个跨 clip 的计数器（参考里也是全局的） |
| `task_index` | `action_brief` 去重后的下标 | |
| `tasks.parquet.task` | `scene.json.action100m_metadata.action_brief` | 大小写要规范化 |
| `episodes.length` | T | |
| `episodes.tasks` | `[action_brief]` | |
| `info.json.total_{episodes,frames,tasks}` | 统计得到 | |
| `stats/*` 那 10 项 | 从关节角现算 | 一维特征，`np.quantile` 就够 |

### 4.2 需要换算 / 需要定口径

| 目标字段 | 我们有什么 | 缺口 |
|---|---|---|
| `observation.state` `float32[14]` | `(T,38)` float32 | **维度不同**（B5）。若按 §2.4 的约定，`state = q[t]`、`action = q[t+1]`（末帧复制）—— 这条不用拍板，参考数据集自己就是这么干的 |
| `action` | 同上 | 同上 |
| `timestamp` | `frame_index / fps` | fps 逐段不同（B9）。参考是 float32 且严格 `1/30` 等距 |
| `info.json.fps` | 15.0000–18.4041 | 一个 `info.json` 只能写一个数（B9） |
| `info.json.robot_type` | `'m7'` / `'l3_4'` | 参考是 `"yam_bimanual"` 这种"结构+构型"命名，建议 `m7_bimanual_dex`（B5 已记） |
| `names.motors` | `*_joint_names` | 名字风格不同：参考 `left_joint_1`（编号），我们 `left_shoulder_pitch_joint`（语义）。我们的更有信息量，但下游若按名字硬匹配就会挂 |
| `splits` | 无 train/val 划分 | 我们从来没分过训练/验证集 |
| 打包规则 | 无 | 100 MB / 200 MB / 1000 文件那三条要照抄；以我们现在的量级全落在 `file-000` |

### 4.3 我们完全没有

| 目标字段 | 状态 |
|---|---|
| `observation.images.cam_high` | **产不出**。这是"机器人头部相机看到的画面"，需要任务B（视觉合成）先跑通，而任务B卡在 B3（没有 RGB 输入） |
| `observation.images.cam_left_wrist` / `cam_right_wrist` | **产不出**，而且比头部相机更难：腕部相机的视角只有在有机器人本体渲染之后才存在，源视频里根本没有这个视角 |
| `left_gripper` / `right_gripper` 标量 | 我们是 12 自由度灵巧手，不是平行夹爪。**不是缺一个数，是结构不同** —— 要么换 robot_type（B5 的建议），要么把 12 维压成 1 维开合度（会丢掉抓握姿态，那正是这条流水线的核心产出） |
| `dense_subtask_*` / `subtask_labels.jsonl` | 我们没有子任务分段。`action_detailed` 是整段一句话；`signals.json` 有逐帧 `contact_l/contact_r`，理论上能切出"接触段"，但那是接触不是子任务语义 |
| `val` split | 没划过 |

### 4.4 我们有但目标格式没地方放

这些是**信息损失**，不是缺口 —— 但如果直接丢掉，下游就没法复现我们的判断了。

| 我们的东西 | 为什么重要 | 参考格式里的位置 |
|---|---|---|
| `frame_status_*` / `frame_cause_*` | `3=rest` 明确标了"这几帧不能当训练数据"（`traj_cleanup.py` 的注释：*NOT usable data*）。丢了它，插值帧、保持帧、静息位兜底帧在下游看起来和真实检测帧一模一样 | **没有对应字段**。要么加自定义特征（LeRobot 允许扩展，参考自己就加了 `dense_subtask_*`），要么在导出时直接丢掉 `status==3` 的帧 —— 但那会把 episode 切断 |
| `root_frames.npz` 的 `t_per_frame` / `R_per_frame` | 机器人底座位姿。grid 路线实测**逐帧零变化**（静态底座）；neural 路线实测逐帧最大 1.28 mm、整段行程 3.1 cm | 参考的 14 维里没有底座自由度（那台机器人是固定的）。grid 路线下没有信息损失；neural 路线下会丢掉一个几厘米的漂移 |
| `metrics.npz`（`ik_rate` / `pos_err` / `ori_err` / `manipulability` / `roughness`） | 这段数据可信到什么程度，全在这里 | 没有对应字段。建议进 episodes parquet 的自定义列（和 `dense_subtask_*` 同一个位置） |
| `scene.json.video_source`（youtube_id / 起止秒） | 溯源、去重、版权 | 有先例：`tenkh_episode_index.jsonl` 的 `source_episode_path_abs` |
| 碰撞审计结果（B0–B4 那套） | 哪几帧穿了躯干/穿了手 | 没有对应字段 |

---

## 5. 需要动的代码范围（只是估算，本次一行没写）

按"改到谁"排，越往下越不该轻易动：

1. **新增一个导出模块**（比如 `src/web2robot/export/lerobot_v30.py` + `scripts/s6_export.sh`）：
   读 `trajectory.npz` + `scene.json`，写 parquet / info.json / episodes。
   **纯新增，一行现有代码都不用碰** —— 这是唯一一条不影响现有产物的路。
2. **依赖**：`pyarrow` —— 2026-08-22 已装进 `envs/rt_env`（`25.0.1`）并写进
   `envs/requirements-rt.txt`，见 §6 的 B10。
3. **可能要开的开关**：`--traj_cleanup`（否则导出的数据里分不出插值帧和真检测帧）。
   这不改代码，改的是批量脚本的默认调用。
4. **不该动的**：`save_trajectory` / `trajectory.npz` 的字段。导出是**读**它，
   不是改它 —— 一改，`docs/VERIFICATION.md` 里那五条 md5 参照线全部失效
   （`trajectory.npz = 9ef35b4eed590c543ae4af9c9b89e5c9` 那条）。

---

## 6. 要拍板的 —— **2026-08-22 全部拍板了**

五条当时都是"我不该自己定"，用户已逐条给了答案。**整套格式口径明确标注为临时占位方案**，
等张勃的正式格式规范文档到了再回头调整；现在的目标只是让链路跑通、产出看得见的东西。

| 编号 | 用户定的 | 具体口径 |
|---|---|---|
| B4 | 视觉合成**不是可并行支线，是必须的一环** | 最终发布的数据集必须带画面，所以三路视频不能推到第二版；先解决"哪来的 RGB"（B3） |
| B5 | 采用临时占位方案，先跟通链路 | `robot_type` 用真实维度的临时名（`m7_bimanual_dex`）；`action` / `observation.state` **直接写 38 维**；字段名直接用 `trajectory.npz` 里的 `*_joint_names`（已经是全称，不改）；`info.json` 的 `fps` 写名义值 |
| B9 | 方案③ | `info.json` 写一个名义 fps，每段的真实 fps 进 episodes parquet 的**自定义列**（先例是参考数据集自己的 `dense_subtask_*`）。**不做重采样、不按 fps 分 shard** |
| B10 | 批准安装 | `pyarrow==25.0.1` 已装入 `envs/rt_env` 并写进 `envs/requirements-rt.txt`（`pip install --no-deps`，freeze 前后 diff 只多这一行，numpy 仍 2.2.6；365 个测试全绿） |
| B11 | 现在不改 | 只有**真正要打包发布**的视频才用 `-c:v libx264` 生成；现有调试产物（`robot_sim.mp4` 等）保持 mpeg4 不动，`docs/VERIFICATION.md` 里已建立的 md5 基线一条都不碰 |

§1–§5 的实测结论不受这些决定影响（它们量的是现状），只有 §4.2「需要换算」那几行的
处置口径按上表定死了：**维度不降、fps 不重采样。**

---

## 7. 复现命令

```bash
# 参考数据集：布局与 schema（系统 python 有 pyarrow 24.0.0，我们三个 env 都没有）
S=/mnt/vlm/common/datasets/ABC-130k_lerobot_v30/train/roll_the_towels/shard-0000-of-0001
cat $S/meta/info.json
/usr/bin/python3 -c "import pyarrow.parquet as pq; print(pq.ParquetFile('$S/data/chunk-000/file-000.parquet').schema_arrow)"
/usr/bin/python3 -c "import pyarrow.parquet as pq; print(pq.ParquetFile('$S/meta/episodes/chunk-000/file-000.parquet').schema_arrow)"
ffprobe -v error -select_streams v:0 -show_entries stream=codec_name,pix_fmt,width,height,r_frame_rate,nb_frames \
        -of default=nw=1 $S/videos/observation.images.cam_high/chunk-000/file-000.mp4

# action ≡ state 前移一帧（§2.4）
/usr/bin/python3 - "$S" <<'EOF'
import sys, numpy as np, pyarrow.parquet as pq
t = pq.ParquetFile(f"{sys.argv[1]}/data/chunk-000/file-000.parquet").read_row_groups([0,1]).to_pydict()
ep = np.array(t["episode_index"]); m = ep == 0
st = np.array(t["observation.state"], dtype=np.float64)[m]
ac = np.array(t["action"],           dtype=np.float64)[m]
print("|a[t]-s[t+1]| 最大", np.abs(ac[:-1]-st[1:]).max(), " |a[-1]-s[-1]|", np.abs(ac[-1]-st[-1]).max())
EOF

# 我们的产出
PYTHONPATH=src envs/rt_env/bin/python -c "
import numpy as np
z = np.load('outputs/retarget/collcmp/-1r9yl-P-Ao_86.3_90.8_grid/trajectory.npz', allow_pickle=True)
for k in z.files: print(k, z[k].dtype, z[k].shape)
print('fps', float(z['fps']))"

# 每段的 fps / 帧数 / 任务文本
PYTHONPATH=src envs/rt_env/bin/python -c "
import json, glob, os
for c in sorted(glob.glob('data/clips_official/*/scene.json')):
    d = json.load(open(c))
    n = json.load(open(os.path.join(os.path.dirname(c), 'hand_meta.json')))['n_frames']
    print(f\"{d['id']:28s} {n:4d} {d['duration']:6.2f} {d['fps']:.4f}  {d['action100m_metadata']['action_brief']}\")"
```
