# external/patches/ —— 上游仓库的改动

`external/EgoInfinity` 和 `external/HaWoR` 是第三方仓库的 symlink，**里面不改代码**。
我们对上游的改动只以两种形式存在：

1. **逻辑归我方** —— 碰撞检测、坏帧兜底、M7 定制这些实质逻辑，作为一等公民放在
   `src/web2robot/` 下，不寄生在上游目录里。
2. **一个 patch 文件** —— 上游文件里真正必须改的注入点（`test.py` 的 argparse 接线
   ＋ 指向我方包的 import），记在 `egoinfinity-modified.patch`。

这样做的依据是量出来的，不是猜的：我们那 4 个核心模块（998 行）只 import
numpy / mujoco / scipy，**零上游 import**；`test.py` 那 +190 行的 diff 是参数接线
和调用点。所以"逻辑搬走 + 留一个 patch 记注入点"是行得通的，不会把上游代码拆散。

## egoinfinity-modified.patch

2026-08-21 把 `--root_solver` 的默认值改成 `grid` 之后重新导出，覆盖 6 个
被改过的**已跟踪**文件（6 files changed, 530 insertions(+), 32 deletions(-)，46139 字节）。

**这一版数字是往上走的，原因见下面 2026-08-18 起的六节** ——
上一版 520
insertions。一般来说 patch 变大就是有人在往上游文件里写实质逻辑，所以每次变大都必须在
这里写清楚多出来的是什么；这六次多的分别是一个 if/else 开关＋把新求解器包成 callable 的
接线、两个默认关闭的开关、一处"去 `presets.py` 查表 + 允许 CLI 覆盖"的接线、一台新机器人
在注册表/IK/手部重定向器三处的注册、和一次默认值翻转（10 行全是注释，零新代码），肉仍然在
`src/web2robot/retarget/root_grid.py`、`src/web2robot/twin/`、`src/web2robot/refine/`、
`src/web2robot/collision/presets.py` 和 `src/web2robot/robots/l3_4/`。

```bash
cd external/EgoInfinity && git apply ../patches/egoinfinity-modified.patch
```

校验方式（比 `git apply --check` 强）：把 `git show HEAD:<file>` 的原始版本铺进临时目录、
replay 这个 patch、再和真实工作区逐字节 md5 比对 —— 6/6 完全一致。
`--check` 只能说"能打上"，这个能说"打上之后就是现在的状态"。

### 这一版和上一版的区别：4 个模块已经不在上游了

`collision/` + `trajectory/` 迁移（2026-08-07）之后，上游那 4 份实现文件
（`models/{arm_torso_filter,capsule_collision,dual_hand_filter}.py`、
`utils/traj_cleanup.py`）已从上游工作区**删除**，只在 `src/web2robot/` 下有一份。
所以 `test.py` 的 diff 现在除了 argparse 接线，还包含 6 行 import 改写：

```
from models.arm_torso_filter import ArmTorsoFilter
    → from web2robot.collision.arm_torso_filter import ArmTorsoFilter
from utils.traj_cleanup import ...
    → from web2robot.trajectory.traj_cleanup import ...
```

**留两份是比留一份更危险的做法** —— 下一次改代码会改到错的那一份，而这正是这次重构
要消灭的失效模式。删除前的顺序是：新位置先提交（`7a71fb4`）→ md5 比对确认内容一致
→ 才删上游 → 再端到端跑一次确认整条链完全走新包（`ArmTorsoFilter` 有开火：
左手 fixed 19/20、右手 5/9）。

打这个 patch 的人要知道：`test.py` 需要 `web2robot/src` 在 `PYTHONPATH` 里才能跑，
`scripts/s4_retarget.sh` 已经替你设好了。

### 2026-08-09 又多了一处：`wrist_ik.py` 的 M7 MJCF 路径

`robots/m7/` 迁移之后，上游 `kinematics/wrist_ik.py::RobotIKConfig.m7` 里那行

```python
mjcf_path = _ROBOTS_DIR / "m7" / "m7.xml"      # 指向已删除的上游目录
    → from web2robot.robots.m7 import MJCF_PATH as _M7_MJCF
      mjcf_path = _M7_MJCF
```

必须改，否则建 IK 串链时 `FileNotFoundError`。import 写在函数体里而不是模块顶层，
是为了让 g1/r2/franka 这些**上游自带的**机器人在没有 web2robot 的环境里也能 import
这个模块 —— 只有真要 M7 时才要求我方包在 path 上。

**这一处是被端到端跑出来的，不是看代码看出来的**，值得记一笔：迁移时我用
`grep -rn "robots\.m7\|robots/m7"` 找上游对 M7 的引用，而这行路径是**拼**出来的
（`_ROBOTS_DIR / "m7" / "m7.xml"`），源码里压根没有 `robots/m7` 这个字样，于是躲过了
grep。更麻烦的是：删上游旧目录**之前**跑的那次端到端是绿的 —— 那会儿旧文件还在，
它悄悄读的是旧文件。所以"删除后必须再跑一次端到端"这一步不是形式，它是唯一能发现
这类漏网的手段。同一次 grep 还漏了 `scripts/generate_m7_mjx.py`（它写的是
`Path(__file__).parents[1] / "robots" / "m7"`），那份重复副本也已删除。

现在这两件事都由 `tests/test_m7_robot.py::TestUpstreamAssetPaths` 钉住：它不 grep，
而是**把 IK config 造出来看 `mjcf_path` 存不存在**，拼接的路径也躲不掉；另一个用例
断言上游不再残留 `robots/m7/`、`sim/robots/m7/`、`scripts/generate_m7_mjx.py`。
（把修复临时改回旧路径验证过：测试确实变红，报的就是端到端那条
`FileNotFoundError` 的同一个路径。）

### 2026-08-10 `retarget/` 迁移：patch 从 313 行缩到 233 行

`test.py` 里原本内联的四段逻辑搬进了 `src/web2robot/`：

| 搬走的 | 新家 |
|---|---|
| 手腕清洗＋三档补洞的调用编排 | `retarget/fallback.py` |
| rest 姿态兜底、relax 松弛 | `retarget/fallback.py` |
| best-of-N 锚点采样（含 per-sample seed 偏移） | `retarget/anchor.py` |
| 叠字文案 | `retarget/fallback.py` |
| M7 的 IK 链根/末端/关节限位 | `robots/m7/ik_config.py` |

上游只留调用点。这也是为什么 `test.py` 的 diff 从 +190 变成 +151：**接线还在，肉没了**。

`ik_config.py` 这一处顺手减了一份重复：链根/末端的 body 名（`waist_pitch_link` 等）
原来在我方也写死了一遍，现在改成从 `robots/m7/config.py::CONFIG` 里取。理由是这类
名字写死两份，改 MJCF 时必然漏一处，而漏了**不报错** —— pytorch_kinematics 会拿旧
名字建一条空链，IK 全帧失败但不抛异常。改完用 `ik_spec()` 的 JSON md5
（`29e67f03d53117b197087f6c4b6f5daa`）前后比对确认行为没变。

### 2026-08-10 `perception/` 迁移：HaWoR 侧一行上游代码都没改

`external/HaWoR` 的 `git status --porcelain | grep -v "^??"` 是**空的** —— 我们从来
没改过 HaWoR 的任何已跟踪文件，所以它不需要 patch，一个字都不用记。

我方那个寄生文件 `convert_hawor_to_clip.py`（未被上游 git 跟踪，就是我们塞进第三方
checkout 里的）已经删掉，逻辑进了 `src/web2robot/perception/{to_clip,hawor}.py`。
删之前先证明新模块**逐字节复现**它的三个产物
（`hand_joints.bin` `783a7de5…` / `hand_meta.json` `353fb9a9…` / `scene.json` `5657ced0…`），
备份也在 `_pre_migration_snapshot/hawor-ours.tar.gz` 里（`91bee7a9…`，删除前核对过）。

`external/HaWoR` 里还剩 68 MB 的 `webvid/`（我们爬的片段、抽帧、contact sheet）和
34 个 `.log` —— 同一类"产物落在 external/ 里"的问题，但那是素材不是代码，单独处理。

### 2026-08-10 `webvid/` 和日志清空：`external/HaWoR` 里再没有我们的散件

`webvid/` 那 436 个文件按"能不能重新算出来"分了两处，目录本身已删除：

| 去哪了 | 什么 | 数量 / 体积 |
|---|---|---|
| `data/webvid/raw/` | 6 段 mp4 + 1 段 mkv 原片，**手工找的，没有下载脚本可以重来** | 7 / 55 MB |
| `outputs/archive/webvid_2026-07/` | 抽帧、contact sheet、demo 渲染，都能从上面 7 段重算 | 429 / 13 MB |

`436 = 7 + 429` 对上了，6 个 mp4 搬家前后 md5 逐个核对 OK（清单在
`data/webvid/raw/MANIFEST.md5`，`md5sum -c` 可随时复核）。

34 个 `.log` 里有 3 个不是垃圾：`run_{abf12,smu41,mc4}.log` 记着 HaWoR **每段现估的
度量尺度**（0.19 / 2.34 / 3.92，差 20 倍）和"focal 走默认 600"这两件事，而深度误差表
的可复现性全靠它们。这 3 个压缩后（22 KB）进了
`evidence/depth_benchmark_ho3d/provenance/`，压缩副本和原件 md5 逐个核对一致之后才删原件，
数值本身由 `tests/test_depth_benchmark.py::TestProvenance` 钉住。其余 31 个（装环境、
下权重、调试）删了 —— 它们记的是"环境怎么装起来的"，那件事该由 `envs/` 说清楚。

还留在 `external/HaWoR` 里的只剩两样，都**不是散件**，故意留着：
`example/`（1.3 GB，HaWoR 自己的 example 目录 + 各次运行的产物，freeze 脚本的旧复现路径
指着它；证据已冻成 24 KB npz，所以它现在只是可选的重跑入口）和 `hawor_env/`（6.2 GB venv，
`envs/hawor_env` 就是它的 symlink）。这两样要不要动是另一件事，不在这次范围里。

## _pre_migration_snapshot/ —— 临时保险，迁移完成后可以从最新提交里删掉

重构开始前（2026-08-06）的一次性快照，防的是"一个 `git checkout .` 或 `git pull`
把一个多月的未跟踪工作全冲掉"这件事 —— 那才是这次重构真正的动因。

- `egoinfinity-ours.tar.gz`（7.7 MB，128 个文件 = 18 个 .py + 110 个 M7 资产）
- `hawor-ours.tar.gz`
- `egoinfinity-ours.filelist`
- `README.md` 记录两项校验的过程和取回命令

**它会跟迁移后的正式文件重复**（M7 资产最终以 `assets/robots/m7/` 的真实文件形式提交）。
留着是因为迁移还没做完；`collision/`、`trajectory/` 已于 2026-08-07 迁完并各自跑过验证，
等 `robots/m7/`、`retarget/`、`perception/` 也迁完之后，这个目录就可以从最新提交里移除
（内容仍在 git 历史里，随时 `git show` 取回）。

### 2026-08-10 再多一处：`test.py` 的 `--out` 落点兜底

上游 `--out` 的默认值是 `clip_path.parent / robot_name`，也就是**把产物写在输入素材
旁边** —— 素材在 `external/` 里，于是产物也进了 `external/`（实测攒了 408 MB / 243 个
mp4+npz，上游 git 只跟踪其中 1 个）。薄壳 `scripts/s4_retarget.sh` 已经顶掉这个默认值，
patch 里再兜一道，防止有人绕过薄壳直接跑 `test.py`：

```python
out_dir = Path(args.out).resolve() if args.out else clip_path.parent / robot_name
    → 后面接一行 out_dir = P.check_output_dir(out_dir)   # 落在 external/ 里就 SystemExit
```

同样的判据有测试钉着：`tests/test_outputs_not_in_external.py`。

### 2026-08-18 `--root_solver`：patch 从 233 行长到 342 行，多出来的是开关不是逻辑

新增的第二条根位姿路线（静态网格搜索，Qwen-RobotManip 公式 3）在
`src/web2robot/retarget/root_grid.py`，**零上游 import**、纯 numpy 单测
（`tests/test_root_grid.py`，44 例）。上游 `test.py` 里多出来的是：

| 多出来的 | 行数（`git diff -w`） | 是什么 |
|---|---|---|
| 7 个 `--grid_*` / `--root_solver` argparse 选项 | ~30 | 参数接线（含 `--grid_tie_break`，帮助文本里写了 66.7% vs 100% 那笔账） |
| `if args.root_solver == "grid":` 分支 | ~55 | 把上游 FK / IK / `cam_to_root_targets` 包成 callable 传给我方求解器 |
| 原 Steps 1+3+4+5 挪进 `else:` | 0（`-w` 下算 context） | 纯缩进 |

所以 `test.py` 从 151 → 233 insertions，全 patch 233 → 342。分支体里没有一条
"方法"语句：关键帧怎么选、网格怎么排、剪枝、平局怎么破，全在我方模块里。

判据仍然是"分支里有没有出现只有这里才有的逻辑"，不是行数本身。这次多出来的
55 行里唯一值得盯的是两个闭包（`_fk_norm` / `_converged`）——它们只做
tensor 装箱和 `.cpu().numpy()`，判据（`info["converged"]`）是上游的，没有另立标准。

`r_max` 在这里**现量**（`estimate_reach` 打 FK，M7 实测 1.007 m），不写死常数，
换机器人不用改上游文件。

校验照旧：`git show HEAD:<file>` 铺进 `/tmp/patch_replay` → replay 这个 patch →
和真实工作区逐字节 md5 比对，**6/6 完全一致**（2026-08-19 复核过）。

### 2026-08-19 `--object_tracking`：342 → 367，多出来的 25 行全是接线

物体 6D 位姿跟踪（EgoEngine arXiv 2606.12604 §3.1 数字孪生）在
`src/web2robot/twin/`，**零上游 import**、也不反向依赖 `root_anchor.py` /
`root_grid.py`，纯 numpy 单测（`tests/test_twin_object_pose.py`，58 例）。
上游 `test.py` 里多出来的是：

| 多出来的 | 行数 | 是什么 |
|---|---|---|
| `--object_tracking off\|on` + `--object_source` | ~10 | 参数接线，默认 `off` |
| `if args.object_tracking == "on":` 调用点 | ~15 | 调 `track_objects()` / `save_object_poses()`，帧数帧率用 `seq` 的口径传，保证和手部轨迹逐帧对齐 |

**默认 off 的含义是"一行都不执行"，不是"执行了但结果一样"** —— import 也在 if 里面，
不传参数时 `web2robot.twin` 根本不会被加载。这一条是量出来的：同一段片段
（`-1r9yl-P-Ao_86.3_90.8`，M7 / neural / seed 0）跑 base / off / on 三遍，
`trajectory.npz`、`root_frames.npz`、`metrics.npz`、`robot_sim.mp4`、`input_viz.mp4`
五个产物 md5 三份全同，`on` 只多出一个 `object_poses.npz`
（跑法：`scripts/dev/check_object_tracking_bytes.sh`）。

### 2026-08-19 `--action_refine`：367 → 428，多出来的 61 行是开关＋两处取数

动作分级精修的判决（EgoEngine §3.2.2 自适应模式切换）在 `src/web2robot/refine/`，
**零上游 import**、纯 numpy 单测（`tests/test_refine_action.py`，55 例）。
上游 `test.py` 里多出来的是：

| 多出来的 | 行数 | 是什么 |
|---|---|---|
| `--action_refine none\|mpc\|rl` + 4 个 `--refine_*` | ~25 | 参数接线，默认 `none`。λp / λR / 每帧预算论文都没给数值，所以做成参数 |
| `run()` 开头的参数矛盾检查 | ~8 | `mpc\|rl` 没配 `--object_tracking on` 就当场 `SystemExit`，**不静默降级** |
| `if args.action_refine != "none":` 调用点 | ~28 | 取两样东西喂给我方模块：IK 目标（`left_pos/left_quat` …）和 IK 实际（`opt.ik_*_traj._fk(q)`），外加 `root_frames` 用来换系 |

那个 `_fk7` 闭包是唯一值得盯的地方：它只做 tensor 装箱和 `.cpu().numpy()`，用的是上游
自己的 `_fk`，没有另写一份运动学。**比的是 IK 残差（实际 FK vs 交给 IK 的目标）**，
不是"机器人手 vs 人手"—— 后者本来就差一截（`workspace_center` 会整体平移目标），
拿绝对位姿比没有意义。这个判断写在调用点的注释里，别在重构时顺手改掉。

同一道"默认关闭 ⇒ 逐字节不变"的检查：`scripts/dev/check_action_refine_bytes.sh`
跑 base / `none` / `on+none` / `on+mpc` 四遍，2026-08-19 实测原有 5 个产物 md5 四份
全同；`on+none` 只多 `object_poses.npz`，`on+mpc` 再多
`action_refine.json` / `action_refine.npz` / `hand_poses.npz`。

### 2026-08-20 `--atf_preset`：428 → 459，多出来的 31 行是"按路线取标定参数"的接线

`--root_solver grid` 把底座搜到手边，手臂贴身的角度和频次都和 `neural` 不同，所以臂-躯
代理盒的余量得分路线标定（13 段实测残留穿躯 28.9% vs 23.8%）。标定表在
`src/web2robot/collision/presets.py`，标定脚本
`scripts/dev/sweep_arm_torso_params.py`。上游 `test.py` 里多出来的是：

| 多出来的 | 行数 | 是什么 |
|---|---|---|
| `--atf_preset` + `--atf_torso_half/enter_thresh/margin` | ~20 | 参数接线。`--atf_preset auto`（默认）跟着 `--root_solver` 走；`legacy` = 过滤器自己的未标定默认值，专门留给"调参前 vs 调参后"的 A/B |
| `ArmTorsoFilter(...)` 构造点前的 5 行 | ~11 | 取预设 + 让显式 CLI 覆盖压过预设 |

**没有一个数字写在上游文件里** —— 盒半长和两个门槛全在 `presets.py`，上游只知道"去查
表"。这一点是故意的：换机器人或重新标定时不该动 patch。

`neural` 那条路线的预设是**空字典**，即"照旧构造"。这不是口头承诺：
`scripts/dev/check_neural_bytes.sh` 同一段片段（`-1r9yl-P-Ao_86.3_90.8`，M7 / neural /
seed 0 / 两条碰撞过滤都开）跑 base 和 `--atf_preset auto` 两遍，2026-08-20 实测
`trajectory.npz` / `metrics.npz` / `robot_sim.mp4` md5 两份全同
（`205d96dba4a701e4be19a88ff1ec0483` 是那个 mp4）。另有不用 GPU 的一半守在
`tests/test_module_boundaries.py::TestArmTorsoPresets`（预设为空、键必须真是构造参数、
返回副本、盒不得大于真实网格 AABB、没标定过的路线要抛而不是套用 grid 的数）。

校验照旧：`git show HEAD:<file>` 铺进 `/tmp/replay` → replay 这个 patch → 和真实工作区
逐字节 md5 比对，**6/6 完全一致**（2026-08-20 复核过，41857 字节）。

### 2026-08-20 第二台机器人 L3.4：459 → 520，多出来的 61 行是三处注册，一个数字都没写在上游

L3.4（rel3_4）的机器人定义在 `src/web2robot/robots/l3_4/`（五个模块，**零上游 import、
也不 import `robots/m7/`**），资产由 `scripts/dev/build_l3_4_assets.py` 从厂家原包生成
（七步自检），单测 `tests/test_l3_4_robot.py`（12 例）。上游三个文件里多出来的是：

| 文件 | 行数 | 是什么 |
|---|---|---|
| `sim/robots/__init__.py` | +14 | 注册表三张表各加一行 `"l3_4"`，外加 `RobotConfig(**ENV_SPEC)` 一行把我方纯 dict 包成上游 dataclass |
| `kinematics/wrist_ik.py` | +23 | `RobotIKConfig.l3_4(side)`：函数体内 import 我方 `ik_spec()` + `MJCF_PATH`，把限位转 `torch.tensor` |
| `kinematics/wilor_retargeter.py` | +24 | `_l3_4_12dof_from_keypoints(side)`（把我方 `HAND_JOINT_SPEC` 包成 `RobotHandConfig`）+ 工厂里一个 `if robot == "l3_4"` |
| `scripts/test.py` / `scripts/train.py` | ±0 | `--robot` 的 `choices` 里多一个字符串（替换同一行，所以 insertions 没变） |

**链根、末端帧、7×2 臂限位、12 个手指限位、锁死的 17 个自由度，一个数字都不在上游文件里。**
和 M7 同一个套路：上游只知道"去 `web2robot.robots.<name>` 取"。换机器人不该动这个 patch，
这一条由 `tests/test_l3_4_robot.py::TestUpstreamWiring` 钉住（造出 `RobotIKConfig.l3_4`
看 `mjcf_path` 存不存在、和我方纯数据表逐位相等，拼接出来的路径也躲不掉）。

方法名必须**恰好**是 `l3_4`：上游 dispatch 是 `getattr(RobotIKConfig, cfg["ik_robot"])`
（`models/root_opt.py` / `scripts/train.py` / `scripts/viz_trajs.py` 各一处），名字错了不会在
import 时报，而是跑到建 IK 串链那一步才 `AttributeError`。同理**不要**图省事写
`l3_4 = m7`（alias）—— 真源是各自的 MJCF，alias 会在哪天两台机器人限位真的分叉时静默地
继续用错的那份。

`--robot` 默认值仍是上游原样，choices 只增不改。但"对 M7 没影响"这句不能只靠读代码：
`sim/robots/__init__.py` 里那两行 `from web2robot.robots.l3_4 import ...` 是**模块顶层**的，
跑 M7 也会执行（加载 L3.4 的 MJCF 路径、构造 `RobotConfig`）。所以照旧过一道字节级检查：
[`check_m7_unchanged_by_l3_4.sh`](../../scripts/dev/check_m7_unchanged_by_l3_4.sh) 拿这份代码
再跑一遍 M7（同片段 / 同 seed / neural / 两条碰撞过滤都开），和 L3.4 改动**之前**留下的
`outputs/dev/neural_bytecheck/base/` 比 md5 —— 2026-08-20 实测
`trajectory.npz` / `metrics.npz` / `robot_sim.mp4` **三个 SAME**
（那个 mp4 仍是 `205d96dba4a701e4be19a88ff1ec0483`，和上一节的数字对得上）。

校验照旧：`git show HEAD:<file>` 铺进 `/tmp/replay_l34` → replay 这个 patch → 和真实工作区
逐字节 md5 比对，**6/6 完全一致**（2026-08-20，45256 字节）。

### 2026-08-21 `--root_solver` 默认值改成 `grid`：520 → 530，多出来的 10 行全是注释

**这一版没有一行新代码** —— 唯一的实质改动是 argparse 里一个字符串
（`default="neural"` → `default="grid"`），多出来的 10 行是帮助文本重写 + 一段
把依据写在旁边的中文注释（13 段 A/B 的四个数字、为什么按穿模而不是按 ρ̄ 取默认）。
数字本身在 `docs/VERIFICATION.md`，这里只留结论，免得读上游文件的人以为默认值是随手定的。

拍板依据（碰撞过滤参数按路线分开标定**之后**的 13 段官方片段）：
`grid` 可行率 96.7% / 残留穿躯 13.3% / ρ̄ 0.393，`neural` 87.1% / 23.8% / 0.441。
grid 唯一输的是 ρ̄ 更偏离 Ego2Robot 的 0.65。

**改默认值不会让前面几节的字节比对失效**：四个守卫脚本
（`check_neural_bytes.sh` / `check_m7_unchanged_by_l3_4.sh` /
`check_object_tracking_bytes.sh` / `check_action_refine_bytes.sh`）**都显式传
`--root_solver neural`**，所以它们比的一直是同一条路线；显式选 `neural` 的行为
逐字节不变。反过来说，以后写新的字节比对脚本也必须显式传这个参数，别再依赖默认值。

一个容易被过度宣传的点：grid 不用模型出根位姿，但 `test.py` 是**无条件** `_load_model`
的，IK 求解器（`opt.ik_left/right`）挂在那个对象上，所以**裸跑 grid 仍然要 `--ckpt`**。
帮助文本里写的是 "training-free"（不需要训根位姿模型），不是 "checkpoint-free"。

校验照旧：`git archive HEAD` 铺进临时目录 → replay 这个 patch → 和真实工作区逐字节
md5 比对，**6/6 完全一致**（2026-08-21，46139 字节）。
