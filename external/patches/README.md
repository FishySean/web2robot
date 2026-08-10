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

2026-08-10 `retarget/` 迁移后重新导出，覆盖 6 个被改过的**已跟踪**文件
（6 files changed, 233 insertions(+), 15 deletions(-)，23926 字节）。

**注意这个数字是往下走的**：上一版 313 insertions，这一版 233。迁移做对了的话
patch 就该变小 —— 逻辑挪进 `src/web2robot/`，上游只剩接线。哪天它又开始变大，就是
有人在往上游文件里写实质逻辑了。

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
