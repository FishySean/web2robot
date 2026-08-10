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

2026-08-10 重新从 `EgoInfinity` 工作区导出，覆盖 6 个被改过的**已跟踪**文件
（6 files changed, 313 insertions(+), 15 deletions(-)，28982 字节）。

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
