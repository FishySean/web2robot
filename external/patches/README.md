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

2026-08-07 重新从 `EgoInfinity` 工作区导出，覆盖 6 个被改过的**已跟踪**文件
（6 files changed, 284 insertions(+), 15 deletions(-)，27104 字节）。

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
