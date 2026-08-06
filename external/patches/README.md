# external/patches/ —— 上游仓库的改动

`external/EgoInfinity` 和 `external/HaWoR` 是第三方仓库的 symlink，**里面不改代码**。
我们对上游的改动只以两种形式存在：

1. **逻辑归我方** —— 碰撞检测、坏帧兜底、M7 定制这些实质逻辑，作为一等公民放在
   `src/web2robot/` 下，不寄生在上游目录里。
2. **一个 patch 文件** —— 上游文件里真正必须改的注入点（主要是 `test.py` 的
   argparse 接线），记在 `egoinfinity-modified.patch`。

这样做的依据是量出来的，不是猜的：我们那 4 个核心模块（998 行）只 import
numpy / mujoco / scipy，**零上游 import**；`test.py` 那 +179 行的 diff 是纯参数接线。
所以"逻辑搬走 + 留一个 patch 记注入点"是行得通的，不会把上游代码拆散。

## egoinfinity-modified.patch

2026-08-06 从 `EgoInfinity` 工作区导出，覆盖 6 个被改过的**已跟踪**文件。

```bash
cd external/EgoInfinity && git apply ../patches/egoinfinity-modified.patch
```

校验方式（比 `git apply --check` 强）：把 `git show HEAD:<file>` 的原始版本铺进临时目录、
replay 这个 patch、再和真实工作区逐字节 md5 比对 —— 6/6 完全一致。
`--check` 只能说"能打上"，这个能说"打上之后就是现在的状态"。

## _pre_migration_snapshot/ —— 临时保险，迁移完成后可以从最新提交里删掉

重构开始前（2026-08-06）的一次性快照，防的是"一个 `git checkout .` 或 `git pull`
把一个多月的未跟踪工作全冲掉"这件事 —— 那才是这次重构真正的动因。

- `egoinfinity-ours.tar.gz`（7.7 MB，128 个文件 = 18 个 .py + 110 个 M7 资产）
- `hawor-ours.tar.gz`
- `egoinfinity-ours.filelist`
- `README.md` 记录两项校验的过程和取回命令

**它会跟迁移后的正式文件重复**（M7 资产最终以 `assets/robots/m7/` 的真实文件形式提交）。
留着是因为迁移还没做完；等 `collision/`、`trajectory/`、`robots/m7/`、`retarget/`、
`perception/` 全部迁完并各自跑过验证之后，这个目录就可以从最新提交里移除
（内容仍在 git 历史里，随时 `git show` 取回）。
