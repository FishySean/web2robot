# 第0步备份 —— 2026-08-06

重构前的保险。备份的是**一个多月来从未被任何 git 记录过**的工作：它们全是第三方 clone
（`EgoInfinity`、`HaWoR`）里的 `git status` 条目，一次 `git checkout .` 或 `git pull` 就会消失。

**这个目录在迁移全部验证通过之前不要删。**

## 内容

| 文件 | 是什么 | md5 |
|---|---|---|
| `egoinfinity-modified.patch` | 我们对上游 **6 个已跟踪文件**的改动（+284 行接线） | `86ee23d6488d6407227daa031e250252` |
| `egoinfinity-ours.tar.gz` | 我们**新增的 128 个文件**（18 个 py + 110 个 M7 资产，21 MB 原始） | `0dad4b68782ad89abc8e72f9f8ca9b27` |
| `egoinfinity-ours.filelist` | 上面那个 tar 的来源清单（`git ls-files --others`） | — |
| `hawor-ours.tar.gz` | `convert_hawor_to_clip.py`（HaWoR→EgoInfinity clip 桥接） | `7318ea2958270402e36fa915285e9a38` |
| `requirements-rt.txt` | rt_env 166 个包（jax 0.6.2 / mujoco 3.6.0 / torch 2.4.1 / numpy 2.2.6） | `e8f9e2f7da81ef6325e09585d8000ef5` |
| `requirements-hawor.txt` | hawor_env 134 个包 | `f2b2cd26af83fa06f3834a5ebb5713b4` |
| `requirements-perception.txt` | hand2robot/env 112 个包 | `1a3529615cfbb27aafcd861a69150d54` |

三份 requirements 是**此前完全不存在**的东西：三个 venv 共 21 GB / 14 万文件，之前没有任何
lock 文件，等于不可复现。现在冻下来了。

## 校验结果（两项都过，不是"看着像对"）

**校验 1 —— tar 逐文件 md5：** 解压后与原件比对 **128 个文件，不一致 0 个**。
（128 = 18 个 py + 110 个 M7 资产，与 `find` 独立计数吻合。）

**校验 2 —— patch 可重放：** 从上游原始版本（`git show HEAD:<file>`）重新应用 patch，
结果与当前工作区**逐字节一致**，6/6 全 OK。这比 `git apply --check` 强：它证明的不是"能打上"，
而是"打上之后确实还原成现在这个样子"。

```
patch 应用成功
  OK  retarget/kinematics/wilor_retargeter.py
  OK  retarget/kinematics/wrist_ik.py
  OK  retarget/scripts/test.py
  OK  retarget/scripts/train.py
  OK  retarget/sim/robots/__init__.py
  OK  retarget/utils/clip_io.py
```

## 怎么还原

```bash
cd /mnt/vlm/fanshaoheng/EgoInfinity
git apply /mnt/vlm/fanshaoheng/_backup_20260806/egoinfinity-modified.patch
tar xzf /mnt/vlm/fanshaoheng/_backup_20260806/egoinfinity-ours.tar.gz
cd /mnt/vlm/fanshaoheng/HaWoR
tar xzf /mnt/vlm/fanshaoheng/_backup_20260806/hawor-ours.tar.gz
```

## 故意排除的东西

| 排除项 | 原因 |
|---|---|
| `retarget/rt_env/`（7.1 GB） | venv，不可搬移；依赖已由 `requirements-rt.txt` 记录 |
| `retarget/examples/`（0.27 GB） | 重定向输出产物，不是代码；原地保留 |
| `artifacts/`、`tools/_batch_logs/` | 运行日志 |
| `HaWoR/example/`、`hawor_env/`、`*.log` | 同上（HaWoR 那边只有 1 个文件是我们写的） |
