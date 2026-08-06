# web2robot —— 网络视频 → 机器人动作训练数据

```
海量从网上抓取的视频
    │
    ├─①  取景质检        拍全了没有 / 镜头稳不稳 / 背景合不合适        src/web2robot/quality/
    ├─②  视角与运动分类   第一或第三人称 / 相机动不动 → 选技术路线      src/web2robot/routing/
    ├─③  感知前端        WiLoR+MoGe（相机固定）或 HaWoR（相机运动）    src/web2robot/perception/
    ├─④  重定向          EgoInfinity 官方框架 + M7 定制               src/web2robot/retarget/
    ├─⑤  碰撞检测与平滑   臂-躯 / 双手 / 手指，坏帧兜底                src/web2robot/collision/
    │                                                              src/web2robot/trajectory/
    └─→  干净、可信的机器人动作训练数据
```

**第①步的输出不是二元的合格/不合格。** 每一段通过的片段都带路由标签 —— 因为
"第一人称"不是缺陷，它只是走另一条技术路线；二元门会把那条分支饿死。

## 目录

| 位置 | 是什么 |
|---|---|
| `src/web2robot/` | 全部逻辑。模块按**流水线环节**命名，一个环节一个包 |
| `configs/paths.yaml` | **全工程唯一允许出现绝对路径的地方**。换机器只改这一个文件 |
| `scripts/` | 薄壳，只负责"用对的解释器 + 设好 PYTHONPATH"，不含逻辑 |
| `scripts/dev/` | 开发期工具（回归比对之类） |
| `external/` | 第三方仓库的 symlink（EgoInfinity / HaWoR），**里面不改代码** |
| `external/patches/` | 我们对上游的改动：逻辑归我方 + 一个 patch 记注入点 |
| `envs/` | 三个 venv 的 symlink + `requirements-*.txt` 依赖清单 |
| `assets/robots/m7/` | M7 的 MJCF / URDF / mesh（我们自己产出的资产，进 git） |
| `tests/` | stdlib unittest（秒级）+ `tests/regression/` 回归基准 |
| `data/` `outputs/` | 素材与产物，都不进 git |

## 跑起来

这台机器上 **`conda activate` 不生效**，一律用绝对路径的解释器；薄壳已经替你处理好了：

```bash
scripts/s1_quality_gate.sh data/videos/ --out outputs/qc.jsonl --viz outputs/ev/
```

不用 `pip install`。三个 venv 已经装好了要用的东西（`envs/requirements-*.txt` 记着精确版本），
而这是**共享机器，不要往里装包**。

```bash
envs/rt_env/bin/python -m unittest discover -s tests -v     # 秒级，10/10
```

## 两个踩过的坑，都已经用测试钉住

**1. venv 的解释器路径不能 `.resolve()`。** venv 的 `bin/python` 本身就是指向基础环境的
symlink，隔离靠的是 `pyvenv.cfg` 所在的目录。跟着 symlink 走会掉回基础环境，包完全是另一套 ——
实测 `envs/rt_env/bin/python` 有 `ultralytics`，resolve 成 `gs3dgs_env/bin/python3.10`
之后就没有。这种错以 `ModuleNotFoundError` 的形式出现，看起来像"环境装漏了"，很难查到真因。
`tests/test_paths.py` 里两个用例专门钉这件事，防止将来有人"顺手整理"再把它引回来。

**2. `src/` 里不许出现绝对路径字面量。** 重构前有 40 个 `.py` 散着 `/mnt/vlm/fanshaoheng`，
换机器或搬目录就全碎。`tests/test_no_hardcoded_paths.py` 0.1 秒跑完，当场报出来 ——
写成测试而不是写在文档里，因为"约定"会被下一次赶工时的一行硬编码悄悄破掉。

## 权重缺失 ≠ 判为不合格

`P.weights()` 查不到权重时返回 `None` 而不抛异常。调用方据此报 **unknown + 人看**，
不能报 reject。理由是实测的：body-pose 模型的手腕统计在"纯手部"这条边界上是**反向的**
（单手 0.25 > 双手 0.21，四个检测阈值都成立），用一个反向信号去猜，比老实承认测不出来更糟。

## 改了质检代码之后

判决必须与基准逐字一致：

```bash
PYTHONPATH=src envs/rt_env/bin/python -m web2robot.quality \
    data/videos/ tests/regression/*.mp4 --out /tmp/re/qc.jsonl --viz /tmp/re/ev
envs/rt_env/bin/python scripts/dev/diff_quality_run.py /tmp/re/qc.jsonl
```

比对**不要求数值逐位相同** —— KeypointRCNN 在 GPU 上不是逐位确定的，实测重跑一次
`cup_cpvH8gzUTko` 的 `torso_rate` 就从 0.4828 变 0.4655（n=58，差值正好 1/58，一帧翻转）。
判的是两件更贴近实质的事：判决字段逐字一致，以及每个参与判决的信号没有越过它的阈值。

还有一条规矩：**指标 ≠ 画面**。数值对了不等于数据可信，出片或看 contact sheet 确认，
别只看表格。视频统一 h264 / yuv420p，否则 VSCode 里放不出来。
