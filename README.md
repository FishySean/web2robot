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
| `src/web2robot/robots/` | 机器人定义（一台机器人一个子包，**不 import 任何重定向框架**）|
| `configs/paths.yaml` | **全工程唯一允许出现绝对路径的地方**。换机器只改这一个文件 |
| `scripts/` | 薄壳，只负责"用对的解释器 + 设好 PYTHONPATH"，不含逻辑 |
| `scripts/dev/` | 开发期工具（回归比对之类） |
| `external/` | 第三方仓库的 symlink（EgoInfinity / HaWoR），**里面不改代码** |
| `external/patches/` | 我们对上游的改动：逻辑归我方 + 一个 patch 记注入点 |
| `envs/` | 三个 venv 的 symlink + `requirements-*.txt` 依赖清单 |
| `assets/robots/m7/` | M7 的 MJCF / URDF / mesh（我们自己产出的资产，进 git） |
| `tests/` | stdlib unittest（秒级）+ `tests/regression/` 回归基准 |
| `data/` `outputs/` | 素材与产物，都不进 git。**产物只许落这里**，不许落 `external/` |

## 跑起来

这台机器上 **`conda activate` 不生效**，一律用绝对路径的解释器；薄壳已经替你处理好了：

```bash
scripts/s1_quality_gate.sh data/videos/ --out outputs/qc.jsonl --viz outputs/ev/

# ③感知前端产物 → EgoInfinity clip 目录（子命令一个前端一个，各自的 venv）
scripts/s3_to_clip.sh hawor external/HaWoR/example/ho3d_SMu41 --frames 55 \
    --out outputs/clips/ho3d_SMu41 --fps 15 --hands right

# ④重定向 + ⑤碰撞纠正（上游 EgoInfinity 主流程，碰撞/清洗逻辑走我方包）
scripts/s4_retarget.sh examples/fill_jar --robot m7 --out outputs/retarget/fill_jar \
    --ckpt runs/m7/taskspace_v2/checkpoints/final.pt --seed 0 --n_samples 5 \
    --arm_torso_collision --dual_hand_collision
```

不用 `pip install`。三个 venv 已经装好了要用的东西（`envs/requirements-*.txt` 记着精确版本），
而这是**共享机器，不要往里装包**。

```bash
envs/rt_env/bin/python -m unittest discover -s tests -v     # 秒级，60/60
```

## 重定向这一步的三个环境坑（薄壳已经替你处理，但要知道为什么）

1. **无头机器没有 X11**，默认 GLFW 后端直接 `could not initialize GLFW`：轨迹算完了却出不了片。
   `egl` 在这套 driver 上清理时抛 `EGLError`，**`osmesa`（CPU 软渲染）实测可用**。
2. **`--no-preview` 不加会丢掉整份日志。** 上游跑完拉一个交互式 GLFW 预览窗，无头机上它在
   C 层 abort，python 的 stdout 缓冲区来不及 flush —— 踩过：`robot_sim.mp4` 都写出来了，
   日志里只剩一行 GLFW 报错，`ArmTorsoFilter` 的统计全丢。配 `PYTHONUNBUFFERED=1` 双保险。
   （注意是 `--no-preview`，连字符，不是下划线。）
3. **不给 `--seed` 就没法和任何人对比结果。** 根锚点是从**随机先验**积 ODE 得来的
   （上游 `test.py` 第 241 区块自己写了这件事），所以锚点和 IK 可达性每次都不同。
   要做新旧代码对比，必须 `--seed` 固定 + `--n_samples` 相同。

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

## 改了碰撞检测 / 轨迹清洗之后

碰撞过滤是有限差分梯度下降、纯 CPU、没有随机源，**要求逐位相同**（比第①步那条
"判决一致"的判据严格）。三条线各测各的：

```bash
# 1. 隔离对比：把过滤器从整条链里拽出来，喂同一份输入跑旧/新两份实现
envs/rt_env/bin/python scripts/dev/diff_collision_migration.py

# 2. 端到端：同 seed 跑两遍，trajectory.npz 每个 key 逐位相同、robot_sim.mp4 md5 相同
# 3. 出四宫格看画面（源视频 / 不开碰撞 / 新代码 / 旧代码）
scripts/dev/render_quad.sh ...
```

端到端之所以不能单独作为判据：上游锚点的随机性会把迁移带来的差异**掩盖或伪造**成
几十度的关节角差 —— 第一次跑就被这个骗过，以为迁移改坏了 108°。

## 改了重定向兜底 / 锚点采样之后（`src/web2robot/retarget/`）

同样三条线，但这里多一个"参照物"的讲究：`tests/test_retarget_modules.py` 里
`_old_*` 那几个函数是**迁移前 `test.py` 内联版的逐字复制，故意没整理**。整理它就等于
把参照物改成了被测物，比对就不作数了 —— 文件头的注释写着"勿整理"，请当真。

```bash
# 1. 隔离对比（合成输入，秒级，不需要 external/）：40 个用例
envs/rt_env/bin/python -m unittest tests.test_retarget_modules -v

# 2. 隔离对比（真实片段，需要 external/）：12 个数组 + 叠字逐位比
envs/rt_env/bin/python scripts/dev/check_fallback_vs_baseline.py
#   期望 11 个片段全部"逐位一致 ✓"，ours_webapple 那段整段单手→两边都拒掉

# 3. 端到端 seed-0，再出四宫格看画面
scripts/dev/render_compare_grid.py --runs ... --out outputs/dev/compare_grid_retarget/
```

合成输入那份有一个用例专门断言"三档补洞 + 判坏"四种状态都被打到了
（`test_the_synthetic_input_actually_exercises_all_three_branches`）——
**参照物再准，输入没打到分支也证明不了什么**。

## 改了感知前端之后（`src/web2robot/perception/`）

分两层，因为变更理由不同：`to_clip.py` 是**下游的输入契约**（EgoInfinity clip 目录），
跟用哪个前端无关；`hawor.py` 这类是一个前端一个。前端的函数（`run_mano` /
`load_slam_cam`）是**参数注入**进来的，所以单测不需要 GPU、不需要 checkpoint、
不需要第三方仓库。

```bash
envs/rt_env/bin/python -m unittest tests.test_perception_modules -v   # 20 个用例
```

两处**错了不报错**的地方由测试钉住，都是实际会咬人的：

1. `world_to_camera` 的 einsum 下标顺序。转置反了相当于用逆旋转，手跑到相机后面、
   深度全负，但流水线照样跑到底出片。所以有一个用例和最笨的三重循环对齐，
   另一个用例断言"转置版结果确实不同"—— 否则第一个用例只是在测 einsum 会不会跑。
2. `hand_joints.bin` 的形状/dtype 与 `hand_meta.json` 的 `joints_shape` 一致。
   上游是 `np.fromfile` + reshape，不一致**不抛异常**，只会 reshape 出错位的轨迹。

还有一条约定：缺失关节填 **NaN 而不是 0** —— 0 是合法的相机系坐标，而
`trajectory/traj_cleanup.py` 正是靠 NaN 找空洞的。左手固定 slot 0、右手 slot 1，
不压缩空 slot：上游按 slot 取手，压缩会让片段中途换手，IK 照样解得出来，几乎看不出来。

## 改了 M7 的机器人定义（`src/web2robot/robots/m7/` 或 `assets/robots/m7/`）之后

两个验收脚本，输出要和上一次逐字节一致；`hand_frame` 那条约定尤其不能动：

```bash
scripts/dev/m7_tool.sh verify_m7_mjx_fk.py            # 期望 0.0000 mm / 0.0000 deg  MATCH ✓
scripts/dev/m7_tool.sh check_handframe_convention.py  # 期望 m7 左手 finger=+y thumb=-x palm=+z，右手镜像

# 再加一档：拿一段真实重定向轨迹逐帧验，而不是只验资产里写死的 home 姿态
scripts/dev/m7_tool.sh check_handframe_convention.py \
    --traj outputs/legacy_runs/runs/m7/validation/fill_jar
#   期望 "违反约定的帧: 0/216 ✓"，且两只手整段各只出现过一种轴向组合
#   （--traj 要给绝对路径或相对**上游 retarget/** 的路径，m7_tool.sh 会 cd 过去）
```

**`hand_frame` 的轴向是 finger+y / thumb−x / palm+z（左手），右手镜像。**
2026-07-24 吃过一次亏：检查脚本只验左手、还用了退化的 r2 body，错误地得出"两只手同
一套约定"，结果 M7 右手手掌/拇指被建成翻了 180°。现在脚本两只手都验、拿 g1/r2 当参照
断言镜像关系，`tests/test_m7_robot.py` 里还有一份秒级回归版。**永远不要只验一侧。**

还有一条只能靠跑才发现的：**动完 M7 资产的位置，删掉旧目录之后必须再跑一次端到端。**
上游有拼接出来的资产路径（`_ROBOTS_DIR / "m7" / "m7.xml"`），grep 找不到，而删除之前
跑的端到端会悄悄读旧文件、绿得很好看。详见 `external/patches/README.md`。

`m7_mjx.xml` 是生成物（`scripts/dev/generate_m7_mjx.py`），但**重跑生成器不会得到逐位
相同的文件** —— 原因和处置写在那个脚本的头部注释里，动它之前先看。

## 产物只许落 `outputs/`，不许落 `external/`

这条和"`src/` 里不许有绝对路径字面量"同级，理由同样是量出来的：

**上游 `test.py` 的 `--out` 默认值是 `<clip_parent>/<robot>/`** —— 把产物写在输入素材旁边。
再叠上薄壳**必须** `cd` 到上游 `retarget/`（它的 config / checkpoint 路径都是相对自己算的），
于是任何相对的 `--out` 也一起落进去。两件事一叠，实测结果是 `external/EgoInfinity/retarget/`
下攒了 408 MB、243 个 mp4/npz，而上游 git 只跟踪其中 1 个 —— 其余全是我们跑的；
同期 `outputs/` 里只有一个目录。

危害不只是乱：`external/` 是第三方 checkout，一次 `git clean -xdf` 或重新 clone
就把结果全带走 —— **这正是这次重构最初的动因**；而且产物和素材混在同一棵树里之后，
"哪份是官方素材、哪份是我们跑的"只能靠 mtime 猜。

所以判据写成了代码，不是文档：

```python
P.check_output_dir(path)     # 解析相对路径 + 拒绝 external/ 内的落点，违反就 SystemExit
```

四个写入口都过这道闸：`s4_retarget.sh`（cd **之前**把 `--out` 按调用方 cwd 转绝对路径，
没给就顶掉上游默认值，落 `outputs/retarget/<片段名>/`）、上游 `test.py`（兜底）、
`scripts/dev/_devcli.py`（默认落 `outputs/dev/<run 名>/`，7 个出片脚本共用）、以及
`scripts/dev/render_compare_grid.py`（自己一套 `--runs` / `--out`，因为它要同时读三台
机器人的 run）。`tests/test_outputs_not_in_external.py` 钉住结果：判据不是"数 mp4"，
而是**上游 git 认不认** —— 含 `robot_sim.mp4`/`trajectory.npz` 又不被上游跟踪的目录，
就是我们的产物躺在别人家里。

`external/` 下现在只剩输入：每个片段的 4 个输入文件（`depth.mp4` / `hand_joints.bin` /
`hand_meta.json` / `scene.json`）和 `runs/m7/taskspace*/`（训练 run + checkpoint，
上游 `train.py` 写在那里）。两者都已在 `configs/paths.yaml` 注册
（`roots.egoinfinity_clips` / `weights.m7_root_model`），代码不再用相对路径引它们。

2026-08-10 搬出来的 316 MiB 存量产物在 `outputs/legacy_runs/`，**保持原相对路径、
没有重命名或重组**，逐文件清单见 `outputs/legacy_runs/MANIFEST.tsv`。搬迁过程中撞到
两个 root 拥有的 run 目录（早先有人用 root 跑过），`mv` 会 copy 成功但删不掉源 ——
所以搬迁脚本改成"copy 完逐文件 md5 比对，比对通过才删源"，可重复跑。脚本本身留在
`scripts/dev/move_legacy_outputs.py`（`--dry` 只看清单），现在再跑是"待处理 0 项"。

## 一条贯穿全流程的规矩：指标 ≠ 画面

数值对了不等于数据可信。任何一步改完都要出片或看 contact sheet 用眼睛确认，别只看表格 ——
IK 成功率 100% 的片段照样可能手穿进躯干。视频统一 h264 / yuv420p，否则 VSCode 里放不出来。

