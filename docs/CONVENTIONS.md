# 工程规矩（不变量）

这里放的是**必须遵守的约定**，不是建议。每一条后面都标着由哪个测试钉住 ——
写成测试而不是写在文档里，因为"约定"会被下一次赶工时的一行代码悄悄破掉。

现象层面的陷阱（"为什么会报这个错"）在 [`PITFALLS.md`](PITFALLS.md)；
改完代码怎么验收在 [`VERIFICATION.md`](VERIFICATION.md)。

| # | 规矩 | 谁钉着 |
|---|---|---|
| 1 | 产物只许落 `outputs/`，不许落 `external/` | `tests/test_outputs_not_in_external.py` |
| 2 | `src/` 里不许出现绝对路径字面量 | `tests/test_no_hardcoded_paths.py` |
| 3 | 视频一律 h264 / yuv420p | 人工（VSCode 放不出来就是错的） |
| 4 | 缺失关节填 NaN，不填 0 | `tests/test_perception_modules.py` |
| 5 | `robots/` 下不许 import 重定向框架 | `tests/test_module_boundaries.py` |
| 6 | `external/` 里不改代码 | `external/patches/` 的单一 patch |
| 7 | 共享机器不 `pip install` | 人工（`envs/requirements-*.txt` 记着精确版本） |
| 8 | 指标 ≠ 画面：改完必须出片看 | 人工，但这是最重要的一条 |
| 9 | 结构文档不许和现实脱钩 | `tests/test_docs_layout.py` |

---

## 1. 产物只许落 `outputs/`，不许落 `external/`

理由是量出来的，不是洁癖：

**上游 `test.py` 的 `--out` 默认值是 `<clip_parent>/<robot>/`** —— 把产物写在输入素材
旁边。再叠上薄壳**必须** `cd` 到上游 `retarget/`（它的 config / checkpoint 路径都是相对
自己算的），于是任何相对的 `--out` 也一起落进去。两件事一叠，实测结果是
`external/EgoInfinity/retarget/` 下攒了 **408 MB、243 个 mp4/npz**，而上游 git 只跟踪
其中 1 个 —— 其余全是我们跑的；同期 `outputs/` 里只有一个目录。

危害不只是乱：`external/` 是第三方 checkout，一次 `git clean -xdf` 或重新 clone 就把
结果全带走 —— **这正是整次重构最初的动因**。而且产物和素材混在同一棵树里之后，
"哪份是官方素材、哪份是我们跑的"只能靠 mtime 猜。

所以判据写成了代码：

```python
P.check_output_dir(path)     # 解析相对路径 + 拒绝 external/ 内的落点，违反就 SystemExit
```

四个写入口都过这道闸：`s4_retarget.sh`（cd **之前**把 `--out` 按调用方 cwd 转绝对路径，
没给就顶掉上游默认值）、上游 `test.py`（兜底）、`scripts/dev/_devcli.py`（默认落
`outputs/dev/<run 名>/`，7 个出片脚本共用）、`scripts/dev/render_compare_grid.py`。

测试的判据不是"数 mp4"，而是**上游 git 认不认**：含 `robot_sim.mp4`/`trajectory.npz`
又不被上游跟踪的目录，就是我们的产物躺在别人家里。

`external/` 下现在只剩输入：每个片段的 4 个输入文件和 `runs/m7/taskspace*/`（训练 run
＋ checkpoint）。两者都已在 `configs/paths.yaml` 注册，代码不再用相对路径引它们。
2026-08-10 搬出来的 316 MiB 存量在 `outputs/legacy_runs/`（保持原相对路径，清单见
其中的 `MANIFEST.tsv`）。

## 2. `src/` 里不许出现绝对路径字面量

重构前有 40 个 `.py` 散着 `/mnt/vlm/fanshaoheng`，换机器或搬目录就全碎。唯一的来源是
`configs/paths.yaml`，代码一律走 `from web2robot.paths import P`。测试 0.1 秒跑完，
当场报出行号。

薄壳（`.sh`）同理：凡是要指仓库内的东西，必须用 `BASH_SOURCE` 相对自己推出仓库根。

## 3. 视频一律 h264 / yuv420p

`mpeg4` 编出来的片在 VSCode 里放不出来（踩过）。出片脚本统一 `-c:v libx264 -pix_fmt
yuv420p`，文件名带 `_h264` 的是转码产物。

## 4. 缺失关节填 NaN，不填 0

0 是**合法的**相机系坐标，而 `trajectory/traj_cleanup.py` 正是靠 NaN 找空洞的。填 0
会让坏帧检测看不见坏帧，IK 照样解得出来，画面上是手突然跳到相机原点。

配套约定：左手固定 slot 0、右手 slot 1，**不压缩空 slot**。上游按 slot 取手，压缩会让
片段中途换手 —— IK 依然收敛，几乎看不出来。

## 5. `robots/` 下不许 import 重定向框架

`robots/m7/` 回答的是"这台机器人长什么样"，跟用哪个框架无关，所以也不该知道框架的存在。
**加一台新机器人 = 在 `robots/` 下新建一个子包，别的模块一行不用改。**

为此这一层只出数据和自己的类，不出上游类型：`CONFIG`/`ENV_SPEC` 是纯 dict，
`M7Env` 实现上游 `BaseEnv` 的接口但**不继承**它（`BaseEnv` 是纯抽象类，全仓库无
`isinstance` 检查；继承换成了一致性断言，报错比继承更早）。

## 6. `external/` 里不改代码

`external/EgoInfinity` 和 `external/HaWoR` 是第三方仓库的 symlink。我们对上游的改动
只以两种形式存在：

1. **逻辑归我方** —— 实质逻辑作为一等公民放在 `src/web2robot/` 下，不寄生在上游目录里。
2. **一个 patch 文件** —— 真正必须改的注入点记在 `external/patches/egoinfinity-modified.patch`。

**判断这条做对了没有，看 patch 的行数**：迁移做对的话逻辑进 `src/`、上游只剩接线，
patch 就该变小（313 → 233 insertions）。哪天它开始变大，就是有人在往上游文件里写实质
逻辑了。改动的完整记录在 [`external/patches/README.md`](../external/patches/README.md)。

## 7. 共享机器不 `pip install`

这台机器上大约 100/128 核常年有人在用，三个 venv 是共用的。要装东西用
`pip install --target` 或新建 venv，别动共享环境。长任务要 `nice` + 后台跑。

`conda activate` 在这台机器上**不生效**，一律用绝对路径的解释器
（`envs/rt_env/bin/python`）；薄壳已经替你处理好了。

## 8. 指标 ≠ 画面

数值对了不等于数据可信。**任何一步改完都要出片或看 contact sheet 用眼睛确认。**

这条是吃过亏来的：IK 成功率 100% 的片段照样可能手穿进躯干；两条深度策略的"开合"
数字差 10 倍，光看数解释不了谁对，是渲出来 + 用骨长这个"该是常数的量"才定位到病因的。

## 9. 结构文档不许和现实脱钩

新建顶层目录、新增一类产物、搬走一份重要材料，都要同步更新
[`PROJECT_LAYOUT.md`](PROJECT_LAYOUT.md)。烂掉的结构文档比没有更糟 —— 读的人会信。
测试会在你新建目录却没写说明时变红。
