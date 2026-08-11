# 踩过的坑

每条的格式都是**现象 → 真因 → 现在怎么防**。收在这里是因为它们有一个共同点：
**报错信息指向错的方向**，不知道真因的话能查很久。

必须遵守的约定在 [`CONVENTIONS.md`](CONVENTIONS.md)，改完怎么验收在
[`VERIFICATION.md`](VERIFICATION.md)。

---

## 环境类

### 1. 无头机器出不了片：GLFW 起不来，`egl` 也不行，要用 `osmesa`

**现象**：轨迹算完了，`could not initialize GLFW`，没有视频。
**真因**：这台机器没有 X11。`egl` 在这套 driver 上清理时抛 `EGLError`。
**处置**：`MUJOCO_GL=osmesa`（CPU 软渲染），实测可用。`scripts/s4_retarget.sh` 已设好。

### 2. 不加 `--no-preview` 会**丢掉整份日志**

**现象**：`robot_sim.mp4` 都写出来了，日志里只剩一行 GLFW 报错，
`ArmTorsoFilter` 的统计全没了 —— 看起来像"过滤器没生效"。
**真因**：上游跑完会拉一个交互式 GLFW 预览窗，无头机上它在 C 层 `abort`，
Python 的 stdout 缓冲区来不及 flush。
**处置**：`--no-preview`（连字符，不是下划线）＋ `PYTHONUNBUFFERED=1` 双保险。

### 3. 不给 `--seed` 就没法和任何人对比结果

**现象**：同样的代码跑两遍，关节角差几十度，以为迁移改坏了（真的被骗过一次，以为差 108°）。
**真因**：根锚点是 flow-matching 生成模型从**随机先验**积 ODE 出来的，每次都不同。
**处置**：任何新旧对比都必须 `--seed` 固定 **＋** `--n_samples` 相同。

### 4. venv 的解释器路径不能 `.resolve()`

**现象**：`ModuleNotFoundError`，看起来像"环境装漏了包"。
**真因**：venv 的 `bin/python` 本身就是指向基础环境的 symlink，隔离靠的是 `pyvenv.cfg`
所在的目录。跟着 symlink 走会掉回基础环境，包完全是另一套 —— 实测
`envs/rt_env/bin/python` 有 `ultralytics`，resolve 成 `gs3dgs_env/bin/python3.10` 之后就没有。
**处置**：`web2robot.paths` 里不 resolve；`tests/test_paths.py` 两个用例专门钉这件事，
防止将来有人"顺手整理"再把它引回来。

### 5. `HF_HOME` 必须覆盖，否则 MoGe 权重报"没下载"

**现象**：`PermissionError: .../models--Ruicheng--moge-2-vitl-normal`，看着像权重没下载。
**真因**：这台机器的 shell 把 `HF_HOME` 指向共享的 `/mnt/vlm/common/cache`（我们没写权限），
hf_hub 想在只读目录里建缓存。权重其实在 `$HOME/.cache/huggingface`。
**处置**：`scripts/s3_to_clip.sh` 已经替你设了（可用 `WEB2ROBOT_HF_HOME` 覆盖）。

---

## 实现类

### 6. 两条深度取样路径的取整方式**故意不同**，别顺手统一

`sample_pointmap` 用 `round`，`sample_depth` 用 `int()` 截断 —— 差半个像素，是从两个
原脚本照抄的。**"顺手统一"会改掉 `evidence/` 里 11.0 cm 那个数，而代码照样跑。**

还有一处更要紧的差异：前者不 clip 中心像素，完全出画的关键点得 NaN；后者会 clip 到画内，
出画时返回一个**编出来的**边缘深度。两条都照原样留着，各自钉了测试。

### 7. 反投影必须用取整后的像素

深度是在整数像素处取的，用亚像素坐标反投影得到的 XY 和深度不对应。误差只有半像素量级，
**肉眼和单测都容易放过去**。所以取整这一步拎成了 `moge.pixel_index()`，免得调用方各写一遍。

### 8. `world_to_camera` 的 einsum 下标顺序

转置反了相当于用逆旋转：手跑到相机后面、深度全负，**但流水线照样跑到底出片**。
所以有一个用例和最笨的三重循环对齐，另一个用例断言"转置版结果确实不同" ——
否则第一个用例只是在测 einsum 会不会跑。

### 9. `hand_joints.bin` 的形状必须和 `hand_meta.json` 的 `joints_shape` 一致

上游是 `np.fromfile` + reshape，不一致**不抛异常**，只会 reshape 出错位的轨迹。

### 10. 权重缺失 ≠ 判为不合格

`P.weights()` 查不到权重时返回 `None` 而不抛异常，调用方据此报 **unknown + 人看**，
不能报 reject。理由是实测的：body-pose 模型的手腕统计在"纯手部"这条边界上是**反向的**
（单手 0.25 > 双手 0.21，四个检测阈值都成立）。用一个反向信号去猜，比老实承认测不出来更糟。

### 11. 拼接出来的路径会躲过 grep，所以"删掉旧文件后必须再跑一次端到端"

**现象**：迁移 M7 资产时 `grep -rn "robots/m7"` 找不到任何引用，删掉旧目录后端到端
`FileNotFoundError`。而**删除之前**跑的那次端到端是绿的。
**真因**：上游那行路径是拼出来的（`_ROBOTS_DIR / "m7" / "m7.xml"`），源码里压根没有
`robots/m7` 这个字样；删除前旧文件还在，它悄悄读的是旧文件。
**处置**：`tests/test_m7_robot.py::TestUpstreamAssetPaths` 不 grep，而是**把 IK config
造出来看 `mjcf_path` 存不存在**，拼接的路径也躲不掉。流程上：删除后必须再跑一次端到端。

### 12. `m7_mjx.xml` 是生成物，但重跑生成器**不会**得到逐位相同的文件

重跑会和已提交的版本差一行（`right_hand_frame` 的 quat 精度）：已提交的是
`fix_m7_handframe.py` 打印的 6 位小数手工贴回 MJCF 的，模长 1.00000023 ≠ 1；
生成器读的是**编译后**的 `body_quat`，MuJoCo 编译时归一化过，所以是全精度。
两者是同一个旋转（旋转角差 2.4e-6 度）。已查清、无害，详见
`scripts/dev/generate_m7_mjx.py` 的头部注释。

### 13. `.gitignore` 里 `data/*` 会让后面的 `!` 例外失效

`data/*` 把子目录本身也排除了，git 就不再下去看，`!data/**/README.md` 根本轮不到生效。
要写 `data/**` + `!data/**/`。

---

## 机器学习/数据类

### 14. `sample_config.py` 的比例参数不能照抄别的机器人

M7 的采样配置抄自 robonaut2，但其中两个参数（肘 jitter、`ou_step`）必须按 M7 的实际比例
重算：R2 的 `(-1.6,-0.3)` 会把 M7 的肘压到 `[-2.36,-1.30]`（死弯、永不伸展），
躯干→手腕距离只有臂展的 50%；重算后 60%，重训后手臂折叠和穿模都改善。

### 15. 多台机器人在**同一帧**同时崩坏 → 查共享输入，不是查臂展

崩坏看起来像"这台机器人臂展不够"，但如果几台机器人在同一帧一起崩，真因几乎总在共享的
源感知数据里（手腕深度爆点、四元数跳变）。先看 `trajectory/traj_cleanup.py` 的坏帧统计。

### 16. `hand_frame` 那次翻 180° —— **永远不要只验一侧**

2026-07-24 之前的检查脚本只验左手、还用了退化的 r2 body，错误地得出"两只手同一套约定"，
结果 M7 右手手掌/拇指被建成翻了 180°。现在脚本两只手都验，并拿 g1/r2 当参照断言镜像关系。
轴向约定是 **finger+y / thumb−x / palm+z（左手），右手镜像**。

### 17. IK 后的关节平滑在我们的默认参数下是空操作

上游 `test.py` 的窗口是 `_w = max(1, int(smooth_sigma * seq_fps))`，默认
`smooth_sigma=0.1`；我们的片段都是 `--fps 15`，于是 `int(0.1*15) = 1`，
`uniform_filter1d(size=1)` 是恒等变换 —— **那三轮平滑从来没改过任何东西**。

不是 bug，是参数没对上（`fps ≥ 20` 时它才真的开始平滑）。留着没动，但别指望
"已经平滑过了"；真要平滑得先把 `--smooth_sigma` 调大。
