# 待办 / 被打断的活

**这份文档存在的理由**：新消息是**打断**，不是排队。Claude 没有跨会话自动维护的待办
列表，一件事做到一半被切走，除非当场写下来，否则就丢了 —— 而且几周后连"丢了什么"
都问不出来。所以规矩很简单：

> **被打断时，先往这里补一行，再去做新的那件。**

每条必须带**怎么续**（状态存在哪个目录 / 下一步是哪个文件哪个函数），不能只写标题。
只写标题的待办等于没写：过两周看到"碰撞校准 未完成"，还是得从头把上下文读回来。

做完的条目**删掉**，不要留一片 ~~划掉~~ ——真正做完的东西该在 README / 代码 /
`docs/VERIFICATION.md` 里有落点，这里只留"还欠着的"。

---

## A. 被打断，随时能续

### A1. grid 路线的碰撞过滤参数重校准（**验收过了一半**，2026-08-21）

- **要什么**：`--root_solver grid` 走的是网格搜索根位姿，目标函数不看身体，手臂贴身
  穿模比 neural 多（13 段实测 28.9% vs 23.8%，都是碰撞过滤**跑完之后**的残留）。
  只重新校准代理几何的膨胀/安全余量，**不改过滤器的核心逻辑**。
- **硬约束**：`neural` 那条路线的参数**一个都不许动**，两条路线的参数要能分开配置，
  不能绑成一套。做法是 `src/web2robot/collision/presets.py` 里 `neural` 留空 →
  今天的行为逐位不变。（已验：13 段的 neural 三列和旧表逐位相同 + 字节比对三个 SAME）
- **已做完**：`scripts/dev/sweep_arm_torso_params.py`（两阶段标定）→ 结论落
  `src/web2robot/collision/presets.py`（`neural` 空、`grid` = 盒 `[0.0695,0.119,0.239]`
  + `enter_thresh 0.02` + `margin 0.02`）→ `test.py` 接上 `--atf_preset/--atf_*` 覆盖 →
  `tests/test_module_boundaries.py::TestArmTorsoPresets` → 13 段 A/B 跑完出表
  （`outputs/dev/collcal_ab_table/`，逐段数字和抽帧都在 `docs/VERIFICATION.md`）。
- **验收结论（两条判据，一过一不过）**：
  - ✅ 穿模帧占比（真实网格判据）：13 段 **28.9% → 13.3%**，留出的 10 段 37.4% → 17.3%，
    12/13 段有残留 → 9/13，**没有一段变差**，ik 可行率一位没变。这条是**要什么**里
    真正在意的那条，泛化了。
  - ❌ 代理判据 vs 网格判据的帧数差：只在标定用的 3 段上收窄（226 → 24），留出的 10 段
    180 → **198**，而且方向翻面（误报 423 → 0，漏报 17 → **222**）。
- **为什么不过，以及下一步该动哪**（这是本条还留着的唯一原因）：不是参数没调好，是
  **代理形状到顶了** —— 躯干真身是圆的，轴对齐盒要把角上的误报压到 0 就得把 x 半长压到
  真身的 0.50 倍，于是平面方向欠覆盖，~1.7 cm 以内的真穿透对代理隐形。
  **下一步：把"检测"和"推出目标"解耦** —— 判是否触发用接近真身尺寸的盒
  （`presets.MESH_HALF`），推出目标仍用标定盒。落点是
  `src/web2robot/collision/arm_torso_filter.py`（`_sdf` 现在一个盒兼两职，
  给 `M7CapsuleModel` 加第二个 `torso_half` 或给过滤器加 `detect_half`），
  改完拿 `sweep_arm_torso_params.py phase2` 在同一批 prefilter 素材上重扫一遍，
  再跑一遍 `run_collcal_ab.sh` 看留出 10 段的漏报有没有下来。
  素材还在：`outputs/dev/collcal/prefilter/<短名>`（不带过滤的原始 q，换参数不必重跑 IK）。
- **另一件已经查清、别再当成同一个病的事**：残留里**深**的那些不是漏检 ——
  `--oo8_XIuOM_900.3_917.4` 最坏几帧代理读数 −1.92 ~ −4.70 cm，代理报了，是过滤器
  没修得动（那段 ik 只有 84.8%，源头坏帧）。抽帧确认它**肉眼可见地坏**（左小臂埋进躯干、
  指尖从胸口另一侧戳出来）；而 1.6 cm 那档看不出来。要治它得从坏帧兜底/源头感知那边走。


### A2. L3.4（rel3_4）接入（**第一阶段已完成**，2026-08-20；只剩第二阶段的欠账）

- **要什么**：第二台机器人和 M7 并列可切换（`--robot m7|l3_4`），**只做上肢**（双臂 7×2
  + 双手 12×2），腰/颈/腿锁死在 URDF 默认值（是锁死不是删除，以后要开随时开）。
  不动任何 M7 的现有文件，零上游 import，不反向依赖 `robots/m7/`。
- **已量到的关键事实**（决定了工作量，别再重新推一遍）：`assets/robots/urdf.tar.gz` 里的
  `l3.4.xml` 与我方 `m7.xml` **上肢完全同构** —— 43 个同名关节的 axis/range 全同
  （只有 `neck_pitch` 上限 0.54 vs 我方 0.48）、43 个同名 body 的 pos/quat 全同。
  手是**同一只 12 自由度手**（`thumb_bend + thumb_rota1/2`、其余四指 `bend/joint1/joint2`），
  不是 11 自由度的另一款；"xhand" 只是 URDF mesh 路径里的目录名。所以 MANO→手的映射
  表可以原样复用，不需要重新设计。L3.4 = M7 上身 + 12 个腿关节 + `base_link`。
- **mesh 那件事已经绕过（不是解决）**：包里一个 mesh 都没有。94 个零件和 M7 逐位相同
  （mass/inertia/COM 全同，是同一批零件）→ 建成**相对 symlink**（不是拷 19 MB，而且
  一个文件一个链接，以后拿到真的 L3.4 STL 换掉单个链接就行）；14 个腿部 + 盆骨
  `base_link` 没有正确的 mesh（M7 那个同名 `base_link.STL` 是升降柱底座，另一个零件），
  它们的 `visual`/`collision` 被删掉 → **渲出来腰以下是空的**。上肢的每个数字都不受影响
  （IK 链根和碰撞代理盒都挂 `waist_pitch_link`），但**出片之前得把腿的 mesh 要到**，见 D 节。
- **已做完**：`scripts/dev/build_l3_4_assets.py`（从原包生成整个 `assets/robots/l3_4/`，
  七步自检，含"对厂家 `l3.4.xml` 交叉校验"和"双臂链 vs `m7_mjx.xml` 逐位比对"两道）→
  `src/web2robot/robots/l3_4/` 五个模块（腰/颈/腿 17 个自由度锁在 `LOCKED_JOINTS`，
  `env._apply_locked()` 在每次 `mj_forward` 前按住）→ 上游三处注册（`sim/robots/__init__.py`、
  `RobotIKConfig.l3_4`、`_l3_4_12dof_from_keypoints`）+ `--robot` choices →
  `tests/test_l3_4_robot.py` 12 例、全量 301 全绿 → patch 重导 520 insertions（replay 6/6）
  → `PROJECT_LAYOUT.md` / `VERIFICATION.md` 已写。
- **状态存在哪**：资产 `assets/robots/l3_4/`（可 `--force` 重建）；端到端跑
  `outputs/retarget/l3_4_<片段>`，日志 `outputs/dev/l34_<片段>.log`。
- **第一阶段验收已过（2026-08-20 21:50）**：3 段官方片段端到端跑通（`fill_jar` /
  `serve_cake` / `sip_coffee`），IK 可行率 97.0% / 100% / 100%，`ArmTorsoFilter` 照旧开火
  （`fill_jar` 右臂 164/164 修净、`serve_cake` 左臂 187/188 剩 1、`sip_coffee` 右臂 2/2），
  三段的 `robot_sim.mp4` 逐帧看过，姿态靠谱（h264 版和抽帧在
  `outputs/dev/l3_4_stage1/`）。M7 逐字节不变已验：
  `scripts/dev/check_m7_unchanged_by_l3_4.sh` 三个产物全 `SAME`。
- **还欠着的（第二阶段，不阻塞主线）**：① 腿部 mesh（见 D 节）—— 补上之前 demo 不能用这台；
  ② `--root_solver neural` 那条路线在 L3.4 上没跑过（借的是 M7 的 ckpt，依据和失效条件见
  `VERIFICATION.md` 的 L3.4 一节；真要给 L3.4 单独训根模型时
  `robots/l3_4/sample_config.py` 已经备好但**没被跑过**）；
  ③ 解锁腰/腿要重训根模型（`LOCKED_JOINTS` 删一行就解锁一个自由度）。
  命令形式：`--robot l3_4 --root_solver grid --ckpt runs/m7/taskspace_v2/checkpoints/final.pt`
  （grid 压根不用模型，但上游 `test.py` 无条件 `_load_model`）。

## B. 等人拍板，我不该自己决定

（B2「默认 `--root_solver` 选哪条」2026-08-21 拍了 `grid`，已落地 —— 上游 argparse
默认值 + patch 重导 + README ④ + `docs/VERIFICATION.md` 一节。编号不复用。
注意这里的 B 编号和碰撞过滤那套 B0–B4 是两套东西，别串。）

下面九条是 **2026-08-21 方向调整**（从"打磨 demo"转向"批量产出 LeRobot v3.0 数据集"）
交下来的四个任务在实现过程中撞出来的矛盾。按用户自己的规矩「发现新依赖或矛盾先记录同步，
不要自己假设一个答案接着往下做」记在这里，**没有一条我自己拍了**。
（B9–B11 是 2026-08-22 做任务A 的差距分析时新撞出来的，明细在
[`LEROBOT_ALIGNMENT_GAP.md`](LEROBOT_ALIGNMENT_GAP.md)。）

| # | 撞到什么 | 为什么我不该自己定 | 我建议的答案 |
|---|---|---|---|
| B3 | **任务B（视觉合成）没有输入画面。** `data/clips_official/` 那 15 段官方片段里一帧 RGB 都没有：每段只有 `depth.mp4` `mask.mp4` `hand_joints.bin` `object_pose.bin` 这些，`camera.json` 只有内参（853×480）。`find` 过整个 `data/clips_official/` 加 upstream 的 examples，`*.mp4` 一共只有 15 个 `depth.mp4` + 10 个 `mask.mp4` | "把人的手臂抠掉、贴上渲染的机器人"这件事的输入是 RGB。没有 RGB 就不是"精度不够"，是这一步跑不起来。而我手上唯一有 RGB 的素材是 `data/videos/` 那 10 段抓取视频和自采的 `ours_*` —— 后者被"demo 只用官方片段"的规矩排除了 | 从 HF 那个官方片段库拉带 RGB 的原始片段（106 段那批，md5 可以和现有的对上）。要是 HF 上也没有 RGB，那就得张勃给公司内部 exo 视频批次的位置 |
| B4 | **视觉合成不是并行支线，它在格式对齐的关键路径上。** 参考数据集 `train/roll_the_towels/shard-0000-of-0001/meta/info.json` 里三路特征 `observation.images.cam_high` / `cam_left_wrist` / `cam_right_wrist` 都是 `dtype: video`、`[3,480,640]`、h264/yuv420p。我们现在一路画面都产不出 | 任务清单把"格式对齐"列为关键路径、"视觉合成"列为可并行。但对齐到那份 schema 就必须有三路视频，两件事其实是一件。这是排期假设错了，不是实现细节 | 第一版数据集先只交 `observation.state` / `action`（LeRobot 允许特征子集），把三路视频列成第二版；或者反过来把任务B提到和任务A同级。要张勃点头，因为这决定第一批数据能不能直接进训练 |
| B5 | **动作维度和帧率都对不上。** 参考数据集是 `robot_type: "yam_bimanual"`、`fps: 30`、`action`/`observation.state` 都是 `float32[14]`（`left_joint_1..6` + `left_gripper` ×2）。我们的 `trajectory.npz` 是 M7：`q_left/q_right` 各 (T,7) + `q_left_fingers/q_right_fingers` 各 (T,12) = **38 维**，`fps` 是 **15.401786**（浮点，不是整数） | 38→14 要么丢掉手指、要么换 robot_type 和特征命名；15.4→30 要重采样，那会动 `trajectory.npz` 的时间轴。两件都不是格式转换，是**改数据本身**，而且一旦发出去就是别人训练用的口径 | 新增一个 `robot_type: "m7_bimanual_dex"`，`action` 写 38 维、名字直接用 `trajectory.npz` 里的 `*_joint_names`（已经是全称），fps 字段写实测的 15.4 并在 `info.json` 里注明不重采样。要张勃确认公司训练侧能不能吃非 30 fps、非 14 维 |
| B6 | **任务C（MPC）缺前向模型，"误差降到阈值以内"会自指。** `refine/attach.py` 现在用刚连假设（物体跟手刚性连接）算物体位姿，`refine/modes.py::mpc_solve` 的 `NotImplementedError` 里写明了两个缺口：没有带物体的仿真 rollout、论文没给采样时域/样本数/代价权重 | 如果目标位姿和"仿真结果"都出自同一个刚连假设，那局部搜索一定能把误差压到 0，而画面里的物体不会有任何改变 —— 这种验收数字是假的。要么承认它是运动学层面的平滑（有用，但不能叫"物体位姿跟踪误差达标"），要么把物体网格搬进 MuJoCo（卡 C2：缺网格 + 缺米制深度） | 先做运动学 MPC，产物和文档里**明写"非物理，只是在参考轨迹附近做带约束的局部平滑"**，四宫格对比照做；物理 rollout 挂在 C2 后面。要张勃认这个降级，否则验收标准要改 |
| B7 | **"G1 已接入完成"和仓库现状不符。** upstream 有 `external/EgoInfinity/retarget/sim/robots/g1/`（config/env/sample_config）和官方权重 `/mnt/vlm/fanshaoheng/EgoInfinity/retarget/ckpts/g1.pt`，但我们仓库里没有 G1 的 MJCF、没有 hand_frame 约定、没有碰撞覆盖，`configs/robots/` 只有 `m7.yaml` 和 `l3_4.yaml` | 任务3要"批量转成 M7 和 G1 两种格式"。按 M7 的经验，接一台新机器人真正花时间的是 hand_frame 约定（M7 那次转错手掌）和自碰撞标定，不是跑通。说"已接入"可能指的是 upstream 那套官方 G1，两种理解的工作量差一个数量级 | 先只用 upstream 官方那套 G1 + 官方 ckpt 批量出数据（不做我们的碰撞过滤，产物里标明"未做自碰撞审计"）；要不要按 M7 的路子补一遍 hand_frame + 标定，另开一件事 |
| B8 | **批量的"现有 exo 视频"在哪。** 本地只有 `data/videos/` 10 段抓取的 exo（去重后 7 段）和 15 段官方 ego 片段，都是 demo 规模。HF 那 106 段是 ego，不是 exo | 任务3的输入规模决定要不要写并行调度、要不要断点续跑、要不要按 shard 切分 —— 这些是架构决定，规模差两个数量级就是两套写法 | 要张勃给出：这批 exo 视频在哪个路径/哪个内部库、大概多少段、有没有已经切好的片段边界 |
| B9 | **fps 不只是"不是 30"，是逐段都不一样。** 实测 10 段官方片段的 `scene.json.fps`：15.0000 / 15.0468 / 15.0778 / 15.1442 / 15.1927 / 15.4018 / 15.4762 / **18.4041** / 15.0000 / 15.0000。而 `info.json` 里 `fps` 是**整个数据集一个数**，`timestamp` 在参考里严格等距（实测间隔 0.03333334 = 1/30） | B5 只记了"15.4 不是 30"，当时以为是一个固定值。现在是三条路各有代价：① 全部重采样到统一 fps —— 改的是数据本身，插值会改关节角；② 按 fps 分 shard —— 10 段能分出 8 个 shard，等于没有数据集；③ 写一个名义 fps 并接受 `timestamp` 和真实时间偏差（18.4 那段 8 秒会偏 1.8 秒）。选哪条决定下游读到的时间轴是真的还是名义的 | 倾向 ③ 但把真实 fps 逐 episode 写进 episodes parquet 的自定义列（参考自己就加了 `dense_subtask_*` 这种非标准列），`info.json` 的 `fps` 写名义值并注明。要张勃确认训练侧读不读 `timestamp` |
| B10 | **没有任何 env 装了 `pyarrow`，写 parquet 缺依赖。** `rt_env` / `hawor_env` / `perception_env` 三个全试过都是 `ModuleNotFoundError`；系统 `/usr/bin/python3` 有 `pyarrow 24.0.0`（这次的分析就是拿它读的） | 规矩是"共享机器，不要 pip install"（`CONVENTIONS.md`）。而导出 LeRobot v3.0 必须写 parquet，绕不开。用系统 python 读分析没问题，但**导出流程不该依赖一个不在 `envs/requirements-*.txt` 里的解释器** | 往 `envs/requirements-rt.txt` 加 `pyarrow`，由人执行安装（我不动共享 env）；或者给导出单独建第四个 env。前者省事，但会改动一个 31 个测试都在用的环境，所以要人点头 |
| B11 | **我们产的 mp4 是 mpeg4，参考格式要求 h264，而且这违反我们自己的约定 §3。** 实测 `input_viz.mp4` / `robot_sim.mp4` 都是 `codec_name=mpeg4`；源头是上游 `retarget/utils/viz.py::write_video` 里的 `cv2.VideoWriter_fourcc(*"mp4v")`。`CONVENTIONS.md` 第 3 条写的是"视频一律 h264 / yuv420p" | 改它要动 `external/patches/egoinfinity-modified.patch`（唯一允许改上游的通道），而 `docs/VERIFICATION.md` 里有一条参照线正是 `robot_sim.mp4 = 205d96dba4a701e4be19a88ff1ec0483` —— 换编码器这个 md5 必然变，那条基线要重新立。这不是我能顺手改的 | 导出模块自己用 `-c:v libx264` 生成要发布的视频（不碰上游的调试用产物），上游那两个 mp4 留在 mpeg4 并在约定里注明例外；或者一次性换掉并重立基线。前者不动既有基线 |

## C. 有意推后的欠账

按"值不值得现在做"排的，不是按重要性。

| # | 事 | 怎么续 / 卡在哪 |
|---|---|---|
| C1 | `refine/` 真正的修复算法 | 现在只做到**诊断判断**；Replay 实现了，MPC / RL 是占位，调用直接 `NotImplementedError`（不静默降级是故意的） |
| C2 | `twin/` 的 SAM2 + FoundationPose 后端 | 只有 `official` 那条能跑；卡在缺物体网格 + 缺米制深度 |
| C3 | `hand_conf.bin (T,2)` 加进 clip 契约 | 是 Phantom 遮挡关节合并的前置条件 |
| C4 | Ego2Robot 0.65 臂展项的 ablation | 目标函数改动，之前明确划在校准任务范围外 |
| C5 | `make_keyframe_scorer` 的候选循环向量化 | 纯性能 |
| C6 | `scripts/dev/audit_retarget_feasibility.py` | 还没写 |
| C7 | episode 级判决聚合器 | 现在判据都是逐帧的 |
| C8 | 重投影–分割 IoU 自检 | Ego2Robot 质检里值得抄的一条 |
| ~~C9~~ | ~~per-embodiment robot YAML~~ | **2026-08-21 做完**（编号退役不复用）：`configs/robots/{m7,l3_4}.yaml` + `robots/params.py`，照 HandUMI 的格式（一机一 yaml + `verified` 标志位），代码侧唯一来源由 `tests/test_robot_params_yaml.py` 守。搬迁中发现的数值疑点见 C18–C20 |
| C10 | VLM 语义一致性检查 | ①② 暂停期间一起搁着 |
| C11 | 视觉合成（新视角/渲染）那一摊 | 已归档，明确不占精力 |
| C12 | 前端控制台 | 见 [TODO22_FRONTEND_CONSOLE.md](TODO22_FRONTEND_CONSOLE.md) |
| C13 | github.io 页面 + demo 素材 | 目标已经改成 "repo + demo"，页面还没开工；`docs/assets/` 里的两张图是第一笔 |
| C14 | 质检/路由接 [`VIDEO_SELECTION_GUIDE.md`](VIDEO_SELECTION_GUIDE.md) 的 §V1–§V4 | 判据文档 2026-08-21 已重写并搬进本仓库，**代码还是旧认知**。接的时候三件具体事：① `quality/` 现在没有"画面变化是否连续"这个准入判据（`camera_motion` 是路由标签，不是准入，别拿它顶替 §V1）；② `pipeline.py` 的 `trim` 只裁到 `usable_span` 最长一段，§V4 要的是**按切点拆成多段全部保留**、每段各自判；③ 每个判据函数的注释要写 `依据 VIDEO_SELECTION_GUIDE.md §Vx`（编号是接口）—— 这是这次文档任务定的验收标准，本次**只改了文档、没动代码**。卡在①②暂停自研等对接 wangjufei |
| C15 | §V5"机器人抽搐 ⇔ 切镜"的定量复现 | 现在是**有机理支撑的观察，本仓库没有数字** —— 我们端到端跑的官方片段本身不含切镜。做法：找一段有切镜的原始视频跑完整条链，看 `root_frames.npz` 的位姿和 `trajectory.npz` 的关节角在切镜帧上的**一阶差分尖峰**位置和 ffmpeg 报的切点对不对得上。做完把数字写进 §V5，把"未定量复现"那句删掉 |
| C16 | 手部目标 lift 到世界系 + 在世界系里搜根位姿（解开 §V3 的朝向禁令） | 现在整条链的手部 IK 目标是**相机系**的（`utils/pose_utils.py::cam_to_root_targets` 算 `p_root = R_rootᵀ(p_hand_cam − t_root)`），而 grid 路线的躯干位姿是 `np.broadcast_to(_sol.R, (T,3,3))` —— **相机系里的一个常量**。后果：相机一转/一走，假的手部位移 1:1 注入，所以 §V3 只能写成无条件禁止转身转头。要真正支持"人转头/走位"的素材，得 ① 把手腕轨迹用相机位姿 lift 到世界系（HaWoR 那条路线本来就有世界系输出，只是 clip 契约没往下传，喂的是 `left_cam_np`）；② 网格搜索改在世界系里做，或者让根位姿逐帧跟随相机而不是被 `--torso_alpha` 往锚点压。**大改，动的是 upstream 接口，不在当前范围。** 做之前先做 C17 确认收益值不值 |
| C17 | 坐实 §V2 那四个数（1°→0.9 cm / 5°→4.4 cm / 30°→26 cm / 一步→60 cm） | 现在是拿 `cam_to_root_targets` 的公式和默认值（`--tol_pos 0.01`、`margin 0.02`、M7 实测 `r_max 1.007`）推出来的**几何推论**，不是端到端实测。做法：拿一段官方片段，人工往相机位姿上注入已知的旋转 θ / 平移 d，量重定向输出的手部末端位置偏移是不是跟着 `d·θ` 走，顺便看 ik_rate 和残余穿透从哪个角度开始塌。做完把 §V2 那句"几何推论，不是实测"换成实测数 |
| C18 | `verified: false` 那些数字里，真正"从没量过"的三处 | YAML 搬迁（C9）时逐个看过来的，**只记录、一个数都没改** —— 参数改动是单独一件要决策的事，改完还得重跑 `check_neural_bytes.sh`。① `collision.proxy.torso_half=[0.105, 0.135, 0.215]` 和 `tip_radius=0.012`：代理盒比躯干网格 AABB `[0.139, 0.170, 0.239]`（这个是量的，`verified: true`）三轴各收了 3.4/3.5/2.4 cm，**为什么收这么多没有依据**，是当初手挑的；② `ik.start_config` 的肩外展 ±0.20 rad 从没和别的静息姿态比过 ik_rate，就是个看着顺眼的种子；③ `collision.arm_torso.defaults` 那 11 个值里只有 grid 路线覆盖的 3 个（`torso_half`/`enter_thresh`/`margin`）被 sweep 标定过，剩下 8 个（`w_pen`/`w_ee`/`w_prox`/`fd_eps`/…）是默认值。要动的话：先扫一遍，再改 yaml，再重跑字节验证 |
| C19 | 新增两层坏帧粒度的三个阈值是**惯例，不是实测** | `trajectory/tiers.py` 里 `z_thresh=3.5`（Iglewicz–Hoaglin 论文的建议值）、`frac_thresh=0.05`（"5% 帧离群才算整段有问题"）、`seg_sec=2.0`（轨迹段长度）—— 都是拿约定值起的头，没在我们的素材上扫过。做法：拿 HF 那 106 段官方片段跑一遍，人工标"这段镜头是不是真的乱"，看这三个数在什么组合下和人工判断吻合。注意判据是**只警告/只标记**，所以误报的代价比漏报低，别照抄论文的剔除口径来定阈值 |
| C20 | episode 级只能做 clip **内部**的离群，跨语料的做不了 | EgoSmith 原文（arXiv 2607.09701 §3）是在整个语料上算相机平移分布再丢离群 episode；我们的 pipeline 一次只见一个 clip，所以 `episode_camera_check` 判的是"这段片子内部有没有几对帧的机位运动格外大"。真正的跨语料离群该在质检阶段做（C14 那一摊，`quality/` 已经有 `_camera_motion_score_flow` 的分数，缺的是把整批分数存下来再回头比）。同一条：原文那个"硬旋转阈值丢掉头部大幅转动的 episode"我们**没有对应物** —— clip 契约里没有逐帧相机位姿（`camera.json` 只有内参 + 重力），光流也分不开平移和旋转，所以警告文案只能把 §V2/§V3 一起引 |
| C21 | L3.4 一个碰撞参数都没标定过 | `configs/robots/l3_4.yaml` **刻意没有 `collision:` 一节**（`tests/test_robot_params_yaml.py::TestL34HasNoCollisionSection` 把这条"故意不写"钉住了，免得有人把一份 `verified: false` 的复制品读成"L3.4 也支持"）。现在那套过滤器是 M7 专用的：代理盒挂 `waist_pitch_link`、body 名写死 `left/right_hand_frame`。等真要支持 L3.4，加 yaml 那一节的同时必须连标定一起加 |
| C22 | `--quality_gate external` / `--routing external` 第三档 | 2026-08-21 明确**先不加**：现在没有对接对象，不知道公司那套质检输出什么格式的判决，先留一个名字会有人去实现它。开关的取值集合只写在 `src/web2robot/quality/config.py` 的 `GATE_MODES` / `ROUTING_MODES` 两个常量里，加档就改那一处（argparse 的 choices 和单测都引用它，`tests/test_quality_switch.py::test_no_external_mode_yet` 把"现在只有两档"钉住了，加档时会红，那是提醒不是故障）。接的时候要想清楚的是：`external` 读进来的判决要映射到 `Verdict` 的哪一档，以及它给不给 `suggested_route` |

## D. 不是技术活，但会忘

- 把穿透 / ρ̄ 那个发现同步给白琦呈；找曹源江确认 Qwen-RobotManip 公式 (3) 的读法。
- 问魏庆功要内部 UMI 数据。
- **要 L3.4 的腿部 mesh**（14 个：`{left,right}_{hip_roll,hip_yaw,hip_pitch,knee,ankle_pitch,
  ankle_roll,foot_ee}_link.STL`）和盆骨 `base_link` 的真 STL —— 厂家那个 50 KB 的包里
  一个 mesh 都没有。现在渲出来腰以下是空的，**上肢数字不受影响，但 demo 出片之前必须补**。
  拿到之后：丢进 `assets/robots/l3_4/meshes/`（覆盖同名 symlink 即可），把
  `scripts/dev/build_l3_4_assets.py` 里 `NO_MESH_LINKS` 对应的行删掉，`--force` 重建。
- **再要一份 L3.4 的厂家原包 `urdf.tar.gz`** —— 生成资产那次用的原包现在**不在磁盘上了**
  （`assets/robots/urdf.tar.gz` 不存在，全盘也没有），所以
  `build_l3_4_assets.py` 现在跑不了（`P.asset("l3_4_src_tar")` 就会炸）。
  **已生成的资产和所有跑出来的数字都不受影响**，缺的只是"能重跑生成脚本"这件事。
  包里三个文件，两个已经留档在版本库里：`l3.4.xml` → `assets/robots/l3_4/l3_4_vendor.xml`、
  `l3_4.urdf.xacro` → 同目录同名，都是 `shutil.copy2` 的逐字节副本；
  **只缺 `l3_4.urdf` 的原件**（版本库里那份 `l3_4_from_urdf.urdf` 是加了两处改动之后的，
  header 里写了改了什么，理论上能反推回去，但不如直接再要一份）。
  拿到之后：丢到 `assets/robots/urdf.tar.gz`，`--force` 重建，产物应当和现在的逐字节相同
  （脚本七步自检会自己核对），顺手把它提交进去，下次就不会再丢。
- 请人 `chown fanshaoheng` memory 目录里那 6 个 root 所有的文件（现在改不动）。

---

## 更早被打断的活，怎么捞

这份清单是 2026-08-20 从 memory 和当时的上下文里重建的，**只覆盖当时还记得的**。
更早的打断（比如 7 月那些）没有记录，但会话记录还在：

    ls /mnt/vlm/fanshaoheng/.claude/projects/-mnt-vlm-fanshaoheng/*.jsonl

要挖的话，找用户消息紧跟在我一串工具调用之后、且话题突然换掉的位置 —— 那就是打断点。
成本不低，除非真的怀疑漏了要紧的东西，否则不值得挖。
