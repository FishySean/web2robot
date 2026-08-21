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

（暂时空着。B2「默认 `--root_solver` 选哪条」2026-08-21 拍了 `grid`，已落地 ——
上游 argparse 默认值 + patch 重导 + README ④ + `docs/VERIFICATION.md` 一节。）

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
| C9 | per-embodiment robot YAML | 抄 HandUMI 的写法 |
| C10 | VLM 语义一致性检查 | ①② 暂停期间一起搁着 |
| C11 | 视觉合成（新视角/渲染）那一摊 | 已归档，明确不占精力 |
| C12 | 前端控制台 | 见 [TODO22_FRONTEND_CONSOLE.md](TODO22_FRONTEND_CONSOLE.md) |
| C13 | github.io 页面 + demo 素材 | 目标已经改成 "repo + demo"，页面还没开工；`docs/assets/` 里的两张图是第一笔 |

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
