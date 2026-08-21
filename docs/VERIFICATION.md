# 改完之后怎么验收

这个工程的验收判据**一个模块一套**，因为各模块的确定性不同：碰撞过滤是纯 CPU
无随机源，可以要求逐位相同；质检里跑着 GPU 神经网络，逐位相同是不可能的目标。
拿错判据的后果是双向的 —— 要么放过真错误，要么把浮点噪声当成 bug 追半天。

约定见 [`CONVENTIONS.md`](CONVENTIONS.md)，坑见 [`PITFALLS.md`](PITFALLS.md)。

**所有模块共通的最后一步：出片，用眼睛看。** 指标 ≠ 画面。

```bash
envs/rt_env/bin/python -m unittest discover -s tests -v     # 秒级，全套 175 个用例
```

---

## 判据一览

| 改了什么 | 判据 | 为什么是这个判据 |
|---|---|---|
| ①质检 / ②路由 | **判决字段逐字一致** + 每个信号没越阈 | GPU 上的 KeypointRCNN 不是逐位确定的 |
| ③感知前端 | 单测（注入假 callable）+ 冻结值比对 | 前端要 GPU，但算术部分可以纯 numpy 测 |
| ④重定向 | 隔离对比逐位一致 + 固定 seed 的端到端 | 根锚点有随机源，不固定 seed 无法比 |
| ⑤碰撞 / 轨迹 | **逐位相同** | 纯 CPU 有限差分，没有随机源 |
| M7 机器人定义 | 两个验收脚本输出逐字节一致 | 资产是静态的，任何变化都该是有意的 |
| 新加一台机器人 | 生成脚本七步自检 + **表 vs 自己的 MJCF** + 老机器人逐字节不变 | 两台同构机器人之间不许互当真源，见下 |
| `evidence/` 里的数 | 断言**具体数值**，不是大于小于 | 防的是论文数字和证据脱钩 |

---

## ①质检 / ②路由（`src/web2robot/quality/`、`routing/`）

判决必须与基准逐字一致：

```bash
PYTHONPATH=src envs/rt_env/bin/python -m web2robot.quality \
    data/videos/ tests/regression/*.mp4 --out /tmp/re/qc.jsonl --viz /tmp/re/ev
envs/rt_env/bin/python scripts/dev/diff_quality_run.py /tmp/re/qc.jsonl
```

**不要求数值逐位相同** —— KeypointRCNN 在 GPU 上不是逐位确定的，实测重跑一次
`cup_cpvH8gzUTko` 的 `torso_rate` 就从 0.4828 变 0.4655（n=58，差值正好 1/58，
一帧翻转）。判的是两件更贴近实质的事：**判决字段逐字一致**，以及**每个参与判决的
信号都没有越过它的阈值**（后者能抓到"判决碰巧没变但信号已经贴着阈值了"）。

再看一眼 contact sheet（`--viz`）确认画面。

## ③感知前端（`src/web2robot/perception/`）

分两层，因为变更理由不同：`to_clip.py` 是**下游的输入契约**，跟用哪个前端无关；
`hawor.py` / `wilor.py` + `moge.py` 一个前端一个。前端的函数（`run_mano` /
`load_slam_cam` / WiLoR 的 `predict` / MoGe 的 `infer`）是**参数注入**进来的，
所以单测不需要 GPU、不需要 checkpoint、不需要第三方仓库。

```bash
envs/rt_env/bin/python -m unittest tests.test_perception_modules -v   # HaWoR，20 个
envs/rt_env/bin/python -m unittest tests.test_wilor_modules -v        # WiLoR+MoGe，40 个
```

改了算术部分，还要和冻结基线比：

```bash
envs/rt_env/bin/python scripts/dev/check_wilor_vs_baseline.py
#   期望：与 2026-07-14 冻结的 wilor_wrist 最大差 < 6e-8（float32 的量化步）
```

三个"错了不报错"的地方由测试钉住：einsum 下标顺序、`hand_joints.bin` 与
`joints_shape` 一致、两条取样路径故意不同的取整方式。细节见
[`PITFALLS.md`](PITFALLS.md) 第 6~9 条。

想看两条深度策略的差别：

```bash
envs/perception_env/bin/python scripts/dev/viz_wilor_depth_modes.py \
  --clips outputs/clips/<pointmap> outputs/clips/<globalscale> \
  --rgb <图片目录> --out outputs/viz/wilor_depth_modes.mp4
```

## ④重定向（`src/web2robot/retarget/`）

三条线，另外这里多一个"参照物"的讲究：`tests/test_retarget_modules.py` 里
`_old_*` 那几个函数是**迁移前 `test.py` 内联版的逐字复制，故意没整理**。整理它就等于
把参照物改成了被测物，比对就不作数了 —— 文件头的注释写着"勿整理"，请当真。

```bash
# 1. 隔离对比（合成输入，秒级，不需要 external/）：40 个用例
envs/rt_env/bin/python -m unittest tests.test_retarget_modules -v

# 2. 隔离对比（真实片段，需要 external/）：12 个数组 + 叠字逐位比
envs/rt_env/bin/python scripts/dev/check_fallback_vs_baseline.py
#   期望 11 个片段全部"逐位一致 ✓"，ours_webapple 那段整段单手→两边都拒掉

# 3. 端到端 seed-0，再出四宫格看画面
envs/rt_env/bin/python scripts/dev/render_compare_grid.py --runs ... \
    --out outputs/dev/compare_grid_retarget/
```

合成输入那份有一个用例专门断言"三档补洞 + 判坏"四种状态**都被打到了**
（`test_the_synthetic_input_actually_exercises_all_three_branches`）——
参照物再准，输入没打到分支也证明不了什么。

### ④里的第二条根位姿路线（`retarget/root_grid.py`，静态网格搜索）

这条**没有参照物**可比 —— 它不是迁移，是新方法（Qwen-RobotManip 公式 3），
上游没有对应实现。所以判据换成三层：

```bash
# 1. 隔离单测（纯 numpy 假 IK，秒级，不要 GPU/checkpoint）：44 个用例
envs/rt_env/bin/python -m unittest tests.test_root_grid -v

# 2. 端到端切换（同一条流水线，只换 --root_solver）
scripts/s4_retarget.sh examples/fill_jar --robot m7 --seed 0 \
    --ckpt runs/m7/taskspace_v2/checkpoints/final.pt --root_solver grid \
    --out outputs/retarget/rootcmp/fill_jar_grid

# 3. 两条路线在 11 个片段上的对比表（约 15 分钟/片段，要后台跑）
scripts/dev/m7_tool.sh compare_root_pose_solvers.py \
    --spacing 0.05 --rotation both --yaws 12 --device cuda:3 --seed 0
```

单测里必须钉住的三条（都各有一个用例，删了就等于把这个模块的安全网拆了）：

1. **剪枝可采纳** —— 剪枝版和 `exhaustive=True` 版**逐位相同**。这是模块里唯一
   一处"为了快而改变搜索顺序"的地方，破掉之后跑出来的数就不再是 argmax，而单看
   结果发现不了（照样返回一个位姿、照样有个可行率）。
2. **空操作会炸** —— 候选平移不起作用时抛 `RuntimeError`。这条是照
   `postik-smoother-noop`（`uniform_filter1d(size=1)` 恒等）那次教训加的：这里
   对应的坑是给 `cam_to_root_targets` 传了 `workspace_center`，位置被重新居中，
   整个搜索退化成空操作。判据是"分数不变**而可达上界在变**"，所以可行率饱和
   （常态）不会误报。
3. **平局是常态，不是异常** —— 公式 (3) 的 argmax 是个**集合**（K 只覆盖位置极值，
   完全不看腕部朝向），实测 `-QALmP1nHtM_678.2_682.2` 上同分候选 292 个。所以
   `n_tied` 要报出来，且换 `tie_break` 不许改变目标函数值。**这不是学术洁癖**：
   同一个片段上，任意取一个同分成员给出全轨迹 66.7%，取同分集合的最内部点给出
   100.0%——差别全在没进 K 的帧上。

第 3 条也是这条路线上"指标 ≠ 画面"的具体形态：`keyframe_ik_rate` 100% 完全可能
配一个不能用的解，所以对比表里 `kf / 全部帧 / 同分` 三个数要一起看，最后还是要出片。

## ⑤碰撞检测 / 轨迹清洗（`src/web2robot/collision/`、`trajectory/`）

碰撞过滤是有限差分梯度下降、纯 CPU、没有随机源，**要求逐位相同**（比①那条严格）：

```bash
# 1. 隔离对比：把过滤器从整条链里拽出来，喂同一份输入跑旧/新两份实现
envs/rt_env/bin/python scripts/dev/diff_collision_migration.py

# 2. 端到端：同 seed 跑两遍，trajectory.npz 每个 key 逐位相同、robot_sim.mp4 md5 相同
# 3. 出四宫格（源视频 / 不开碰撞 / 新代码 / 旧代码）
scripts/dev/render_quad.sh ...
```

端到端**不能单独作为判据**：上游锚点的随机性会把差异掩盖或伪造成几十度的关节角差 ——
第一次跑就被这个骗过，以为迁移改坏了 108°。

### 独立复核：官方 MuJoCo mesh contacts

上面那三步验的是"我方实现有没有被改坏"，验不了"我方代理判得对不对"。后者用第二个
**独立判据** —— `m7.xml` 里本来就开着的 98 个 mesh 碰撞 geom（只报告，不改轨迹）：

```bash
# 注意要绝对路径：m7_tool.sh 会 cd 到上游 retarget/ 目录
scripts/dev/m7_tool.sh audit_mujoco_contacts.py "$PWD/outputs/retarget/fill_jar_e2e_retarget"
```

它打印三段：两个 MJCF 的碰撞 geom 数（`m7.xml` 98 / `m7_mjx.xml` 0）、上游 geom 集合圈到了
什么（跨臂 contact 应为 0，印证它在 M7 上是瞎的）、以及逐帧分歧。fill_jar 的基线
（2026-08-11，216 帧，已开碰撞纠正）：

| | MuJoCo | 我方代理 | 只有 MuJoCo 判 |
|---|---|---|---|
| 左臂 | 50 帧 / 最深 **8.07 cm** | 96 帧 / 6.20 cm | 3 帧（169/172/173，≤1.86 cm） |
| 右臂 | 34 帧 / 1.15 cm | 46 帧 / 2.36 cm | 0 |

**这两组数不该被当成"通过"**：不开纠正时左臂是 99 帧 / 14.19 cm，所以纠正确实起了作用，
但残留 8.07 cm 是真实网格穿透。已知原因（不用再查）：`enter_thresh=0.04` 只管深过 4 cm 的
（右臂最深 2.36 cm 全在阈值下，整段没被动过），且深帧不收敛 —— 过滤器自己的日志是
`left: fixed 53/71 (remaining 18)`，`w_ee=60` 压着 `w_pen=20` 在 60 步内解不开。
换过滤器参数后拿这张表对比即可。

### 代理盒的标定：怎么量、怎么验（grid 路线，2026-08-20）

上面那张表暴露的问题不是"检测漏了"，而是**代理盒的零点不对**：盒子偏大，很多帧代理
说穿、真实网格说干净。这件事必须拿真实网格 contacts 当真值去标，两阶段：

```bash
# 素材：必须是**没开碰撞过滤**的跑（拿过滤后的产物标定 = 循环论证），落 collcal/prefilter/
# phase1 纯几何穷举盒半长（秒级，不跑过滤器）
scripts/dev/m7_tool.sh sweep_arm_torso_params.py phase1 --lo 0.40
# phase2 真跑过滤器扫门槛（分钟级），默认那一行就是"调参前"基线
scripts/dev/m7_tool.sh sweep_arm_torso_params.py phase2 --half 0.0695 0.119 0.239 \
    --enter_thresh 0.02 0.03 --margin 0.01 0.02
```

3 段 542 帧（`fill_jar` / `sip_coffee` / `-2cNMO9Mm3Q_192.4_209.2`）的结论：

| | 盒半长 [m] | 漏 / 误 | AUC | 穿模帧（真实网格） | 最深 | 手腕挪动 均/最 |
|---|---|---|---|---|---|---|
| 调参前 | `[0.105, 0.135, 0.215]` | 0 / 183 | 0.9997 | 53/542 (9.8%) | 6.07 cm | 2.36 / 15.73 cm |
| 标定后 | `[0.0695, 0.119, 0.239]` | **0 / 0** | **1.0000** | **24/542 (4.4%)** | **4.49 cm** | 2.49 / 16.06 cm |

两件事值得记住，别下次重新推一遍：

- **形状本来就是对的**（调参前 AUC 已经 0.9997），错的是零点。所以标定的目标是"距离 0
  ⇔ 真实网格接触"，不是把检测做得更灵。AUC 在这里的用处是**破平局** —— 只按帧数排会
  出现大片同分配置，`grid_tie_break` 那笔账（66.7% vs 100%）就是这么来的。
- **零点一挪，`enter_thresh` 就不能再兼职了**。旧的大盒子隐含地提供了推出余量，标定后
  的盒子不提供，于是余量必须显式化成 `margin`：*深过 `(enter_thresh − margin)` 才修，
  推到离面 `margin` 才停*。只缩盒不给余量的那一版实测把最坏穿透从 6.07 推到 8.99 cm。

验收（两条路线各一条，都是后台跑）：

```bash
# ① 13 段 grid 重跑，和旧表逐段对比（约 7 小时；底座求解确定性，差别只有过滤器那一步）
nohup bash scripts/dev/run_collcal_ab.sh > outputs/dev/collcal_ab.log 2>&1 &
scripts/dev/m7_tool.sh collcmp_table.py --root outputs/retarget/collcmp_cal \
    --out outputs/dev/collcal_ab_table       # 漏/误两列默认按各路线自己的盒子算
# ② neural 一个字节都没动（预设为空 ⇒ 照旧构造）
bash scripts/dev/check_neural_bytes.sh > outputs/dev/neural_bytecheck.log 2>&1
#   期望 trajectory.npz / metrics.npz / robot_sim.mp4 三个 SAME（2026-08-20 实测全同）
```

#### 验收结果（2026-08-21）：一条判据过了，一条没过

13 段跑完了（`outputs/dev/collcal_ab_table/`），**grid** 的前后对比（`neural` 那三列
逐位不变，已核对，因为预设为空）：

| | 穿躯帧（网格判据） | 有残留的段数 | 代理 vs 网格 帧数差 | 其中 漏 / 误 | 最深 | ik（段均） |
|---|---|---|---|---|---|---|
| 全 13 段 调参前 | 507/1755 (28.9%) | 12/13 | 406 | 17 / 423 | 12.64 cm | 96.7% |
| 全 13 段 标定后 | **234/1755 (13.3%)** | **9/13** | 222 | 222 / **0** | 13.16 cm | 96.7% |
| 留出的 10 段 调参前 | 454/1213 (37.4%) | — | 180 | 6 / 186 | 12.64 cm | 93.8% |
| 留出的 10 段 标定后 | **210/1213 (17.3%)** | — | **198** | 198 / **0** | 13.16 cm | 93.8% |

- **判据二（穿模帧占比不许变差）过了，而且是大幅改善**：28.9% → 13.3%，没有一段变差，
  可行率一位没变（碰撞过滤在 IK 之后，本来就不该动它 —— 这一列没变本身是个正确性检查）。
- **判据一（代理/网格帧数差明显收窄）只在标定用的那 3 段上成立**（226 → 24），留出的
  10 段上 180 → 198，**没收窄，而且方向翻了面**：误报 423 → 0，漏报 17 → 222。

判据一为什么不泛化 —— 是**代理形状的天花板，不是参数没调好**。躯干真身是圆的，用一个
轴对齐盒去拟合：把角上的误报压到 0，就必须把 x 半长压到真身的 0.50 倍（0.0695 vs
0.139），于是**平面方向欠覆盖**，~1.7 cm 以内的真穿透对代理是隐形的。实测
`-0RheyDV3a0_48.6_55.3` 的 90 个残留帧，代理读数是 +0.08 ~ +0.48 cm（"还差半毫米才报警"），
网格读数却是 1.26 cm 已经穿了。所以下一步该做的是**把"检测"和"推出目标"解耦**
（大盒判、标定盒推），或者换个更贴身的代理形状，而不是继续调这三个数字。

顺带确认了一件事，**残留深的帧不是漏检**：`--oo8_XIuOM_900.3_917.4` 最坏那几帧代理读数
是 −1.92 ~ −4.70 cm，代理**报了**，是过滤器没修得动（那段 ik 只有 84.8%，属于源头坏帧）。
所以"漏 222"和"最深 13.16 cm"是两个不同的病，别混成一个。上面这些逐帧的数怎么重出：

```bash
scripts/dev/m7_tool.sh peek_penetration_frames.py \
    "$PWD/outputs/retarget/collcmp_cal/-0RheyDV3a0_48.6_55.3_grid" \
    "$PWD/outputs/retarget/collcmp_cal/--oo8_XIuOM_900.3_917.4_grid" --route grid
#   代理读数的符号就是判据：正数 = 漏检（病在检测），负数 = 报了没修动（病在过滤器/源头）
#   注意传绝对路径 —— m7_tool.sh 会 cd 到上游 retarget/，相对路径会落到别处
```

按"指标≠画面"抽帧看过（`outputs/dev/collcal_ab_table/frames/`，帧号就是上面那条命令
报的最坏帧，`ffmpeg -i <run>/robot_sim.mp4 -vf select=eq(n\,45) -vsync 0 -frames:v 1`）：
1.63 cm 那档（`-0RheyDV3a0` f45）画面上是双手抱在胸前、前臂贴着胸甲，**看不出穿**；
13.16 cm 那档（`--oo8_XIuOM_900.3` f17）**一眼就是坏的** —— 整条左小臂埋进躯干，
指尖从胸口另一侧戳出来。所以这一列必须和深度一起看。

### README 里那两张图：怎么重出、图里的数从哪来

图进 git（`docs/assets/`），所以它比别的产物更容易过期 —— 代码改了、图没重出，
读者看到的就是一张**再也复现不出来**的宣传物料。防这件事的办法是把生成命令和来源 run
写死在这里，任何时候能一条命令重出：

```bash
# ①碰撞修复前后对照图（自动挑"修得最多"的那一帧，当前是 f144）
MUJOCO_GL=osmesa envs/rt_env/bin/python scripts/dev/make_readme_assets.py collision \
    outputs/migration_check/new_nocoll outputs/migration_check/new_coll --lookat 0.15 0 0.30
#   → docs/assets/collision_fix_fill_jar.png
#   预期打印：frame 144: before -10.48 cm -> after +0.04 cm（穿透帧 178 → 141 / 216）

# ②输入-输出并排 GIF
MUJOCO_GL=osmesa envs/rt_env/bin/python scripts/dev/make_readme_assets.py demo \
    outputs/retarget/collcmp/fill_jar_neural --start 20 --count 50 --step 3 --height 250 \
    --out docs/assets/demo_fill_jar.gif
#   → 50 帧 / 约 2.9 MB
```

两条要求：

1. **对照图的两个 run 必须是"同一次 IK、只差碰撞开关"**，脚本会核对 `ik_rate` 一致，
   不一致直接 `SystemExit`。否则图上的差别里混进了 IK 的随机性，就不再是碰撞过滤的功劳。
2. **挑帧规则是"修好得最多"，不是"修复前最深"。** 最深的那些帧过滤器未必修得动
   （`w_ee=60` 的保真项压着推出项），拿修不动的帧当示意图就是自欺。所以脚本挑的是
   `argmax(修复后深度 − 修复前深度)`，并且**同时把整段的穿透帧数打出来**
   （178 → 141，降了没清零）—— 一张挑出来的好帧配一个全段的真实数字，才不算选择性展示。

### 轨迹清洗：空洞填补的位置感知策略

`trajectory/traj_cleanup.py` 的空洞策略是**按位置**分的（2026-08-11 定），`FILL_REST`
是最后兜底。改这块必须跑两样：

```bash
# 1. 单测钉住策略（TestGapPolicyByPosition，6 个用例）
envs/rt_env/bin/python -m unittest tests.test_retarget_modules -v

# 2. 看画面 —— 结尾"沿袭"vs"渐入静息位"左右对比
MUJOCO_GL=osmesa envs/rt_env/bin/python scripts/dev/…（一次性脚本，见下）
```

判据不是数字而是画面：serve_cake 结尾（右手 44 帧 / 2.9 s、左手 17 帧 / 1.1 s）旧策略
渐入静息位，等于**凭空编出最多 74.5° 的关节运动**（右臂逐关节最大
`[10.9, 22.3, 10.2, 45.1, 16.2, 5.7, 74.5]`，均值 18.2°；左臂 24.0°/均值 9.4°）。
f175 / f188 两张图能直接看出来：新策略两手停在最后一次测到的持盘姿态，旧策略两条手臂
垂到体侧默认位。存档 `outputs/dev/tail_policy_serve_cake/tail_policy_hold_vs_rest.mp4`。

结尾保持出来的帧**仍然标 `FILL_HOLD` 而不是 `OK`**，长度记在 `report["tail_hold"]`
并打一行 ⚠ —— 单测 `test_long_tail_hold_is_reported_not_silent` 钉的就是这一点。
全片段普查（11 个官方片段）确认这次改动只翻了三处片尾（serve_cake 左 17f / 右 44f、
ours_webapple 右 58f），别的空洞一帧没动。

## M7 机器人定义（`src/web2robot/robots/m7/`、`assets/robots/m7/`）

两个验收脚本，输出要和上一次逐字节一致：

```bash
scripts/dev/m7_tool.sh verify_m7_mjx_fk.py            # 期望 0.0000 mm / 0.0000 deg  MATCH ✓
scripts/dev/m7_tool.sh check_handframe_convention.py  # 期望 m7 左手 finger=+y thumb=-x palm=+z，右手镜像

# 再加一档：拿一段真实重定向轨迹逐帧验，而不是只验资产里写死的 home 姿态
scripts/dev/m7_tool.sh check_handframe_convention.py \
    --traj outputs/legacy_runs/runs/m7/validation/fill_jar
#   期望 "违反约定的帧: 0/216 ✓"，且两只手整段各只出现过一种轴向组合
#   （--traj 要给绝对路径或相对**上游 retarget/** 的路径，m7_tool.sh 会 cd 过去）
```

**永远不要只验一侧**，理由见 [`PITFALLS.md`](PITFALLS.md) 第 16 条。
动完资产位置、删掉旧目录之后**必须再跑一次端到端**，理由见第 11 条。

## L3.4 机器人定义（`src/web2robot/robots/l3_4/`、`assets/robots/l3_4/`）

第二台机器人，和 M7 **并列可切换**（`--robot m7|l3_4`）。它的上肢和 M7 逐位同构，
所以验收的重点和 M7 那节不同：不是"数值对不对"，而是**"同构有没有被偷偷写成依赖"**。

```bash
# 1. 资产：从厂家原包重建，七步自检，任何一步不过就 SystemExit
envs/rt_env/bin/python scripts/dev/build_l3_4_assets.py --force
#   期望七行全过，其中三行是这台机器人的立身之本：
#   [4/7] hand_frame quat 现算 + 逐轴断言（左 finger+y/thumb-x/palm+z，右镜像）
#         —— 算出来的两个 quat 和 M7 已提交的那两个逐位相同，是对整条链的独立交叉验证
#   [6/7] 双臂链与 m7_mjx.xml 逐 body / 逐关节比对（期望最大偏差 ~3.8e-07）
#         —— **这一行就是"借 M7 根模型 ckpt"的全部依据；它红了就必须重训，不许照跑**
#   [7/7] 对厂家自带的 l3.4.xml 校验（55 个关节 axis/range + 55 个 body pos 全同）
#         —— 回答"厂家给的 .urdf 和 .xml 是不是一致可用"，不靠肉眼看

# 2. 表 vs MJCF：12 个用例，秒级
envs/rt_env/bin/python -m unittest tests.test_l3_4_robot -v

# 3. hand_frame 逐帧验（同 M7，永远两只手都验）
envs/rt_env/bin/python -m unittest tests.test_l3_4_robot.TestHandFrameConvention -v
```

三条一定要知道的：

**① 真源是各自的 MJCF，不是另一台机器人的表。** L3.4 的限位表、start_config、采样参数
和 `robots/m7/` 数值相同（量出来的：43 个同名关节 axis/range 全同、43 个同名 body 的
pos/quat 全同），但两个包**一行代码都不共享**，也没有 alias。测试断言的是
"表 == `l3_4.xml` 里那个关节的 `range`"，**不是** "表 == M7 的表" —— 后者会在哪天真拿到
不同批次的机器人时红在"和 M7 不一样"上，而那时候不一样才是对的。

**② 借 M7 的 ckpt 有明确的失效条件。** 根位姿模型的输入输出由
`waist_pitch_link → hand_frame` 这条链决定，而这条链两台逐位相同，所以
`--robot l3_4 --ckpt runs/m7/taskspace_v2/checkpoints/final.pt` 是有依据的，不是凑合。
**腿一旦从 `LOCKED_JOINTS` 里解锁，base_link 相对地面的高度就变了，这个结论立刻失效。**
（`--root_solver grid` 那条路线压根不用模型，只用 IK；但上游 `test.py` 无条件 `_load_model`，
所以还是得传 `--ckpt`。）

**③ 锁死不是"没人去写"。** 腰/颈/腿 17 个自由度由 `env._apply_locked()` 在**每次
`mj_forward` 之前**按住，因为上层（碰撞过滤、渲染）会直接改 `data.qpos` 再 forward。
测试里专门有一步把这 17 个 qpos 篡改成 0.37 再走一次，验它们回到锁定值。

**碰撞参数不用重标。** 躯干代理盒挂在 `waist_pitch_link` 上，两台机器人这个 mesh 的
AABB 逐位相同（中心 `[0.0057, 0, 0.2166]`、半长 `[0.1313, 0.17, 0.2326]`），
`M7CapsuleModel` 用到的 `BONES` / `FINGERTIPS` 名字在 L3.4 里全部存在 —— 所以
`presets.py` 里那份**已标定**的 grid 预设是精确适用的，不是"按比例套一版粗略的"。

**加了 L3.4，M7 的产物一个字节都没变** —— 这是硬要求，所以有脚本：

```bash
bash scripts/dev/check_m7_unchanged_by_l3_4.sh > outputs/dev/l34_m7_unchanged.log 2>&1
#   同片段（-1r9yl-P-Ao_86.3_90.8）/ 同 seed / 同路线（neural）/ 两条碰撞过滤都开，
#   和 L3.4 改动**之前**留下的 outputs/dev/neural_bytecheck/base/ 比 md5
#   2026-08-20 实测 trajectory.npz / metrics.npz / robot_sim.mp4 三个 SAME
#   （robot_sim.mp4 = 205d96dba4a701e4be19a88ff1ec0483，和 patches/README.md 里那个数一致）
```

只跑一遍就够：参照物是 A1 那次标定验证留下的，时间戳早于 L3.4 的所有改动。
为什么不能只看代码：`sim/robots/__init__.py` 里那两行 `from web2robot.robots.l3_4 import ...`
是**模块顶层**的，跑 M7 也会执行 —— "顶层 import 应该没有副作用"和"产物没变"是两件事。

**腰以下在画面里是空的。** 厂家包里一个 mesh 都没有；94 个零件和 M7 相同（质量/惯量/COM
逐位相同）直接 symlink，14 个腿部 + 盆骨 `base_link` 没有正确的 mesh（M7 那个同名
`base_link.STL` 是升降柱底座，是另一个零件），几何被删掉。上肢重定向的每个数字都不受
影响，但**出片之前得把腿的 mesh 要到**，见 [`BACKLOG.md`](BACKLOG.md)。

## `evidence/` 里的数（`src/web2robot/eval/`）

```bash
envs/rt_env/bin/python -m unittest tests.test_depth_benchmark -v   # 19 个用例，0.3 秒
envs/rt_env/bin/python scripts/dev/render_depth_benchmark_fig.py   # 重画汇总图
```

这份测试的作用和别的不一样：它不防"代码改坏"，防的是**论文里的数字和仓库里的证据
悄悄脱钩**。所以断言写的是具体数值不是"大于小于" —— 把中位偷偷改成均值，ABF12 的
11.0 会变 11.26、SMu41 的 3.5 会变 6.7，测试当场变红（验证过）。

引用那张深度误差表时必须一起写上的两句 caveat，见
[`PROJECT_LAYOUT.md` §3.1](PROJECT_LAYOUT.md)。

---

## 迁移/重构的验收方法论（五步，后续模块照抄）

这套是 2026-08 那轮重构定下来的，五步缺一步都出过事：

1. **逐行 diff 证明是纯移动** —— 先证明"没改逻辑"，再谈别的。
2. **隔离对比**：把模块从整条链里拽出来，喂同一份输入跑旧/新两份实现。纯 CPU 的
   要求逐位相同；有 GPU/随机源的降级到"判决一致 + 信号未越阈"。
3. **端到端必须固定 `--seed`**，否则上游的随机锚点会把差异伪造成几十度关节角差。
4. **确认没有留下重复副本**。留两份比留一份危险 —— 下次改代码会改到错的那一份，
   而这正是重构要消灭的失效模式。删除后再跑一次端到端（拼接路径躲得过 grep）。
5. **出片，用眼睛看。**

另外有一个整体性的指标：**迁移做对了，上游 patch 的行数就该往下走**
（313 → 233 insertions）。逻辑进 `src/`、上游只剩接线，patch 就该变小。

它变大不一定是错，但**必须当场解释清楚多的是什么**：2026-08-18 加
`--root_solver grid` 让它涨到 342，多出来的是一个 if/else 开关＋把 FK/IK 包成
callable 的接线；2026-08-19 加 `--object_tracking` 和 `--action_refine` 再涨到 428，
多出来的是三组 argparse 选项、一处参数矛盾检查、两个调用点；2026-08-20 加
`--atf_preset` 和第二台机器人 L3.4 涨到 520，多出来的是"去 `presets.py` 查表"的接线
和一台新机器人在注册表 / IK / 手部重定向器三处的注册（明细都在
[`external/patches/README.md`](../external/patches/README.md)）。
判据不是行数本身，是"上游文件里有没有出现只有那里才有的方法逻辑"——
没解释的增长才是警报。

**新开关还要额外过一道"默认关闭 ⇒ 产物逐字节不变"**：同一段片段、同一个 seed
跑 base / 显式关 / 显式开三遍以上，原有产物的 md5 必须全同，新开关只许**多**
产物、不许改产物。目前有两道，都是脚本化的：
[`check_object_tracking_bytes.sh`](../scripts/dev/check_object_tracking_bytes.sh)（3 遍）
和 [`check_action_refine_bytes.sh`](../scripts/dev/check_action_refine_bytes.sh)（4 遍，
含 `--object_tracking on --action_refine none` 这种组合）。2026-08-19 实测两道都是
原有 5 个产物全同、新文件只增不改。
这比"读代码看默认值"强 —— 默认值对但 import 有副作用、或者 argparse 顺序变了影响
随机数流，都只有比 md5 才发现。

**参数矛盾要当场退出，不许静默降级。** `--action_refine mpc|rl` 缺
`--object_tracking on` 时直接 `SystemExit`（在 `run()` 第一行，不是跑完才报），
`mpc` / `rl` 求解器本身也 `NotImplementedError` 而不是退回 Replay。理由是同一条：
一份"以为精修过"的数据会直接进训练集，比一次失败贵得多。
