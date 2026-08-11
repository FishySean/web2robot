# 改完之后怎么验收

这个工程的验收判据**一个模块一套**，因为各模块的确定性不同：碰撞过滤是纯 CPU
无随机源，可以要求逐位相同；质检里跑着 GPU 神经网络，逐位相同是不可能的目标。
拿错判据的后果是双向的 —— 要么放过真错误，要么把浮点噪声当成 bug 追半天。

约定见 [`CONVENTIONS.md`](CONVENTIONS.md)，坑见 [`PITFALLS.md`](PITFALLS.md)。

**所有模块共通的最后一步：出片，用眼睛看。** 指标 ≠ 画面。

```bash
envs/rt_env/bin/python -m unittest discover -s tests -v     # 秒级，全套 125 个用例
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

另外有一个整体性的指标：**上游 patch 的行数应该往下走**（313 → 233 insertions）。
逻辑进 `src/`、上游只剩接线，patch 就该变小；它变大就是有人在往上游写实质逻辑了。
