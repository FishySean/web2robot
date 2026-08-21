"""L3.4（rel3_4）机器人定义 —— MJCF/关节名/末端帧/IK 种子姿态，**不 import 任何重定向框架**。

这一层回答"这台机器人长什么样"，跟用哪个重定向框架无关，所以也不该知道框架的存在。
**加一台新机器人 = 在 ``robots/`` 下新建一个子包，别的模块一行不用改**；框架侧怎么
认识它是框架侧的事（EgoInfinity 的注册在它自己的 ``sim/robots/__init__.py``，
见 ``external/patches/``）。这一条由 ``tests/test_module_boundaries.py`` 钉住。

为此这一层只出**数据和自己的类**，不出上游类型：``CONFIG`` / ``ENV_SPEC`` /
``HAND_JOINT_SPEC`` / ``ik_spec()`` 都是纯 dict/list，``L34Env`` 实现上游 ``BaseEnv``
的接口但不继承它。

## L3.4 是什么、这一版做到哪

完整人形：双 7-DoF 臂 + 两只 12-DoF 五指手 + 3-DoF 腰 + 2-DoF 颈 + 两条 6-DoF 腿。
**这一版只做上肢**，腰/颈/腿 17 个自由度锁死在 ``CONFIG`` 旁边的 ``LOCKED_JOINTS``
里（是锁死不是删除：MJCF 里 hinge 都还在，从那张表删掉一行就解锁一个自由度）。

## 三条要先知道的事实

**1. 上肢和 M7 逐位同构，但两个包不共享一行代码。**
43 个同名关节的 axis/range 全同、43 个同名 body 的 pos/quat 全同、
``base_link → waist_pitch_link`` 的变换也相同（证据见
``scripts/dev/build_l3_4_assets.py`` 的 docstring）。所以这里的限位表、start_config、
采样参数和 ``robots/m7/`` 数值相同 —— 是量出来的结果，不是抄的结论。**没有 import
过去**是刻意的：一台机器人的定义不该依赖另一台的模块存在。
防漂靠 ``tests/test_l3_4_robot.py``：它拿 **MJCF** 当唯一真相，断言表里每个限位都等于
``l3_4.xml`` 里那个关节的 ``range``，而不是拿 M7 的表当参照。

**2. 手是同一只 12 自由度手，"xhand" 只是 mesh 路径里的目录名。**
不是 11 自由度的另一款，MANO→手的映射原样成立。依据和逐行理由写在
``hand_mapping.py`` 的 docstring 里。

**3. 借 M7 的根模型 checkpoint 是有依据的，不是凑合。**
根位姿模型学的是"给定手的任务空间目标 → 底座放哪"，输入输出都由
``waist_pitch_link → hand_frame`` 这条链决定，而这条链两边逐位相同
（``build_l3_4_assets.py`` 会对着 ``m7_mjx.xml`` 逐 body/逐关节验，不一致就当场报错）。
**腿一旦解锁，base_link 相对地面的高度会变，这个结论立刻失效，必须重训。**

## hand_frame 的轴向约定（吃过亏的地方）

必须是 **finger+y / thumb−x / palm+z（左手），右手镜像**（thumb+x / palm−z）。
2026-07-24 在 M7 上踩过：检查脚本只验左手，错误地得出"两只手同一套约定"，结果右手
被建成和左手完全一样（掌面/拇指翻了 180°）。L3.4 的两个 quat 由
``build_l3_4_assets.py`` **现算**（拿中指/拇指末节 body 的物理方向，两侧掌面法向取
相反符号），算完还逐轴断言对上这张表；算出来的两个数和 M7 已提交的那两个逐位相同 ——
这是对整条链的独立交叉验证。**永远不要只验一侧。**

## 腰以下在画面里是空的

厂家包里一个 mesh 都没有。94 个零件和 M7 相同（质量/惯量/COM 逐位相同，是同一批
零件），直接 symlink 过去；14 个腿部 + 盆骨 ``base_link`` 没有正确的 mesh
（m7 有同名 ``base_link.STL``，但那是**升降柱底座**，是另一个零件），它们的几何被删掉。
上肢重定向的每个数字都不受影响（IK 串链根和碰撞代理盒都挂在 ``waist_pitch_link``），
但**出片之前得把腿的 mesh 要到**，见 ``docs/BACKLOG.md``。

## 验证脚本

```bash
envs/rt_env/bin/python scripts/dev/build_l3_4_assets.py --force   # 重建资产，7 步自检
envs/rt_env/bin/python -m unittest tests.test_l3_4_robot -v       # 表 vs MJCF 一致性
```
"""
from web2robot.robots.l3_4.config import CONFIG, ENV_SPEC, LOCKED_JOINTS
from web2robot.robots.l3_4.env import L34Env, _MJCF_PATH, _SCENE_PATH
from web2robot.robots.l3_4.hand_mapping import HAND_JOINT_NAMES, HAND_JOINT_SPEC
from web2robot.robots.l3_4.ik_config import ik_spec
from web2robot.robots.l3_4.sample_config import SAMPLE_CONFIG

#: 全身 MJCF（``assets/robots/l3_4/l3_4.xml``）。导出它是因为框架侧还有第二处要用：
#: 上游 ``kinematics/wrist_ik.py::RobotIKConfig.l3_4`` 要拿它建 IK 串链
#: （pytorch_kinematics）。别让框架侧自己拼路径 —— M7 迁移时正是那种拼法躲过了 grep，
#: 删掉旧目录后端到端当场 FileNotFoundError。
MJCF_PATH  = _MJCF_PATH
#: 带地面/光照的场景（``scene_vis.xml``），渲染与 hand_frame 检查用。
SCENE_PATH = _SCENE_PATH

__all__ = ["CONFIG", "ENV_SPEC", "L34Env", "SAMPLE_CONFIG", "MJCF_PATH", "SCENE_PATH",
           "HAND_JOINT_SPEC", "HAND_JOINT_NAMES", "ik_spec", "LOCKED_JOINTS"]
