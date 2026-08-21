"""动作分级精修（EgoEngine arXiv 2606.12604 §3.2.2 的自适应模式切换）。

**这个包只做"判断"，不做"求解"** —— 第一期交的是：把轨迹按固定长度切块、
用物体位姿跟踪误差给每块打分、超阈值早停、以及"这块要不要升级到更贵的模式"的
判决。三档模式 Replay → MPC → RL 里，只有 Replay（照抄参考轨迹，不动）是实现了的，
MPC / RL 是占位，调用直接 :class:`NotImplementedError`，不静默降级。

方法来源是论文，不是自创设计。论文给了的东西照抄（块长 H=20、误差和奖励的形式、
三档阶梯、当前块+下一块的联合窗口）；论文**没给**的数（λp、λR、以及那个阈值 C 的
具体值）做成可配置参数，默认值的来路写在各自的 docstring 里 —— 谁改这些数，先看
那段解释。

和模块一的关系：这里吃的就是 :mod:`web2robot.twin` 写出来的 ``object_poses.npz``。
所以 ``--action_refine mpc|rl`` 必须同时开 ``--object_tracking on``，缺了参考位姿
没法打分，那种情况直接报错退出。

各文件：

* :mod:`~web2robot.refine.score` —— 逐帧误差 e_t / 奖励 r_obj_t，纯几何，无 IO
* :mod:`~web2robot.refine.attach` —— 从机器人手的偏差推物体的"执行后位姿"
  （这是我们对论文里"仿真里 rollout"那一步的替代，见该模块 docstring）
* :mod:`~web2robot.refine.blocks` —— 切块、早停、升级判决、两块联合窗口
* :mod:`~web2robot.refine.modes` —— 三档模式的求解器（只有 replay 是真的）

零上游 import；也不反向依赖 ``retarget/root_anchor.py`` / ``root_grid.py``。
"""
from web2robot.refine.attach import (
    conjugate_delta,
    pose_compose,
    pose_delta,
    pose_inverse,
    predict_object_poses,
)
from web2robot.refine.blocks import (
    H_DEFAULT,
    BlockPlan,
    BlockScore,
    RefineConfig,
    plan_blocks,
    score_block,
    split_blocks,
)
from web2robot.refine.modes import MODES, SOLVERS, mpc_solve, replay_solve, rl_solve
from web2robot.refine.score import ErrorWeights, reward, step_errors

__all__ = [
    "BlockPlan", "BlockScore", "ErrorWeights", "H_DEFAULT", "MODES",
    "RefineConfig", "SOLVERS", "conjugate_delta", "mpc_solve", "plan_blocks",
    "pose_compose", "pose_delta", "pose_inverse", "predict_object_poses",
    "replay_solve", "reward", "rl_solve", "score_block", "split_blocks",
    "step_errors",
]
