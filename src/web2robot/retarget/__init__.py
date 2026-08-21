"""重定向 —— 把人手轨迹变成机器人关节轨迹这一环里**属于我方的策略**。

具体的求解器（IK、根位姿估计、聚类打分）是 EgoInfinity 的，放在 ``external/``；
这一层放的是"怎么用它们"以及原始实现里没有的东西：

- :mod:`~web2robot.retarget.root_anchor` —— best-of-N 根锚点采样。根估计器是生成
  模型，同一段片段每次跑出的躯干锚点不同；抽 N 次按 IK 收敛率取最好的那个。
- :mod:`~web2robot.retarget.root_grid` —— **另一条**根位姿求解路线：网格搜索使
  关键帧 IK 可行率最大的**静态**底座位姿（Qwen-RobotManip 公式 3）。和上面那条是
  两种粒度（逐帧解 vs 静态解），不是同一件事的两种算法，所以并列摆着可切换。
- :mod:`~web2robot.retarget.fallback` —— 坏帧/丢帧兜底在流水线里的编排（判据本身
  在 :mod:`web2robot.trajectory`）。输入侧重做填补，输出侧渐入静息位。

三个模块都**零上游 import**：需要上游的东西一律以 callable 或数组参数传进来。
所以它们能被纯 numpy 单测，不需要 GPU、不需要 checkpoint、不需要 ``external/``
在位。机器人本身的属性（M7 的手部映射表、IK 串链限位）不在这里，在
``web2robot/robots/m7/``。
"""
from web2robot.retarget.fallback import (
    InputCleanup, apply_rest_fallback, clean_input_wrists, relax_fingers_on_rest,
    status_overlay_text,
)
from web2robot.retarget.root_anchor import AnchorChoice, sample_best_anchor
from web2robot.retarget.root_grid import (
    GridSpec, KeyframeSet, RootPoseSolution, build_translation_grid, estimate_reach,
    gravity_yaw_candidates, make_keyframe_scorer, select_extremal_keyframes,
    solve_root_pose_grid,
)

__all__ = ["AnchorChoice", "sample_best_anchor",
           "KeyframeSet", "GridSpec", "RootPoseSolution",
           "select_extremal_keyframes", "build_translation_grid",
           "solve_root_pose_grid", "make_keyframe_scorer",
           "gravity_yaw_candidates", "estimate_reach",
           "InputCleanup", "clean_input_wrists", "apply_rest_fallback",
           "relax_fingers_on_rest", "status_overlay_text"]
