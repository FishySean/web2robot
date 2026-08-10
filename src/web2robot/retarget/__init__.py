"""重定向 —— 把人手轨迹变成机器人关节轨迹这一环里**属于我方的策略**。

具体的求解器（IK、根位姿估计、聚类打分）是 EgoInfinity 的，放在 ``external/``；
这一层放的是"怎么用它们"以及原始实现里没有的东西：

- :mod:`~web2robot.retarget.root_anchor` —— best-of-N 根锚点采样。根估计器是生成
  模型，同一段片段每次跑出的躯干锚点不同；抽 N 次按 IK 收敛率取最好的那个。
- :mod:`~web2robot.retarget.fallback` —— 坏帧/丢帧兜底在流水线里的编排（判据本身
  在 :mod:`web2robot.trajectory`）。输入侧重做填补，输出侧渐入静息位。

两个模块都**零上游 import**：需要上游的东西一律以 callable 或数组参数传进来。
所以它们能被纯 numpy 单测，不需要 GPU、不需要 checkpoint、不需要 ``external/``
在位。机器人本身的属性（M7 的手部映射表、IK 串链限位）不在这里，在
``web2robot/robots/m7/``。
"""
from web2robot.retarget.fallback import (
    InputCleanup, apply_rest_fallback, clean_input_wrists, relax_fingers_on_rest,
    status_overlay_text,
)
from web2robot.retarget.root_anchor import AnchorChoice, sample_best_anchor

__all__ = ["AnchorChoice", "sample_best_anchor",
           "InputCleanup", "clean_input_wrists", "apply_rest_fallback",
           "relax_fingers_on_rest", "status_overlay_text"]
