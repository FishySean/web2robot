"""三档模式的求解器 —— 只有 Replay 是实现了的，MPC / RL 是占位。

论文 §3.2.2 的阶梯（照抄）：

* **Replay** —— 直接执行参考动作，不搜不学。跟得上的块用这档，绝大多数块都是。
* **MPC** —— 在参考轨迹附近采一批短时域动作序列，挑跟踪误差最小的那条。
* **RL** —— PPO 残差手部策略，给 MPC 也搞不定的块兜底。

第一期只交判决，所以后两档在这里**明确报错**，不悄悄退回 Replay。理由和模块一的
``sam2_foundationpose`` backend 一样：静默降级会让下游拿到一份"以为精修过了"的数据，
那比直接失败危险得多。报错信息里写清缺的是什么，谁来接手一眼就知道从哪开始。
"""
from __future__ import annotations

import numpy as np

#: 阶梯顺序，越靠后越贵。``blocks._next_mode`` 按这个顺序升级。
MODES = ("replay", "mpc", "rl")


def replay_solve(actions: np.ndarray, start: int, stop: int, **kw) -> np.ndarray:
    """Replay：原样返回这一块的参考动作。

    返回的是**拷贝**，不是视图 —— 让调用方可以放心改，不会回头污染参考轨迹。
    """
    actions = np.asarray(actions)
    if not 0 <= start <= stop <= len(actions):
        raise ValueError(f"块范围 [{start},{stop}) 越界，轨迹只有 {len(actions)} 帧")
    return actions[start:stop].copy()


def mpc_solve(actions: np.ndarray, start: int, stop: int, **kw) -> np.ndarray:
    """MPC：论文里是在参考附近采短时域动作序列再挑最优的。**还没实现。**"""
    raise NotImplementedError(
        "MPC 档还没实现。缺的是两样东西，都不是这个包里能补的：\n"
        "  1. 一个**场景里有物体**的仿真 rollout —— 采样出来的动作序列要能问"
        "\"执行完物体在哪\"。现在的 robot_sim 是空场景，"
        "web2robot/refine/attach.py 用的是刚连假设，不够用来做搜索（它对任何"
        "动作都给出确定的结果，搜索会退化）。\n"
        "  2. 采样的时域长度、样本数、代价权重 —— 论文没给这些数。\n"
        "先跑 --action_refine none 看判决，确认真有块需要升级再动手。")


def rl_solve(actions: np.ndarray, start: int, stop: int, **kw) -> np.ndarray:
    """RL：论文里是 PPO 残差手部策略。**还没实现。**"""
    raise NotImplementedError(
        "RL 档还没实现。它比 MPC 还多一层依赖：除了带物体的仿真，还需要训练侧的"
        "基建（PPO、奖励用的是 refine/score.py 里的 r_obj_t、以及每块一个策略还是"
        "全段共享一个，论文没说清）。顺序上应该等 MPC 跑通、确认它兜不住的块长什么"
        "样，再决定 RL 要不要做。")


#: 模式名 → 求解器。``rl`` / ``mpc`` 调下去会抛 NotImplementedError，这是故意的。
SOLVERS = {"replay": replay_solve, "mpc": mpc_solve, "rl": rl_solve}
