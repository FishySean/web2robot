"""Best-of-N 根锚点采样 —— 重定向里唯一一处"同一段输入跑两次结果不同"的地方。

## 为什么需要它

EgoInfinity 的根位姿估计器是个 **flow-matching 生成模型**：``model.sample()`` 从一个
随机先验积 ODE 出来，所以同一段片段每次跑出的躯干锚点都不一样，而锚点选偏一点，
整条手臂就可能有一半的帧 IK 解不出来。原始实现只抽一次就用，好坏全看运气。

这里的做法是抽 N 次独立估计，用 **IK 收敛率**给每个锚点打分，留最高的那个。
``n_samples=1`` 时和原始单发路径完全等价（连打印都不多一行）—— 这一条很重要，
因为所有既有基线都是单发跑出来的，改这个模块不许动它们。

## 为什么参数是两个 callable

``estimate_fn`` / ``select_fn`` 是调用方传进来的闭包，这个模块因此**零上游 import**：
根估计器、IK 求解器、聚类打分全是 EgoInfinity 的东西，但"抽 N 次取最好"这个策略
不是。换重定向框架时这一层不用改。副作用是它能被纯 numpy 的假 callable 单测，
不需要 GPU、不需要 checkpoint（见 ``tests/test_root_anchor.py``）。

## 复现性

``seed`` 一给，第 i 次采样前把全局 RNG 播成 ``seed + i``。播种走 ``seed_fn``
而不是直接 ``torch.manual_seed``，同样是为了让这个模块不硬依赖 torch —— 默认值
是懒加载的 torch 播种器，只有真的传了 seed 才会 import。

2026-08-10 从上游 ``scripts/test.py`` 里搬出来（原来是 run() 中间的 20 行内联循环）。
"""
from dataclasses import dataclass
from typing import Any, Callable, Optional


def _torch_seed(seed: int) -> None:
    """默认播种器：播 torch 全局 RNG（flow-matching 的先验就是从它抽的）。

    懒 import：不给 ``seed`` 的调用方（默认路径）不该为此拉进 torch。
    """
    import torch
    torch.manual_seed(seed)


@dataclass
class AnchorChoice:
    """选中的那一次采样的全部产物。

    ``ik_rate`` 是这个锚点的 IK 收敛率（0..1），也就是打分用的那个数。
    ``kf_positions`` / ``Rs`` / ``ts`` 必须和 ``R_anchor`` / ``t_anchor`` 来自
    **同一次**采样 —— 混用不同次的窗口估计和锚点会让后面的逐帧混合直接跑偏。
    """
    ik_rate:      float
    kf_positions: Any
    Rs:           Any
    ts:           Any
    R_anchor:     Any
    t_anchor:     Any


def sample_best_anchor(
    estimate_fn: Callable[[], tuple],
    select_fn:   Callable[[Any, Any], tuple],
    n_samples:   int = 1,
    seed:        Optional[int] = None,
    log:         Callable[[str], None] = print,
    seed_fn:     Callable[[int], None] = _torch_seed,
) -> AnchorChoice:
    """抽 ``n_samples`` 次根位姿估计，返回 IK 收敛率最高的那一次。

    Parameters
    ----------
    estimate_fn
        零参可调用，返回 ``(kf_positions, Rs, ts)``；上游是
        ``estimate_root_poses(model, ...)`` 的闭包。每次调用都要重新抽（生成模型）。
    select_fn
        ``select_fn(Rs, ts)`` → ``(R_anchor, t_anchor, ik_rate)``；上游是
        ``select_best_anchor(..., opt, workspace_center, n_clusters=...)`` 的闭包。
    n_samples
        抽几次。``1``（默认）＝原始单发行为，且**不打印任何东西**。
    seed
        给了就在第 i 次采样前播 ``seed + i``；``None`` 表示不播（不确定性）。
    log
        进度输出。``n_samples == 1`` 时一行都不发。
    seed_fn
        播种器，默认播 torch 全局 RNG。单测里换成假的就不用装 torch。

    Raises
    ------
    ValueError
        ``n_samples < 1``。默默当成 1 会把"配置写错了"变成"结果莫名其妙"。
    """
    if n_samples < 1:
        raise ValueError(f"n_samples 至少为 1，收到 {n_samples}")

    best: Optional[AnchorChoice] = None
    for i in range(n_samples):
        if seed is not None:
            seed_fn(seed + i)
        kf_positions, Rs, ts = estimate_fn()
        R_anchor, t_anchor, ik_rate = select_fn(Rs, ts)
        if n_samples > 1:
            log(f"  [sample {i+1}/{n_samples}] anchor ik_rate={ik_rate*100:.1f}%")
        # 严格 > ：并列时留**先**抽到的那次，这样 seed 固定 → 结果固定。
        if best is None or ik_rate > best.ik_rate:
            best = AnchorChoice(ik_rate, kf_positions, Rs, ts, R_anchor, t_anchor)
    if n_samples > 1:
        log(f"  best-of-{n_samples}: anchor ik_rate={best.ik_rate*100:.1f}%")
    return best


__all__ = ["AnchorChoice", "sample_best_anchor"]
