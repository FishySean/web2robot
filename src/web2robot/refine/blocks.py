"""切块 → 打分 → 早停 → 升不升级 —— EgoEngine §3.2.2 的判决逻辑。

论文照抄的部分：

* **块长 H = 20 个控制步**（论文明写的数，这里就是默认值）。
* **累计跟踪误差超过阈值就早停**这一块，不用把整块跑完。
* **三档阶梯 Replay → MPC → RL**，按需升级：能照抄就照抄，不行才上局部轨迹搜索，
  再不行才上残差策略。
* **两块联合窗口**："jointly solving the current and next chunks but executing only
  the current chunk once both are feasible" —— 当前块自己过了但把下一块逼进死角，
  这块也不算过。这是防贪心的，别为了省事砍掉。

论文**没给**的、这里必须自己定的两处，以及定的理由：

1. **阈值 C 的数值。** 论文只说"cumulative tracking error exceeds threshold"。这里
   写成 ``per_frame_budget × 块长``而不是一个固定常数：一段 69 帧的片段切成
   20/20/20/9，最后那块 9 帧要是和 20 帧共用一个累计上限，它几乎不可能被判坏 ——
   长度感知这一点和 ``trajectory/traj_cleanup.py`` 的填补是同一个道理。
   ``per_frame_budget`` 默认 0.05，即"平均每帧 5 cm 平移误差（或 0.05 rad ≈ 2.9°
   旋转，或两者的加权合成）以内算跟得上"。这个数是拍的，拍的依据是现有链路的量级：
   `metrics.npz` 里 IK 本身的 ``pos_err`` 就在厘米级，把预算设在它的量级上，判的才
   是"物体没跟上"而不是"IK 有残差"。**要调先看这段。**
2. **奖励里那个 C**（``c_reward``）和上面这个阈值在论文里都写 C，但一个是逐步奖励的
   偏移常数、一个是累计误差的上限，量纲不同。这里拆成两个参数，默认 ``c_reward``
   = 1.0，只影响 :func:`~web2robot.refine.score.reward` 的绝对值、不影响任何判决 ——
   判决全看误差，不看奖励。奖励留着是给以后 RL 那档用的。

一个论文没提但必须处理的情况：**这一块根本没法判**。孪生标了不可信（``trust``
为假）、或者手那一帧没检出，误差就是 NaN。这种块的判决是 ``unknown``，不是
``ok`` —— 把量不到的当成好的，是这条流水线最容易犯的错。

纯 numpy，无 IO。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

from web2robot.refine.score import ErrorWeights, step_errors

#: 论文明写的块长：H = 20 个控制步。
H_DEFAULT = 20


@dataclass(frozen=True)
class RefineConfig:
    """判决用到的全部数。默认值的来路见模块 docstring。"""

    horizon: int = H_DEFAULT
    per_frame_budget: float = 0.05       # 阈值 C ← 论文没给数值
    c_reward: float = 1.0                # 奖励里的 C ← 论文没给数值
    weights: ErrorWeights = field(default_factory=ErrorWeights)
    min_valid_frac: float = 0.5          # 有效帧不到这个比例 → unknown，不判 ok
    keep_tail: bool = True               # 末尾不足 H 帧的余块保留（丢掉就是静默丢帧）

    def __post_init__(self):
        if self.horizon < 1:
            raise ValueError(f"horizon 至少 1，收到 {self.horizon}")
        if self.per_frame_budget <= 0:
            raise ValueError(f"per_frame_budget 要正数，收到 {self.per_frame_budget}")
        if not 0.0 <= self.min_valid_frac <= 1.0:
            raise ValueError(f"min_valid_frac 要在 [0,1]，收到 {self.min_valid_frac}")

    def budget(self, block_len: int) -> float:
        """这一块的累计误差上限 —— 按块长缩放，理由见模块 docstring 第 1 条。"""
        return self.per_frame_budget * int(block_len)


@dataclass
class BlockScore:
    """一块的打分结果。"""

    index: int
    start: int
    stop: int                       # 右开
    n_frames: int
    n_valid: int
    e_mean: float                   # 有效帧的均值，全无效时 NaN
    e_max: float
    cum: float                      # 有效帧的累计误差（早停时算到早停那帧为止）
    budget: float
    terminated_at: Optional[int]    # 全局帧号；None = 整块跑完没超
    status: str                     # "ok" | "over" | "unknown"

    @property
    def feasible(self) -> bool:
        """只有 ``ok`` 算过。``unknown`` 不算过 —— 量不到不等于没问题。"""
        return self.status == "ok"


@dataclass
class BlockPlan:
    """一块的判决：这块打算用哪档模式，为什么。"""

    score: BlockScore
    mode: str                       # "replay" | "mpc" | "rl"
    escalate: bool                  # 需不需要比 replay 更贵的东西
    reason: str
    blocked_by_next: bool           # 自己过了、但被下一块拖下来（两块联合窗口）

    def as_dict(self) -> dict:
        s = self.score
        return {"index": s.index, "start": s.start, "stop": s.stop,
                "n_frames": s.n_frames, "n_valid": s.n_valid,
                "e_mean": _j(s.e_mean), "e_max": _j(s.e_max), "cum": _j(s.cum),
                "budget": _j(s.budget), "terminated_at": s.terminated_at,
                "status": s.status, "mode": self.mode,
                "escalate": bool(self.escalate),
                "blocked_by_next": bool(self.blocked_by_next),
                "reason": self.reason}


def _j(x) -> Optional[float]:
    """NaN 在 JSON 里是非法的，转成 None —— 写成 0 会被读成"误差为零"。"""
    x = float(x)
    return None if not np.isfinite(x) else round(x, 6)


def split_blocks(n_frames: int, horizon: int = H_DEFAULT,
                 keep_tail: bool = True) -> List[Tuple[int, int]]:
    """``n_frames`` → ``[(start, stop), …]``，右开，固定长度 ``horizon``。

    末尾不足一块的余帧：``keep_tail=True``（默认）保留成一个短块。丢掉它等于静默
    丢帧 —— 一段 69 帧按 H=20 切，末尾那 9 帧照样要进训练集，不能不判。
    """
    if n_frames < 0:
        raise ValueError(f"n_frames 不能是负的：{n_frames}")
    if horizon < 1:
        raise ValueError(f"horizon 至少 1，收到 {horizon}")
    out = [(s, min(s + horizon, n_frames)) for s in range(0, n_frames, horizon)]
    if out and not keep_tail and out[-1][1] - out[-1][0] < horizon:
        out.pop()
    return out


def score_block(e: np.ndarray, index: int, start: int, stop: int,
                cfg: RefineConfig = RefineConfig()) -> BlockScore:
    """一块的打分 + 早停。``e`` 是整段的逐帧误差 (T,)，无效帧是 NaN。

    早停的判据是**累计**误差（论文的说法），所以从块首开始累加，第一次超过预算的那
    一帧就是 ``terminated_at``；后面的帧不再计入 ``cum``（论文里那之后的动作是不执行
    的，把它们算进来等于给一块已经判死的轨迹继续记账）。
    """
    e = np.asarray(e, dtype=np.float64)
    seg = e[start:stop]
    n = len(seg)
    ok = np.isfinite(seg)
    budget = cfg.budget(n)

    if not ok.any() or ok.mean() < cfg.min_valid_frac:
        return BlockScore(index, start, stop, n, int(ok.sum()),
                          float(np.nanmean(seg)) if ok.any() else float("nan"),
                          float(np.nanmax(seg)) if ok.any() else float("nan"),
                          float(np.nansum(seg)) if ok.any() else float("nan"),
                          budget, None, "unknown")

    csum = np.cumsum(np.where(ok, seg, 0.0))
    over = np.nonzero(csum > budget)[0]
    if len(over):
        k = int(over[0])
        return BlockScore(index, start, stop, n, int(ok.sum()),
                          float(np.nanmean(seg[:k + 1])), float(np.nanmax(seg[:k + 1])),
                          float(csum[k]), budget, start + k, "over")
    return BlockScore(index, start, stop, n, int(ok.sum()),
                      float(np.nanmean(seg)), float(np.nanmax(seg)),
                      float(csum[-1]), budget, None, "ok")


def _next_mode(cur: str) -> str:
    from web2robot.refine.modes import MODES
    i = MODES.index(cur)
    return MODES[min(i + 1, len(MODES) - 1)]


def plan_blocks(ref: np.ndarray, ach: np.ndarray,
                valid: Optional[np.ndarray] = None,
                cfg: RefineConfig = RefineConfig(),
                requested: str = "none") -> Tuple[List[BlockPlan], dict]:
    """整段的判决：切块、打分、按两块联合窗口决定每块用哪档模式。

    Parameters
    ----------
    ref, ach
        ``(T,7)`` 参考 / 执行后的物体位姿，同一个坐标系。
    valid
        ``(T,)`` bool，孪生的 trust。
    requested
        用户传的 ``--action_refine`` 值。``"none"`` 表示只出判决不许升级 ——
        这时候 ``escalate`` 照样如实标出来（这才是"判断逻辑"的意义），但
        ``mode`` 一律停在 ``replay``，并在 ``reason`` 里写明是被 ``none`` 钉住的。

    Returns
    -------
    (plans, summary)
        ``summary`` 里有整段的帧数、块数、各状态的块数、以及 ``needs_escalation``
        —— 那是给上层判"这段数据能不能直接用"的一句话结论。
    """
    err = step_errors(ref, ach, valid, cfg.weights)
    e = err["e"]
    blocks = split_blocks(len(e), cfg.horizon, cfg.keep_tail)
    scores = [score_block(e, i, s, t, cfg) for i, (s, t) in enumerate(blocks)]

    plans: List[BlockPlan] = []
    for i, sc in enumerate(scores):
        nxt = scores[i + 1] if i + 1 < len(scores) else None
        # 两块联合窗口：自己过了还得看下一块。最后一块没有下一块，只看自己 ——
        # 论文的窗口在序列末端本来就退化成一块，这里不硬造一个虚拟块出来。
        blocked = bool(sc.feasible and nxt is not None and not nxt.feasible)
        need = (not sc.feasible) or blocked

        if not need:
            mode, reason = "replay", "照抄参考轨迹即可（当前块与下一块都在预算内）"
        elif sc.status == "unknown":
            mode = "replay" if requested == "none" else _next_mode("replay")
            reason = (f"判不了：{sc.n_valid}/{sc.n_frames} 帧可信，"
                      f"低于 min_valid_frac={cfg.min_valid_frac}")
        elif blocked:
            mode = "replay" if requested == "none" else _next_mode("replay")
            reason = (f"自己在预算内，但下一块（#{nxt.index}）不在 —— 按两块联合窗口"
                      f"一起升级，防贪心")
        else:
            mode = "replay" if requested == "none" else _next_mode("replay")
            reason = (f"累计误差 {sc.cum:.4f} > 预算 {sc.budget:.4f}"
                      + (f"，第 {sc.terminated_at} 帧早停" if sc.terminated_at is not None else ""))
        if need and requested == "none":
            reason += "；--action_refine none，只出判决不升级"
        plans.append(BlockPlan(sc, mode, need, reason, blocked))

    summary = {
        "n_frames": int(len(e)),
        "n_blocks": len(plans),
        "horizon": cfg.horizon,
        "per_frame_budget": cfg.per_frame_budget,
        "lam_p": cfg.weights.lam_p,
        "lam_R": cfg.weights.lam_R,
        "n_valid_frames": int(np.isfinite(e).sum()),
        "status_counts": {k: sum(1 for p in plans if p.score.status == k)
                          for k in ("ok", "over", "unknown")},
        "n_escalate": sum(1 for p in plans if p.escalate),
        "n_blocked_by_next": sum(1 for p in plans if p.blocked_by_next),
        "needs_escalation": any(p.escalate for p in plans),
        "e_mean_all": _j(float(np.nanmean(e)) if np.isfinite(e).any() else float("nan")),
        "e_max_all": _j(float(np.nanmax(e)) if np.isfinite(e).any() else float("nan")),
        "requested": requested,
    }
    return plans, summary


def format_plans(plans: Sequence[BlockPlan]) -> str:
    """一行一块的可读表格 —— 跑完站在终端上看的就是这个。"""
    head = (f"{'blk':>3} {'frames':>11} {'valid':>7} {'e_mean':>8} "
            f"{'cum/预算':>15} {'状态':>8} {'模式':>7}  说明")
    rows = [head]
    for p in plans:
        s = p.score
        em = "   nan" if not np.isfinite(s.e_mean) else f"{s.e_mean:6.4f}"
        cu = "  nan" if not np.isfinite(s.cum) else f"{s.cum:6.3f}"
        rows.append(f"{s.index:>3} {s.start:>5}-{s.stop:<5} "
                    f"{s.n_valid:>3}/{s.n_frames:<3} {em:>8} "
                    f"{cu:>7}/{s.budget:<7.3f} {s.status:>8} {p.mode:>7}  {p.reason}")
    return "\n".join(rows)
