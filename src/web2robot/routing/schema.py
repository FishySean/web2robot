"""路由标签词汇表 —— 流水线第②步（视角与运动分类）的取值定义。

为什么放在 routing/ 而不是 quality/：这两个枚举是"选哪条技术路线"的
判据，第①步只是尽早把它们**测**出来。质检的 ClipReport 引用它们
（quality → routing 的单向依赖），第②步的 labels.suggest 也用它们。
反过来把它们留在 quality/schema.py 会让 routing 依赖 quality，
方向是错的 —— 路由不该依赖质检。
"""
from enum import Enum


class ViewClass(str, Enum):
    THIRD_PERSON_BODY = "third_person_body"
    """Both wrists + both elbows + torso visible: the root-frame estimator has
    what it needs. Eligible for the EgoInfinity route."""

    HANDS_ONLY = "hands_only"
    """Hands (and maybe forearms) but no torso. Cannot be separated from
    egocentric footage by body framing alone -- step 2 decides. NOT a rejection:
    first-person video is a valid route, just a different one."""

    NO_STABLE_HANDS = "no_stable_hands"
    """One hand poking in from an edge, or two hands present in a minority of
    frames. Nothing downstream can use this. Decided by the HAND DETECTOR --
    measured 0.42/0.72/1.00 on two-hand footage vs 0.00 on one-hand and 0.00 on
    no-hand controls (after duplicate-box merging)."""

    UNKNOWN = "unknown"
    """The hands-only boundary was not measurable (hand detector unavailable).
    Never a rejection."""


class CameraMotion(str, Enum):
    STATIC = "static"      # -> WiLoR + MoGe (no parallax for SLAM)
    MOVING = "moving"      # -> HaWoR (real translation available)
    UNKNOWN = "unknown"
