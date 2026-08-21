"""网格搜索式根位姿求解 —— 数学优化路线，和生成模型路线并列的第二条支路。

## 这是哪篇论文的方法

Qwen-RobotManip 技术报告（arXiv 2606.17846）§2.3 Visual Alignment 公式 (3)：

    T*_base = argmax_{T_base}  (1/|K|) Σ_{k∈K}  1[ IK(T_base⁻¹ · T_ee_k) is feasible ]

原文对 K 和候选集的说明只有两句，这里逐字抄下来，因为下面每个实现决定都是对着它做的：

    "K ⊂ {1, ..., N} is a set of representative keyframes covering the spatial extremes
     of the trajectory. Candidate base placements are generated via grid search around
     the trajectory centroid, constrained by the per-morphology kinematic reach r_max."

它为什么需要优化而不是读数：人手轨迹是 **embodiment-free** 的 —— 机器人到机器人的迁移
有个原始底座位姿可以参考，纯手部轨迹压根没有"物理底座"这个概念，所以底座摆哪只能解出来。

## 和 EgoInfinity 生成模型路线的关系：不是等价替代品

这个求解器输出 ``mode="static"`` —— **整条轨迹一个常量 T_base**。这是论文方法的定义
特征，不是待补的缺陷：它取覆盖轨迹极值的关键帧集合 K 做**一次** argmax。而
:func:`~web2robot.retarget.root_anchor.sample_best_anchor` 那条路是逐帧的（窗口估计 +
锚点混合 + 插值）。所以这两条是**两种粒度**，不是同一件事的两种算法。加这条支路的意义
是把"静态搜索"这一类方法做进同一套代码里当可选模式，好横向比较。

## 和 Ego2Robot 打分函数的差异（为什么目标函数必须可插拔）

Ego2Robot（arXiv 2608.02580 附录 A.4）的打分是
``IK 可行率 − 5.0 × |臂展利用率 − 0.65|``，显式惩罚"伸太满或缩太紧"。
本模块**只实现 Qwen 的原版**：纯 IK 可行率 argmax，不带臂展利用率惩罚项 ——
"够得到就行，姿态好不好看不管"。加不加那个惩罚项是一次独立的消融实验，
不塞进第一版；接口上留 ``score_fn`` 这个口子，换目标函数不用改本模块。

## 零上游 import

和 ``root_anchor`` 同一个规矩：IK 求解器、坐标变换全是 EgoInfinity 的东西，
本模块只拿 ``score_fn`` 这个闭包。所以它能被纯 numpy 假 callable 单测，不需要 GPU、
不需要 checkpoint、不需要 ``external/`` 在位（见 ``tests/test_root_grid.py``）。

## 一个实测发现：公式 (3) 的 argmax 经常不唯一

K 是按**位置**选的（凸包顶点），可行率因此很容易饱和：只要所有极值位置都够得到，
分数就是 100%，成千上万个候选同分。实测 ``-QALmP1nHtM_678.2_682.2`` 上两个不同朝向
的解在 K 上都是 100%，但在全部帧上一个 100% 一个 66.7% —— 差别全在没进 K 的帧上，
因为 K 完全没看**手腕朝向**（位置落在凸包内部、腕部朝向别扭的帧照样解不出来）。

原文没规定同分怎么选。本模块的 ``tie_break="plateau"`` 取同分集合的最内部点（离该
集合质心最近的成员）——仍然是精确 argmax 的成员，只是附加了一条"离可行/不可行边界
越远越稳"的准则。``n_tied`` 一并返回，好知道这个平局有多大。

## 一个必须知道的坑：workspace_center 会把整个搜索变成空操作

上游 ``cam_to_root_targets(..., workspace_center=...)`` 在给了 workspace_center 时做

    left_pos = left_pos - left_pos.mean(0) + workspace_center["left"]

**平移分量被重新居中，候选 t 完全不起作用** —— 上游 ``scripts/dev/test_fixed_root_ik.py``
里那句注释写的就是 "t is irrelevant because workspace_center re-centers position"。
所以调用方构造 ``score_fn`` 时必须传 ``workspace_center=None``。传错了不会报错，只会让
上万个候选打出一模一样的分、argmax 随便挑一个返回 —— 和 postik smoother 那次
``uniform_filter1d(size=1)`` 是同一类静默空操作。:func:`solve_root_pose_grid` 因此在
所有候选同分时**直接抛异常**，不返回一个看起来没问题的结果。

2026-08-11 新增。
"""
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence

import numpy as np


# ── 关键帧集合 K ───────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class KeyframeSet:
    """选出来的关键帧，以及它是怎么选出来的。

    ``source`` 一定要跟着结果走：凸包退化时会静默换成 range 兜底，如果不记下来，
    事后看到"只有 6 个关键帧"根本分不清是轨迹本来就贴近一条线，还是选取写错了。
    """
    indices: np.ndarray          # (|K|,) int，升序去重
    source:  str                 # "convex_hull" | "range_extremes"
    n_frames: int                # 原始 N，方便看覆盖比例


def select_extremal_keyframes(
    positions:      np.ndarray,
    max_keyframes:  Optional[int] = None,
) -> KeyframeSet:
    """取"覆盖轨迹空间极值"的那组帧 —— 对末端位置做凸包，取顶点所在的帧。

    为什么是凸包：论文只说 "covering the spatial extremes"，没给算法。凸包顶点正好
    是"往任何方向伸得最远"的那些点的精确定义，而且**无参数、确定性**，不用手调阈值 ——
    这一点比"速度峰值""等间隔采样"之类的启发式重要，因为阈值一旦引入，后面每次
    对比实验都要先解释阈值。物理直觉是：极端点都够得到，中间点大概率也够得到
    （凸包内部的点到任意底座的距离不会超过顶点的最大值）。

    Parameters
    ----------
    positions
        ``(T, 3)`` 单末端，或 ``(T, H, 3)`` 多末端（双手就是 ``H=2``）。
        多末端时**把所有手的点并在一起**算一个凸包，然后回溯到帧号：
        某一帧只要有任意一只手贡献了凸包顶点，这一帧就进 K。理由是底座要同时
        照顾两只手，"左手伸到最远"那一帧对底座的约束和右手的一样硬。
    max_keyframes
        给了就在顶点里**按到质心的距离从远到近**截断。默认 ``None`` 不截断 ——
        凸包顶点数本来就不多（几十帧的片段通常 10~30 个）。

    Returns
    -------
    KeyframeSet
        ``source="convex_hull"``；轨迹退化（点太少、共面、共线）时自动退到
        ``"range_extremes"`` —— 每个坐标轴的最小/最大值所在帧，最多 6 帧。

    Raises
    ------
    ValueError
        ``positions`` 形状不对，或 ``T == 0``。
    """
    pts, frame_of = _flatten_positions(positions)
    n_frames = int(positions.shape[0])

    idx, source = _hull_vertices(pts)
    if idx is None:
        idx, source = _range_extremes(pts), "range_extremes"

    if max_keyframes is not None and max_keyframes > 0:
        centroid = pts.mean(axis=0)
        far_first = np.argsort(-np.linalg.norm(pts[idx] - centroid, axis=1), kind="stable")
        idx = idx[far_first[:max_keyframes]]

    frames = np.unique(frame_of[idx]).astype(int)
    return KeyframeSet(indices=frames, source=source, n_frames=n_frames)


def _flatten_positions(positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(T,3)`` 或 ``(T,H,3)`` → ``(T*H, 3)`` 点集 + 每个点属于哪一帧。"""
    p = np.asarray(positions, dtype=float)
    if p.ndim == 2 and p.shape[1] == 3:
        p = p[:, None, :]
    if p.ndim != 3 or p.shape[2] != 3:
        raise ValueError(f"positions 要是 (T,3) 或 (T,H,3)，收到 {np.shape(positions)}")
    T, H, _ = p.shape
    if T == 0:
        raise ValueError("positions 是空的：T=0")
    frame_of = np.repeat(np.arange(T), H)
    return p.reshape(T * H, 3), frame_of


def _hull_vertices(pts: np.ndarray) -> tuple[Optional[np.ndarray], str]:
    """凸包顶点下标；点太少或退化（共面/共线）时返回 ``(None, ...)``。

    scipy 缺失也走同一条兜底路 —— 本模块不该因为一个可选依赖就整条不可用。
    """
    if len(pts) < 4:
        return None, "range_extremes"
    try:
        from scipy.spatial import ConvexHull, QhullError
    except ImportError:
        return None, "range_extremes"
    try:
        hull = ConvexHull(pts)
    except QhullError:
        # 共面/共线 —— 桌面上的平面轨迹很常见，不是异常输入
        return None, "range_extremes"
    return np.asarray(hull.vertices, dtype=int), "convex_hull"


def _range_extremes(pts: np.ndarray) -> np.ndarray:
    """每个坐标轴的 argmin / argmax，最多 6 个下标（去重后可能更少）。"""
    return np.unique(np.concatenate([pts.argmin(axis=0), pts.argmax(axis=0)]))


# ── 候选网格 ───────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class GridSpec:
    """候选网格的完整定义 —— 结果里必须带上，否则没法复现也没法解释。"""
    centroid:  np.ndarray        # (3,) 轨迹质心，网格中心
    r_max:     float             # 机器人最大臂展，横向搜索半径
    spacing:   float             # 横向（x/y）格距，米
    z_radius:  float             # 竖直搜索半径，米
    z_spacing: float             # 竖直格距，米
    shape:     tuple             # (nx, ny, nz)

    @property
    def n_translations(self) -> int:
        return int(np.prod(self.shape))


def build_translation_grid(
    centroid:  np.ndarray,
    r_max:     float,
    spacing:   float = 0.05,
    z_radius:  Optional[float] = None,
    z_spacing: Optional[float] = None,
) -> tuple[np.ndarray, GridSpec]:
    """以轨迹质心为中心撒候选平移，横向半径 ``r_max``，竖直单独给一个更小的半径。

    Parameters
    ----------
    centroid
        网格中心。用**全部 N 帧**的均值而不是只用 K 帧的：K 是极值点集合，
        它的均值会被伸得最远的那几帧拽偏。
    r_max
        机器人最大臂展。横向 x/y 各搜 ``[-r_max, +r_max]`` —— 论文的
        "constrained by the per-morphology kinematic reach"：底座离轨迹超过臂展就
        必然一帧都够不到，再往外撒是纯浪费。
    spacing
        横向格距，默认 5 cm。先粗跑通，要不要加密看结果。
    z_radius
        竖直半径，默认 ``0.5 * r_max``。**这是先验，不是从 URDF 读出来的约束** ——
        查过 M7 的 IK 链：链根是 ``waist_pitch_link``、模型是只有两条臂的 MJX 资产，
        里面既没有地面也没有固定骨盆，所以 URDF 层面**没有**"底座高度"这种硬约束
        可抄（论文那 15 台是螺在桌上的，天然有）。真正约束竖直方向的是别的东西：
        躯干到手的高度差本来就被臂展兜住，而且人干活时手基本在肩以下、躯干前方，
        所以竖直方向的合理区间比横向窄。给成参数，想搜满就传 ``z_radius=r_max``。
    z_spacing
        竖直格距，默认跟 ``spacing`` 一样。

    Returns
    -------
    (translations, spec)
        ``translations`` 形状 ``(M, 3)``，按 ``(x, y, z)`` C 序展开。

    Raises
    ------
    ValueError
        ``r_max``/``spacing`` 非正，或 ``centroid`` 形状不对。
    """
    c = np.asarray(centroid, dtype=float).reshape(-1)
    if c.shape != (3,):
        raise ValueError(f"centroid 要是 (3,)，收到 {np.shape(centroid)}")
    if not (r_max > 0):
        raise ValueError(f"r_max 必须为正，收到 {r_max}")
    if not (spacing > 0):
        raise ValueError(f"spacing 必须为正，收到 {spacing}")

    zr = float(0.5 * r_max if z_radius is None else z_radius)
    zs = float(spacing if z_spacing is None else z_spacing)
    if zr < 0:
        raise ValueError(f"z_radius 不能为负，收到 {z_radius}")
    if not (zs > 0):
        raise ValueError(f"z_spacing 必须为正，收到 {z_spacing}")

    axis_xy = _symmetric_axis(r_max, spacing)
    axis_z  = _symmetric_axis(zr, zs)
    gx, gy, gz = np.meshgrid(c[0] + axis_xy, c[1] + axis_xy, c[2] + axis_z, indexing="ij")
    grid = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)

    spec = GridSpec(centroid=c, r_max=float(r_max), spacing=float(spacing),
                    z_radius=zr, z_spacing=zs,
                    shape=(len(axis_xy), len(axis_xy), len(axis_z)))
    return grid, spec


def _symmetric_axis(radius: float, step: float) -> np.ndarray:
    """``[-radius, +radius]`` 上步长 ``step`` 且**一定包含 0** 的对称坐标轴。

    包含 0 是有意的：质心本身必须是候选之一，否则"最优解恰好在质心"这种情况
    要靠格点碰巧对齐才能找到。``radius < step`` 时退化成只有 ``[0.0]``。
    """
    n = int(np.floor(radius / step + 1e-9))
    return np.arange(-n, n + 1, dtype=float) * step


# ── 求解 ───────────────────────────────────────────────────────────────────────

@dataclass
class RootPoseSolution:
    """一次网格搜索的结果 + 全部诊断信息。

    ``mode`` 恒为 ``"static"``：这条支路的定义就是整条轨迹一个常量位姿。下游
    （重定向、碰撞）只认 ``(mode, R, t)`` 这个契约，不关心位姿是搜出来的还是模型出的。
    """
    mode:            str                 # 恒为 "static"
    R:               np.ndarray          # (3,3) 最优底座朝向（相机系→根系）
    t:               np.ndarray          # (3,)  最优底座平移
    ik_rate:         float               # 最优候选在 K 上的 IK 可行率
    keyframes:       KeyframeSet
    grid:            GridSpec
    n_candidates:    int                 # 网格总候选数（含被剪掉的）
    n_scored:        int                 # 真正送去打分的候选数
    best_R_index:    int                 # 最优候选用的是第几个朝向
    scores:          np.ndarray          # (n_candidates,) 未打分的是 nan
    reach_bound:     np.ndarray          # (n_candidates,) 可行率的可达上界
    n_tied:          int = 1             # 和最优同分的候选数。tie_break="plateau"
                                         # 时是 argmax 集合的**准确**大小；"first"
                                         # 时剪枝会剪掉同分候选，它退化成下界
    tie_break:       str = "plateau"     # 同分时怎么选，见 solve_root_pose_grid
    extras:          dict = field(default_factory=dict)

    def summary(self) -> str:
        kf = self.keyframes
        n_rot = max(1, self.n_candidates // max(1, self.grid.n_translations))
        return (f"static root pose: ik_rate={self.ik_rate*100:.1f}%  "
                f"t={np.round(self.t, 3).tolist()}  R#{self.best_R_index}/{n_rot}\n"
                f"  K: {len(kf.indices)}/{kf.n_frames} 帧（{kf.source}）\n"
                f"  网格: {self.grid.shape} × {n_rot} 朝向 = {self.n_candidates} 候选，"
                f"实打 {self.n_scored} 个（剪掉 {self.n_candidates - self.n_scored}）\n"
                f"  同分候选 {self.n_tied} 个（tie_break={self.tie_break}）")


def solve_root_pose_grid(
    ee_positions:  np.ndarray,
    score_fn:      Callable[[np.ndarray, np.ndarray], np.ndarray],
    r_max:         float,
    R_candidates:  Optional[Sequence[np.ndarray]] = None,
    spacing:       float = 0.05,
    z_radius:      Optional[float] = None,
    z_spacing:     Optional[float] = None,
    reach_frac:    float = 1.0,
    min_dist:      float = 0.0,
    max_keyframes: Optional[int] = None,
    chunk:         int = 256,
    tie_break:     str = "plateau",
    exhaustive:    bool = False,
    log:           Callable[[str], None] = print,
) -> RootPoseSolution:
    """公式 (3) 的实现：网格搜索使关键帧 IK 可行率最大的静态底座位姿。

    Parameters
    ----------
    ee_positions
        ``(T, H, 3)`` 相机系下的末端目标位置（双手 ``H=2``），或 ``(T, 3)``。
        只吃位置：关键帧选取和可达上界都只跟位置有关，朝向在 ``score_fn`` 里用。
    score_fn
        ``score_fn(R_batch, t_batch) -> (B,)`` 的 IK 可行率，``R_batch`` 形状
        ``(B,3,3)``、``t_batch`` 形状 ``(B,3)``。**必须只在关键帧 K 上算**，
        并且复用现有那套 IK 和 ``converged`` 判据（口径一致才能横向比）。
        调用方那侧记得 ``workspace_center=None``，见模块 docstring。
    r_max
        机器人最大臂展，米。
    R_candidates
        候选朝向列表，默认 ``[I]`` —— 只搜平移。论文原文的候选集也只说"围绕质心
        撒网格"，没有朝向枚举（Ego2Robot 那篇才有 pitch×yaw×roll 45 种）。M7 的
        躯干朝向不是自由参数（人形站姿基本竖直），所以默认交给调用方给一个先验朝向，
        要搜就自己传一串进来。
    reach_frac, min_dist
        可达上界用的两个门槛：目标点到底座的距离超过 ``reach_frac * r_max`` 或
        小于 ``min_dist`` 的 (帧, 手) 对**必然** IK 不可行。默认 ``1.0 / 0.0``
        即只用"超过臂展必然不可行"这一个物理事实，不额外收紧。
    max_keyframes
        传给 :func:`select_extremal_keyframes`。
    chunk
        每次送进 ``score_fn`` 的候选数。
    tie_break
        公式 (3) 的 argmax 是一个**集合**，不是一个点 —— 原文没说同分时选哪个。
        实测这个集合经常很大：K 只覆盖位置极值，一旦所有极值都够得到，可行率就
        饱和在 100%，成千上万个候选同分。选哪个不影响目标函数的值，但**很影响
        没进 K 的那些帧**（K 是按位置选的，完全没看手腕朝向；位置在凸包内部但
        腕部朝向别扭的帧照样可能解不出来）。

        * ``"plateau"``（默认）—— 取同分集合里离该集合**质心最近**的候选，也就是
          可行域"高原"的最内部那点。理由是内部点离可行/不可行的边界最远，对网格
          离散化和对没进 K 的帧都更稳。
        * ``"first"`` —— 取上界排序里最先撞到最高分的那个（旧行为，可复现但任意）。

        两者都返回**同分集合里的成员**，所以都是公式 (3) 的精确 argmax；差别只在
        用哪条附加准则打破平局。想改成"按臂展利用率打破平局"（Ego2Robot 那种）
        请走 ``score_fn``，不要在这里加惩罚项。

        它还决定剪枝有多狠，这点要知道再选：``"plateau"`` 必须留住同分候选，
        门槛只能是 ``>=``，可行率一饱和就基本退化成穷举（实测 M7 上 40 万候选
        跑十几分钟）；``"first"`` 用 ``>``，实测能少打一半多的分（``-QALmP1nHtM``
        上 210912 → 97244 个候选，42 万总候选），但
        ``n_tied`` 退化成"打过分的候选里的同分数"，只是个下界，不能当 argmax
        集合的大小报出去。
    exhaustive
        ``True`` 时关掉上界剪枝，把每个候选都打一遍分。剪枝是**可采纳的**
        （上界 ≥ 真实可行率，所以剪掉的不可能是最优解），这个开关只为验证
        "剪枝版和穷举版结果相同"，单测里钉的就是这一条。
    log
        进度输出。

    Returns
    -------
    RootPoseSolution

    Raises
    ------
    ValueError
        参数非法，或 ``score_fn`` 返回的形状不对。
    RuntimeError
        所有被打分的候选**分数完全相同**。这几乎一定是调用方把
        ``workspace_center`` 传进去了、平移被重新居中、整个搜索退化成空操作。
        与其返回一个看起来没问题的位姿，不如当场炸。
    """
    if not (0.0 < reach_frac <= 1.0):
        raise ValueError(f"reach_frac 要在 (0,1] 内，收到 {reach_frac}")
    if min_dist < 0:
        raise ValueError(f"min_dist 不能为负，收到 {min_dist}")
    if chunk < 1:
        raise ValueError(f"chunk 至少为 1，收到 {chunk}")
    if tie_break not in ("plateau", "first"):
        raise ValueError(f"tie_break 只能是 'plateau'/'first'，收到 {tie_break!r}")

    pts_all, _ = _flatten_positions(ee_positions)
    kf = select_extremal_keyframes(ee_positions, max_keyframes=max_keyframes)
    kf_pts, _ = _flatten_positions(np.asarray(ee_positions, dtype=float)[kf.indices])

    centroid = pts_all.mean(axis=0)
    grid, spec = build_translation_grid(centroid, r_max, spacing, z_radius, z_spacing)

    Rs = np.stack([np.eye(3)] if R_candidates is None
                  else [np.asarray(R, dtype=float).reshape(3, 3) for R in R_candidates])

    # 可达上界：‖R^T(p−t)‖ = ‖p−t‖，跟朝向无关，所以每个平移只算一次。
    d = np.linalg.norm(kf_pts[None, :, :] - grid[:, None, :], axis=2)   # (M, |K|*H)
    reachable = (d <= reach_frac * r_max) & (d >= min_dist)
    bound_t = reachable.mean(axis=1)                                    # (M,)

    n_cand = len(grid) * len(Rs)
    bound = np.repeat(bound_t, len(Rs))
    cand_t = np.repeat(grid, len(Rs), axis=0)
    cand_R = np.tile(np.arange(len(Rs)), len(grid))

    log(f"  K={len(kf.indices)}/{kf.n_frames} 帧（{kf.source}）  "
        f"质心={np.round(centroid, 3).tolist()}  r_max={r_max:.3f}")
    log(f"  网格 {spec.shape} × {len(Rs)} 朝向 = {n_cand} 候选；"
        f"可达上界 >0 的 {int((bound > 0).sum())} 个")

    scores = np.full(n_cand, np.nan)
    order  = np.argsort(-bound, kind="stable")     # 上界高的先打，剪枝才剪得掉东西
    best_i, best_s, n_scored = -1, -np.inf, 0

    # 剪枝门槛的强弱由 tie_break 决定，这不是可有可无的开关，是一笔实打实的账：
    #
    # * ``plateau`` 要在同分集合里挑最内部点，**同分候选就不能被剪掉**，门槛只能用
    #   ``>=``。代价是目标函数一饱和（可行率打到 1.0，容易的片段上很常见）时，
    #   所有上界 1.0 的候选都得真打一遍 —— 搜索基本退化成穷举。
    # * ``first`` 不需要同分集合，门槛用 ``>``：上界**等于**当前最高分的候选顶多打平，
    #   打平也不换解（下面选 top 用严格 >），所以剪掉不改变返回的位姿。实测省掉一半
    #   多的打分（``-QALmP1nHtM`` 上 210912 → 97244）。这时 ``n_tied`` 只是**打过分的
    #   候选里**的同分数（那条片段上报 1，真实是 292），是个下界，不能当成 argmax
    #   集合的大小报出去 —— 也就没法做高原 tie_break，而这条片段上正是它决定了
    #   全轨迹 100.0% 还是 66.7%。
    #
    # 两条路的 ``R``/``t``/``ik_rate`` 在唯一最优时完全一致（测试钉着）。
    tie_cmp = np.greater_equal if tie_break == "plateau" else np.greater

    for start in range(0, n_cand, chunk):
        sel = order[start:start + chunk]
        if not exhaustive and not tie_cmp(bound[sel[0]], best_s):
            break                                   # 后面上界只会更低，全剪掉
        if not exhaustive:
            sel = sel[tie_cmp(bound[sel], best_s)]
            if len(sel) == 0:
                continue
        s = np.asarray(score_fn(Rs[cand_R[sel]], cand_t[sel]), dtype=float).reshape(-1)
        if s.shape != (len(sel),):
            raise ValueError(f"score_fn 该返回 ({len(sel)},)，收到 {s.shape}")
        scores[sel] = s
        n_scored += len(sel)
        top = int(np.argmax(s))
        # 严格 > ：并列时留**上界排序里更靠前**的那个，结果因此是确定的
        if s[top] > best_s:
            best_s, best_i = float(s[top]), int(sel[top])

    if best_i < 0:
        raise ValueError("一个候选都没打上分：检查 r_max / spacing / reach_frac")
    _assert_scores_vary(scores, bound)

    tied = np.flatnonzero(scores == best_s)
    if tie_break == "plateau" and len(tied) > 1:
        # 高原质心最近点。同距时取下标最小的（np.argmin 的行为），所以确定。
        centre = cand_t[tied].mean(axis=0)
        best_i = int(tied[int(np.argmin(np.linalg.norm(cand_t[tied] - centre, axis=1)))])

    log(f"  最优: ik_rate={best_s*100:.1f}%  t={np.round(cand_t[best_i], 3).tolist()}  "
        f"（实打 {n_scored}/{n_cand}，同分 {len(tied)}）")

    return RootPoseSolution(
        mode="static", R=Rs[cand_R[best_i]].copy(), t=cand_t[best_i].copy(),
        ik_rate=best_s, keyframes=kf, grid=spec,
        n_candidates=n_cand, n_scored=n_scored, best_R_index=int(cand_R[best_i]),
        scores=scores, reach_bound=bound,
        n_tied=int(len(tied)), tie_break=tie_break,
        extras={"candidate_t": cand_t, "candidate_R_index": cand_R},
    )


def _assert_scores_vary(scores: np.ndarray, bound: np.ndarray) -> None:
    """所有打过分的候选同分 → 炸。见模块 docstring 里的 workspace_center 坑。

    这里有一个必须避开的**假警报**：可行率是会饱和的。K 只覆盖位置极值，容易的
    片段上"够得到全部极值"的候选成千上万，分数全是 1.0。剪枝又是按可达上界从高到
    低打的，所以完全可能出现"打过分的候选恰好全是 1.0"这种**正常**情况。

    区分办法看**可达上界有没有跟着变**：

    * workspace_center 的坑 —— 候选平移被抹掉，分数与 t 无关，但上界是自己算的、
      照样随 t 变化。所以"分数不变而上界在变" = 真出问题了。
    * 正常饱和 —— 打过分的候选就是上界最高那一批（都=1.0），上界本身也不怎么变。

    所以只有"分数全同 **且** 上界有明显差异"才判定为空操作。
    """
    got_mask = ~np.isnan(scores)
    got = scores[got_mask]
    if len(got) <= 1 or np.ptp(got) != 0.0:
        return
    if np.ptp(bound[got_mask]) <= 0.1:      # 上界也一样 → 是饱和，不是空操作
        return
    raise RuntimeError(
        f"{len(got)} 个候选打出完全相同的分（{got[0]:.6f}），而它们的可达上界"
        f"跨了 {np.ptp(bound[got_mask]):.3f} —— 底座平移对目标点没有产生任何影响。"
        f"最常见的原因是 score_fn 里给 cam_to_root_targets 传了 workspace_center，"
        f"位置被 `pos - pos.mean(0) + center` 重新居中，候选 t 被抹掉、整个网格搜索"
        f"退化成空操作。搜索时必须传 workspace_center=None。"
    )


# ── 把上游 IK 包成 score_fn ────────────────────────────────────────────────────

def make_keyframe_scorer(
    kf_left:       np.ndarray,
    kf_right:      np.ndarray,
    to_root_fn:    Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], tuple],
    converged_fn:  Callable[[str, np.ndarray, np.ndarray], np.ndarray],
) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """把"上游的坐标变换 + 上游的 IK"包成 :func:`solve_root_pose_grid` 要的 ``score_fn``。

    这层胶水放在这里而不是各个调用点，是因为它有一个**非平凡的批处理约定**：候选
    位姿是常量，但上游 ``cam_to_root_targets`` 只吃逐帧数组，所以每个候选都要先把
    ``(R, t)`` 广播成 ``(|K|, ·)``，再把 B 个候选的 B×|K| 条目标拼成一批送进 IK。
    这套批法是照着上游 ``select_best_anchor`` 抄的（它给 K 个聚类中心打分时就这么干）。
    逐候选串行调 IK 会慢两个数量级，所以这不是"顺手优化"，是能不能跑的问题。

    两个 callable 就是这一层保持**零上游 import** 的方式（同
    :func:`~web2robot.retarget.root_anchor.sample_best_anchor` 的做法）：

    Parameters
    ----------
    kf_left, kf_right
        ``(|K|, 7)`` 关键帧上的手腕位姿（位置 + 四元数），相机系。格式由
        ``to_root_fn`` 决定，本函数只做切片和拼接，不解释内容。
    to_root_fn
        ``(left, right, R_per_frame, t_per_frame) -> (lp, lq, rp, rq)``。调用方
        通常是 ``functools.partial(cam_to_root_targets, workspace_center=None)``
        —— **必须把 workspace_center 绑成 None**，否则位置被重新居中、候选平移
        失效（见模块 docstring）。
    converged_fn
        ``(side, pos, quat) -> (N,) bool``，``side`` 是 ``"left"``/``"right"``。
        调用方把上游 ``WristIK.solve_batch(...)[1]["converged"]`` 包进来即可 ——
        判据必须是上游那个，不要在这里另立标准。

    Returns
    -------
    score_fn
        ``score_fn(R_batch, t_batch) -> (B,)``，值是
        ``(左臂可行率 + 右臂可行率) / 2``，和上游 ``select_best_anchor`` 里
        ``ik_rates = (conv_l.mean(axis=1) + conv_r.mean(axis=1)) / 2`` 同一个口径。
    """
    kf_l = np.ascontiguousarray(kf_left)
    kf_r = np.ascontiguousarray(kf_right)
    K = len(kf_l)
    if K == 0:
        raise ValueError("关键帧集合是空的")
    if len(kf_r) != K:
        raise ValueError(f"左右关键帧数不一致：{K} vs {len(kf_r)}")

    def score_fn(R_batch: np.ndarray, t_batch: np.ndarray) -> np.ndarray:
        B = len(t_batch)
        lp, lq, rp, rq = [], [], [], []
        for b in range(B):
            R_pf = np.broadcast_to(R_batch[b], (K, 3, 3))
            t_pf = np.broadcast_to(t_batch[b], (K, 3))
            a, aq, c, cq = to_root_fn(kf_l, kf_r, R_pf, t_pf)
            lp.append(a); lq.append(aq); rp.append(c); rq.append(cq)
        conv_l = np.asarray(converged_fn("left", np.concatenate(lp), np.concatenate(lq)))
        conv_r = np.asarray(converged_fn("right", np.concatenate(rp), np.concatenate(rq)))
        for name, conv in (("left", conv_l), ("right", conv_r)):
            if conv.shape != (B * K,):
                raise ValueError(f"converged_fn({name!r}) 该返回 ({B*K},)，收到 {conv.shape}")
        return (conv_l.reshape(B, K).mean(axis=1)
                + conv_r.reshape(B, K).mean(axis=1)) / 2

    return score_fn


# ── 朝向候选：竖直轴由重力钉死，偏航一圈枚举 ───────────────────────────────────

def gravity_yaw_candidates(
    gravity_up:       np.ndarray,
    left_positions:   np.ndarray,
    right_positions:  np.ndarray,
    n_yaw:            int = 12,
) -> list:
    """training-free 的朝向候选：三个自由度分开处理，只搜真正没有先验的那一个。

    论文只给了**平移**网格，朝向怎么定它没说（Ego2Robot 那篇才枚举
    pitch×yaw×roll 共 45 种）。但如果朝向借生成模型解出来的锚点，这条"数学优化
    路线"就还是要 checkpoint，和"与训练模型并列可切换"的定位不符。所以：

    1. **竖直轴（2 个自由度）—— 由数据钉死。** 片段元数据自带 ``gravity_up``
       （上游 ``clip_io`` 里的 ``g_cam = -gravity_up``），人形机器人躯干站姿竖直，
       所以躯干 z 轴直接取 ``gravity_up`` 的单位向量。实测这一轴和生成模型解出的
       锚点第三列几乎重合（fill_jar 上差 <2°），不需要搜。
    2. **偏航（1 个自由度）—— 全圆枚举。** 手在身体前方哪个角度、人有没有正对着
       活儿，都没有可靠先验，所以按 ``360/n_yaw`` 步长把一圈撒满，让 IK 可行率
       自己挑。这就是 Ego2Robot 枚举朝向的思路，只是靠重力先砍掉两个自由度之后
       枚举量小得多。**这一步是必要的**：早期版本用"相机原点指向双手质心"当前向，
       在官方片段上偏了 150°+（第一视角相机不在躯干上，这个启发式站不住），
       IK 可行率因此掉了一半。
    3. 枚举的**零点**取"左手均值 − 右手均值的水平分量当躯干左轴"——躯干的左手确实
       在左边。零点选哪儿在全圆枚举下不影响能不能找到好解，只影响采样落点；取一个
       有物理含义的零点让 ``best_R_index`` 变得可读：它直接是"最优朝向偏离人正对
       自己双手多少度"。

    ``n_yaw=1`` 时退化成只用这个零点，可以当消融项（启发式朝向 vs 枚举朝向）。

    Parameters
    ----------
    gravity_up
        ``(3,)`` 相机系下的"上"方向。注意上游存的是 ``g_cam = -gravity_up``，
        传 ``g_cam`` 进来要先取负。
    left_positions, right_positions
        ``(T, 3)`` 或 ``(T, ≥3)``（多余的列当四元数忽略）的双手轨迹，相机系。
        只用它们的均值定零点。
    n_yaw
        偏航候选数，步长 ``360/n_yaw`` 度。

    Returns
    -------
    list of (3,3)
        每个矩阵的**列**是躯干三轴（前 / 左 / 上）在相机系下的表示 —— 上游
        ``cam_to_root_targets`` 算的是 ``R^T (p_cam − t)``，所以列向量就是这个约定。
        躯干系本身是 +x 前 / +y 左 / +z 上：由 FK 实测（左手在 +y、抬臂往 +z、
        ``start_config`` 伸手往 +x），见 ``scripts/dev/measure_m7_torso_axes.py``。

    Raises
    ------
    ValueError
        ``n_yaw < 1``，或 ``gravity_up`` 形状/模长非法。
    """
    if n_yaw < 1:
        raise ValueError(f"n_yaw 至少为 1，收到 {n_yaw}")
    up = np.asarray(gravity_up, dtype=float).reshape(-1)
    if up.shape != (3,):
        raise ValueError(f"gravity_up 要是 (3,)，收到 {np.shape(gravity_up)}")
    n_up = np.linalg.norm(up)
    if n_up < 1e-9:
        raise ValueError("gravity_up 模长为 0，定不出竖直轴")
    up = up / n_up

    lm = np.asarray(left_positions, dtype=float)[:, :3].mean(axis=0)
    rm = np.asarray(right_positions, dtype=float)[:, :3].mean(axis=0)
    sep = lm - rm
    sep = sep - up * float(sep @ up)              # 只要水平分量
    if np.linalg.norm(sep) < 1e-6:                # 双手轨迹重合，零点随便取个正交方向
        sep = np.cross(up, [1.0, 0.0, 0.0])
        if np.linalg.norm(sep) < 1e-6:
            sep = np.cross(up, [0.0, 1.0, 0.0])
    left0 = sep / np.linalg.norm(sep)
    fwd0 = np.cross(left0, up)                    # 右手系：x = y × z
    fwd0 /= np.linalg.norm(fwd0)

    out = []
    for a in np.arange(n_yaw) * (2 * np.pi / n_yaw):
        x = fwd0 * np.cos(a) + np.cross(up, fwd0) * np.sin(a)     # 绕 up 转 a
        x /= np.linalg.norm(x)
        y = np.cross(up, x)
        out.append(np.stack([x, y, up], axis=1))                  # 列 = 躯干三轴
    return out


# ── 臂展 r_max：论文里的 per-morphology kinematic reach ────────────────────────

def estimate_reach(
    fk_fn:      Callable[[np.ndarray], np.ndarray],
    limits:     np.ndarray,
    n_random:   int = 200_000,
    seed:       int = 0,
    batch:      int = 65_536,
) -> float:
    """量出机器人末端离链根最远能到多远 —— 网格的横向搜索半径。

    论文说候选集 "constrained by the per-morphology kinematic reach r_max"，但没给
    r_max 怎么来。写死一张"机器人→臂展"的表是最脆的做法（换机器人就得记得改），
    所以这里从**运动学链本身**量：在关节限位内采样构型，跑 FK，取 ‖p_ee‖ 的最大值。

    只采限位角点是**不够**的：M7 实测限位角点只到 0.7176 m，随机采样能到 1.0067 m
    （臂展最大的构型在限位内部，不在角点上）。所以两种都采，取大。

    低估比高估危险：高估只是多撒些必然不可行的候选（会被可达上界剪掉），低估会把
    真正的最优解排除在网格外。所以采样量给得大，且**只往大的方向取**。

    Parameters
    ----------
    fk_fn
        ``fk_fn(Q) -> (N, 3)``，``Q`` 形状 ``(N, n_dof)``，返回末端在**链根系**下的
        位置。零上游 import 就靠这个口子：调用方把 ``WristIK._fk`` 包一层传进来。
    limits
        ``(n_dof, 2)`` 关节下界 / 上界。
    n_random
        随机采样构型数。
    seed
        随机种子 —— 这个数会进结果元数据，必须可复现。
    batch
        一次送进 ``fk_fn`` 的构型数（显存/内存上限）。

    Returns
    -------
    float
        ``max ‖p_ee‖``，米。

    Raises
    ------
    ValueError
        ``limits`` 形状不对，或上下界反了。
    """
    lim = np.asarray(limits, dtype=float)
    if lim.ndim != 2 or lim.shape[1] != 2:
        raise ValueError(f"limits 要是 (n_dof, 2)，收到 {np.shape(limits)}")
    if np.any(lim[:, 1] < lim[:, 0]):
        raise ValueError("limits 里有上界小于下界的关节")
    n_dof = len(lim)

    Qs = []
    if n_dof <= 12:      # 2^12 = 4096，再多角点枚举就爆了（也没必要）
        corners = np.array(np.meshgrid(*[lim[i] for i in range(n_dof)], indexing="ij"))
        Qs.append(corners.reshape(n_dof, -1).T)
    rng = np.random.default_rng(seed)
    Qs.append(rng.uniform(lim[:, 0], lim[:, 1], size=(int(n_random), n_dof)))
    Qs.append(np.zeros((1, n_dof)))               # 零位，交叉验证用
    Q = np.concatenate(Qs).astype(np.float32)

    r_max = 0.0
    for start in range(0, len(Q), batch):
        p = np.asarray(fk_fn(Q[start:start + batch]), dtype=float)
        if p.ndim != 2 or p.shape[1] != 3:
            raise ValueError(f"fk_fn 该返回 (N,3)，收到 {p.shape}")
        r_max = max(r_max, float(np.linalg.norm(p, axis=1).max()))
    return r_max


__all__ = ["KeyframeSet", "GridSpec", "RootPoseSolution",
           "select_extremal_keyframes", "build_translation_grid",
           "solve_root_pose_grid", "make_keyframe_scorer",
           "gravity_yaw_candidates", "estimate_reach"]
