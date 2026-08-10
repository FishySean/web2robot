"""MoGe 侧的取数：从单目场景深度里取出手部关键点的**度量**深度。

MoGe 给的是整幅图的深度/点云，手在哪它不知道 —— 手的位置由 WiLoR 的 2D 关键点给。
所以这个模块全是"在给定像素处取深度"这一类操作，零模型依赖、纯 numpy，MoGe 的
``infer`` 由调用方注入（见 :mod:`web2robot.perception.wilor`）。

## 两种取法，不是一种

它们不是新旧关系，是**两个不同的历史用法**，产出的数不一样，所以都得留着：

- :func:`sample_pointmap` 走 ``infer()["points"]``（(H,W,3) 度量点云），直接取三维点。
  ``step3_perception.py`` 用这条。
- :func:`sample_depth` + :func:`unproject` 走 ``infer()["depth"]``（(H,W) 深度）再用
  **外部给的内参**反投影。``step16b_wilor_moge_abf12.py`` 用这条 —— HO-3D 评测时手上有
  真值 ``camMat``，用真内参比用 MoGe 自己那套更公平。

两者的像素取整方式**故意不同**，因为原脚本就不同：`sample_pointmap` 用 ``round``，
`sample_depth` 用 ``int()`` 截断。差半个像素，但迁移的判据是逐位一致，所以照抄。

还有一处更要紧的差异：`sample_pointmap` **不** clip 中心像素，所以完全出画的关键点会
得到空 patch → NaN；`sample_depth` 会把中心 clip 进画内，永远返回一个值（出画时那是
边缘像素的深度，等于编数）。前者的行为更可取，但两条都照原样留着并各自钉了测试。

## 为什么要取 patch 的中位而不是取那一个像素

手边缘的像素常落在手/背景的深度断层上，单点取到背景就会得到一个差几十厘米的深度。
7×7 取中位是原脚本的做法，也确实是对的：中位对"一半像素落在背景上"这件事免疫，
均值不免疫。

## 全局尺度锚

:func:`scene_depth_anchor` 是 ``step3d_wilor3d.py`` 的做法：抽 6 帧算场景深度中位的中位，
再用它去缩放 WiLoR 自己那套非度量的 3D。**这是个很粗的锚** —— 它假设"场景中位深度"
和"手的深度"成固定比例，整段只给一个数。HO-3D 上量出来的代价就是
`evidence/depth_benchmark_ho3d/`：ABF12 深度误差 11 cm，而且深度是**反相关**的。
更直接的一次是 ABF12 前 30 帧：这个锚把整只手缩到骨长 0.45 cm（真手 2~4 cm），
也就是差约 6.5 倍 —— 手比场景中位深度近得多的时候就会这样。详见
:mod:`web2robot.perception.wilor` 里那张表。
"""
import numpy as np

#: patch 半径，7×7。原脚本写死 3，这里给默认值但允许改
DEFAULT_PATCH_R = 3


def sample_pointmap(pointmap, uv, r=DEFAULT_PATCH_R):
    """在 ``uv`` 处取 MoGe 点云的中位三维点。取不到返回 ``[nan,nan,nan]``。

    ``step3_perception.py::sample_pts`` 的逐字行为：``round`` 取整、patch 双侧都 clip、
    先滤掉任何一维非有限的点再取中位。
    """
    pointmap = np.asarray(pointmap)
    if pointmap.ndim != 3 or pointmap.shape[2] != 3:
        raise ValueError(f"pointmap 得是 (H,W,3)，给的是 {pointmap.shape}")
    H, W = pointmap.shape[:2]
    x, y = int(round(float(uv[0]))), int(round(float(uv[1])))
    x0, x1 = max(0, x - r), min(W, x + r + 1)
    y0, y1 = max(0, y - r), min(H, y + r + 1)
    patch = pointmap[y0:y1, x0:x1].reshape(-1, 3)
    patch = patch[np.isfinite(patch).all(1)]
    return np.nanmedian(patch, 0) if len(patch) else np.array([np.nan] * 3)


def sample_depth(depth, uv, r=DEFAULT_PATCH_R):
    """在 ``uv`` 处取 MoGe 深度图的中位深度（标量）。取不到返回 ``nan``。

    ``step16b_wilor_moge_abf12.py`` 的逐字行为，和上面那个有三处**故意保留**的差异：
    先 clip 到 ``[0, W-1]`` 再 ``int()`` 截断（不是 round）、patch 上界不 clip
    （靠 numpy 切片自己截）、返回标量。
    """
    depth = np.asarray(depth)
    if depth.ndim != 2:
        raise ValueError(f"depth 得是 (H,W)，给的是 {depth.shape}")
    H, W = depth.shape
    xi = int(np.clip(float(uv[0]), 0, W - 1))
    yi = int(np.clip(float(uv[1]), 0, H - 1))
    patch = depth[max(0, yi - r):yi + r + 1, max(0, xi - r):xi + r + 1]
    finite = patch[np.isfinite(patch)]
    return float(np.nanmedian(finite)) if finite.size else float("nan")


def pixel_index(uv, shape):
    """``sample_depth`` 用的那个整数像素坐标，单独暴露出来。

    反投影必须用**取整后**的坐标，不能用原始亚像素坐标 —— 深度是在整数像素处取的，
    用亚像素坐标反投影会得到一个和深度不对应的 XY。原脚本是对的（它用了 ``xi,yi``），
    这里把它拎成一个函数，免得调用方各写一遍写歪。``shape`` 是 ``(H, W)``。
    """
    H, W = shape
    return (int(np.clip(float(uv[0]), 0, W - 1)), int(np.clip(float(uv[1]), 0, H - 1)))


def unproject(pixel, depth_m, K):
    """整数像素 + 深度 → 相机系三维点（针孔，z 朝前，米）。

    ``K`` 是 3×3 内参。深度是 nan 时整个点是 nan，不静默变成 0。
    """
    K = np.asarray(K, float)
    if K.shape != (3, 3):
        raise ValueError(f"K 得是 3x3，给的是 {K.shape}")
    if not np.isfinite(depth_m):
        return np.array([np.nan] * 3)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    xi, yi = pixel
    return np.array([(xi - cx) / fx * depth_m, (yi - cy) / fy * depth_m, depth_m])


def scene_depth_anchor(depth_maps):
    """一组深度图 → 一个全局场景深度（中位的中位），``step3d_wilor3d.py`` 的做法。

    **这是个很粗的锚**，代价量在 `evidence/depth_benchmark_ho3d/`，用之前先看那份。
    """
    meds = [np.nanmedian(np.asarray(d)) for d in depth_maps]
    meds = [m for m in meds if np.isfinite(m)]
    if not meds:
        raise ValueError("所有深度图都取不到有限中位值 —— MoGe 这一路整段失败了")
    return float(np.nanmedian(meds))


def anchor_frame_indices(n_frames, n_samples=6):
    """抽哪几帧去算全局锚。``step3d`` 的 ``frames[::N//6 or 1][:6]``，逐字保留。"""
    if n_frames <= 0:
        raise ValueError("n_frames 得 > 0")
    step = n_frames // n_samples or 1
    return list(range(0, n_frames, step))[:n_samples]
