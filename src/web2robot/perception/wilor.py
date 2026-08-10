"""WiLoR 前端：相机固定（或几乎不动）的片段走这条。

和 HaWoR 那条的分工写在 `perception/__init__.py`：HaWoR 要相机有视差、背景有纹理、
帧数够，满足时深度准；不满足就崩溃或退化，那时候只能走这条。WiLoR **从不崩溃，只优雅
降级** —— 它是逐帧检测，跟不上就少检出几帧，不会像 SLAM 那样整段失败。

代价是 WiLoR 自己没有度量深度：``pred_cam_t_full`` 实测抖到 210 cm/s、量级 ~13 m，
非度量。所以深度必须外挂，这就是下面两条策略的由来。

## 两条深度策略，选哪条不是这里决定的

| | `DEPTH_POINTMAP` | `DEPTH_GLOBAL_SCALE` |
|---|---|---|
| 手形从哪来 | MoGe 点云（逐关节独立取） | WiLoR 原生 3D（手形自洽） |
| 深度从哪来 | MoGe，**逐帧逐关节** | MoGe，**整段一个数** |
| 出处 | `step3_perception.py` / `step16b` | `step3d_wilor3d.py` |
| 已量到的代价 | ABF12 深度误差 11 cm、反相关 | 手腕深度变化是 WiLoR 的，非度量 |

两条都有明确的短板，而且**短板在不同的地方**。ABF12 前 30 帧上量过一次，两条各错一半
（``scripts/dev/viz_wilor_depth_modes.py``，图在 ``outputs/viz/wilor_depth_modes.mp4``）：

| | 骨长均值 | 骨长逐帧变异 | 病在哪 |
|---|---|---|---|
| 真手 | 2~4 cm | ~0 | —— |
| `DEPTH_POINTMAP` | 2.94 cm ✓ | **5.7%** ✗ | 尺度对，**手形被深度噪声撕开** |
| `DEPTH_GLOBAL_SCALE` | **0.45 cm** ✗ | 0.5% ✓ | 手形对，**整只手缩小了约 6.5 倍** |

所以"开合"这个数在两条策略下完全不可比：pointmap 报 8~15 cm 是**手形抖**抖出来的
（真手张不到 15 cm），global-scale 报 1.4 cm 是**整只手被缩小**的结果。两个数都不能直接用。

那个 6.5 倍不是常数，是这段视频上的：全局尺度 = 场景深度中位 / WiLoR 手腕深度中位，
手离相机比场景中位近得多时就会缩得厉害。换一段视频缩放倍数就变，所以
**`DEPTH_GLOBAL_SCALE` 出来的绝对尺寸整段都不可信**，只有形状和相对变化可信。

这两条是**迁移过来的既有实现**，不是推荐方案。真正该做的大概是杂交
（WiLoR 的手形 + MoGe 的逐帧手腕深度锚 —— 正好各取上表里对的那一半），但那是**新设计**，
得单独立项、单独量，不能塞进一次迁移里假装是搬代码。所以这里两条都如实留着，各自标好代价。

## 注入式依赖

``predict_fn``（WiLoR）和 ``infer_fn``（MoGe）由调用方传进来，模块层零第三方 import。
理由和 `perception/hawor.py` 一样：这样单测不需要 GPU、不需要 checkpoint。
"""
import numpy as np

from web2robot.perception import moge as MG
from web2robot.perception.to_clip import (
    HAND_LEFT, HAND_RIGHT, MANO_INDEX_TIP, MANO_THUMB_TIP, MAX_HANDS, N_JOINTS,
    empty_joints,
)

DEPTH_POINTMAP = "pointmap"
DEPTH_GLOBAL_SCALE = "global-scale"
DEPTH_MODES = (DEPTH_POINTMAP, DEPTH_GLOBAL_SCALE)

#: WiLoR 判左右手的阈值。原脚本四处都写 0.5，这里收成一个常量
IS_RIGHT_THRESHOLD = 0.5


def hand_slot(det):
    """一条 WiLoR 检测 → clip 里的手位。左 0 右 1，见 `to_clip` 里为什么固定。"""
    return HAND_RIGHT if float(det["is_right"]) > IS_RIGHT_THRESHOLD else HAND_LEFT


def keypoints_2d(det):
    """(21,2) 像素坐标。"""
    kp = np.array(det["wilor_preds"]["pred_keypoints_2d"]).reshape(-1, 2)
    if kp.shape[0] != N_JOINTS:
        raise ValueError(f"WiLoR 给了 {kp.shape[0]} 个 2D 关键点，期望 {N_JOINTS}")
    return kp


def native_joints_3d(det):
    """(21,3) WiLoR 原生相机系 3D = ``pred_keypoints_3d + pred_cam_t_full``。

    **这个量的绝对深度不可信**（非度量、抖动大），只有手形可信。所以它只在
    `DEPTH_GLOBAL_SCALE` 那条策略里用，而且必须再乘一个 MoGe 锚出来的尺度。
    """
    wp = det["wilor_preds"]
    kp3d = np.array(wp["pred_keypoints_3d"]).reshape(-1, 3)
    camt = np.array(wp["pred_cam_t_full"]).reshape(3)
    if kp3d.shape[0] != N_JOINTS:
        raise ValueError(f"WiLoR 给了 {kp3d.shape[0]} 个 3D 关键点，期望 {N_JOINTS}")
    return kp3d + camt[None, :]


def joints_from_pointmap(det, pointmap, joint_indices=None, patch_r=MG.DEFAULT_PATCH_R):
    """策略 A：逐关节在 MoGe 点云上取深度。返回 (21,3)，取不到的关节是 NaN。

    ``joint_indices`` 给 None 就取全部 21 个关节。原 `step3_perception.py` 只取了
    ``[0,4,8]`` 三个（它只要手腕和开合），这里默认取全 —— 下游 clip 契约要 21 个点，
    取三个的话另外 18 个只能留 NaN，重定向就没有手形了。**取哪些关节是参数，
    取一个关节的算法逐字不变**，所以子集上和原脚本逐位一致。
    """
    kp2d = keypoints_2d(det)
    idx = range(N_JOINTS) if joint_indices is None else joint_indices
    out = np.full((N_JOINTS, 3), np.nan)
    for j in idx:
        out[j] = MG.sample_pointmap(pointmap, kp2d[j], r=patch_r)
    return out


def joints_from_depth_and_K(det, depth, K, joint_indices=None,
                            patch_r=MG.DEFAULT_PATCH_R):
    """策略 A 的变体：用 MoGe 深度图 + **外部内参**反投影。

    `step16b_wilor_moge_abf12.py` 的做法，HO-3D 评测用的就是这条（那里手上有真值
    ``camMat``）。和 `joints_from_pointmap` 的差别不只是"深度图 vs 点云"——
    像素取整方式也不同，见 `moge.py` 里的说明。
    """
    kp2d = keypoints_2d(det)
    idx = range(N_JOINTS) if joint_indices is None else joint_indices
    out = np.full((N_JOINTS, 3), np.nan)
    for j in idx:
        px = MG.pixel_index(kp2d[j], depth.shape)
        out[j] = MG.unproject(px, MG.sample_depth(depth, kp2d[j], r=patch_r), K)
    return out


def global_scale(joints, scene_depth):
    """WiLoR 非度量 3D → 度量的那个全局尺度 = 场景深度 / WiLoR 手腕深度中位。

    ``step3d_wilor3d.py`` 逐字：两只手的手腕深度**混在一起**取中位（不是各算各的），
    手腕深度中位 ≤0 时退回 1.0（不缩放）而不是报错 —— 那种片段本来就该被筛掉，
    但让它安静地过去比抛异常更符合原行为。
    """
    wz = np.concatenate([joints[:, HAND_LEFT, 0, 2], joints[:, HAND_RIGHT, 0, 2]])
    wz = wz[np.isfinite(wz)]
    wilor_med = float(np.nanmedian(wz)) if len(wz) else 1.0
    return scene_depth / wilor_med if wilor_med > 0 else 1.0


def aperture(joints):
    """拇指尖-食指尖距离 (T,)，米。和 `hawor.aperture` 同一个定义。"""
    return np.linalg.norm(joints[:, MANO_THUMB_TIP] - joints[:, MANO_INDEX_TIP], axis=-1)


def wilor_to_joints(images, predict_fn, infer_fn, depth_mode=DEPTH_POINTMAP,
                    K=None, patch_r=MG.DEFAULT_PATCH_R, anchor_samples=6,
                    progress=None):
    """整段：逐帧图像 → (T,2,21,3) 相机系米制关节，喂给 `to_clip.write_clip`。

    ``images`` 是 RGB 数组的序列（``(H,W,3)``，uint8 或 float）。``predict_fn(rgb)``
    是 WiLoR 的 ``pipeline.predict``；``infer_fn(rgb)`` 要返回带 ``"points"`` 或
    ``"depth"`` 的 dict（哪个取决于 ``depth_mode`` 和 ``K``）。

    一帧检出两只同侧手时**后来的覆盖先来的** —— 这是原脚本的行为（``tgt[i,j]=...``
    直接赋值）。WiLoR 偶尔会在同一只手上出两个框，此时保留哪个都是猜，保持原行为。
    """
    if depth_mode not in DEPTH_MODES:
        raise ValueError(f"depth_mode 得是 {DEPTH_MODES} 之一，给的是 {depth_mode!r}")
    images = list(images)
    n = len(images)
    if n == 0:
        raise ValueError("一帧图像都没有")
    joints = empty_joints(n).astype(np.float64)

    if depth_mode == DEPTH_POINTMAP:
        for i, rgb in enumerate(images):
            if progress:
                progress(i, n)
            out = infer_fn(rgb)
            for det in _detections(predict_fn, rgb):
                slot = hand_slot(det)
                if K is None:
                    joints[i, slot] = joints_from_pointmap(det, out["points"],
                                                           patch_r=patch_r)
                else:
                    joints[i, slot] = joints_from_depth_and_K(det, out["depth"], K,
                                                              patch_r=patch_r)
        return joints

    # DEPTH_GLOBAL_SCALE：先把 WiLoR 原生 3D 攒齐，再统一乘一个尺度
    for i, rgb in enumerate(images):
        if progress:
            progress(i, n)
        for det in _detections(predict_fn, rgb):
            joints[i, hand_slot(det)] = native_joints_3d(det)
    anchor = [infer_fn(images[i])["depth"]
              for i in MG.anchor_frame_indices(n, anchor_samples)]
    scale = global_scale(joints, MG.scene_depth_anchor(anchor))
    return joints * scale


def _detections(predict_fn, rgb):
    """WiLoR 抛异常时当作"这帧没检出"，不中断整段 —— 原脚本的 ``try/except: outs=[]``。

    这条是对的：逐帧检测器在个别帧上炸掉是常态，整段中断反而丢掉了其余可用帧。
    """
    try:
        out = predict_fn(rgb)
    except Exception:
        return []
    return out or []


def valid_frame_counts(joints):
    """左右手各有多少帧手腕是有限值。诊断用。"""
    return {"left": int(np.isfinite(joints[:, HAND_LEFT, 0]).all(1).sum()),
            "right": int(np.isfinite(joints[:, HAND_RIGHT, 0]).all(1).sum())}


assert MAX_HANDS == 2  # 上面几处 [:, HAND_LEFT] / [:, HAND_RIGHT] 是按两只手写的
