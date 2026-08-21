"""M7 的 IK 根坐标系（``waist_pitch_link``）三个轴各自朝哪 —— 用 FK 量，不猜。

## 为什么要量

网格搜索那条路线要给候选朝向 ``R``（列 = 躯干三轴在相机系下的表示，因为上游
``cam_to_root_targets`` 算的是 ``R^T (p_cam − t)``）。想不靠 checkpoint 直接从重力
方向构造 ``R``，就得知道"躯干系里哪个轴是上、哪个轴是前"。

这个不能从命名猜（``waist_pitch_link`` 不告诉你轴序），也不该从 URDF 里的
``rpy`` 一层层手推。直接前向运动学：在 ``start_config`` 和零位下算两只手的位置，

* 左手的 y（或哪个轴）应该显著大于右手 → 那个轴就是"左"；
* 两手都在身体前方 → 两手位置均值为正的那个轴是"前"；
* 肩到手的连线基本水平 → 剩下那个轴是"上"，符号由"手比腰低还是高"定不了，
  所以额外用限位扫一下：抬臂时手该往哪个轴走。

跑法：``scripts/dev/m7_tool.sh measure_m7_torso_axes.py``
"""
import numpy as np
import torch
from kinematics.wrist_ik import RobotIKConfig, WristIK

from web2robot.robots.m7.config import CONFIG

AXES = "xyz"


def hand_position(ik: WristIK, q: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        pos, _ = ik._fk(torch.tensor(q, dtype=torch.float32).unsqueeze(0))
    return pos.cpu().numpy().reshape(3)


def main() -> None:
    iks, out = {}, {}
    for side in ("left", "right"):
        ik = WristIK(side=side, robot=RobotIKConfig.m7(side), device="cpu")
        iks[side] = ik
        n = len(ik.limits)
        qs = np.asarray(CONFIG["start_config"][side], dtype=np.float32)
        out[side] = {"zero": hand_position(ik, np.zeros(n, np.float32)),
                     "start": hand_position(ik, qs), "n_dof": n}
        print(f"[{side}] {n} dof   零位 {np.round(out[side]['zero'], 4)}   "
              f"start_config {np.round(out[side]['start'], 4)}")

    print()
    for tag in ("zero", "start"):
        l, r = out["left"][tag], out["right"][tag]
        d = l - r                                    # 左手减右手
        i_left = int(np.argmax(np.abs(d)))
        mid = (l + r) / 2
        print(f"{tag}: 左−右 = {np.round(d, 4)}  → 「左」轴 = "
              f"{'+' if d[i_left] > 0 else '-'}{AXES[i_left]}"
              f"（分量 {d[i_left]:+.3f} m，其余 {np.round(np.delete(d, i_left), 3)}）")
        print(f"      两手中点 = {np.round(mid, 4)}")

    # 「上」和「前」：把每个关节单独推到限位两端，看手往哪个轴走得最远。
    # 抬臂（肩 pitch）主要改变竖直高度，伸手（肩/肘）主要改变前向距离 —— 但不用
    # 逐关节解释语义，只需要知道哪个轴的可达范围最像"前"（单侧为主）、
    # 哪个最像"上"（双向都能走）。
    print()
    for side in ("left", "right"):
        ik = iks[side]
        lim = ik.limits.cpu().numpy()
        n = len(lim)
        rng = np.random.default_rng(0)
        Q = rng.uniform(lim[:, 0], lim[:, 1], size=(20000, n)).astype(np.float32)
        with torch.no_grad():
            pos, _ = ik._fk(torch.tensor(Q))
        p = pos.cpu().numpy()
        lo, hi = p.min(axis=0), p.max(axis=0)
        print(f"[{side}] 2 万随机构型下手部可达盒："
              + "  ".join(f"{AXES[i]}∈[{lo[i]:+.3f},{hi[i]:+.3f}]" for i in range(3)))

    print("\n判读：单侧为主（min 和 max 同号或几乎不过零）的轴 = 「前」；"
          "左右手符号相反的轴 = 「左/右」；剩下那个 = 「上」。")


if __name__ == "__main__":
    main()
