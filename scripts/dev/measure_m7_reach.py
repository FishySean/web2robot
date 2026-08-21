"""M7 的最大臂展 r_max —— 网格搜索的搜索半径，``CONFIG`` 里没有，只能实测。

## 为什么要单独测

Qwen-RobotManip 公式 (3) 的候选集是"以轨迹质心为心、被 per-morphology kinematic
reach r_max 约束的网格"。论文那 15 台机器人的 r_max 是现成的（螺在桌上的机械臂，
臂展是产品参数）。M7 这边 ``robots/m7/config.py`` 里没有这个数，所以在跑搜索之前
必须先把它测出来 —— 不然搜索半径就是拍的，而半径直接决定候选数和结果可信度。

## 怎么测

r_max = max over 关节限位内所有构型 ‖p_ee‖（在 IK 链根 ``waist_pitch_link`` 系下）。
7 自由度没法网格穷举，所以三样加起来取最大：

1. **限位角点** 2^7 = 128 个（伸到最直的构型几乎一定在角点上）
2. **固定种子的随机采样** 20 万个（补角点漏掉的中间构型）
3. 零位 / start_config（sanity 参照）

固定种子 → 结果确定，两次跑一样。这是**下界估计**（真实上确界 ≥ 测出的值），所以
外面用的时候给一点余量；但对"网格撒到多远"这个用途，低估比高估安全 —— 高估只是
多撒些必然不可行的候选，低估会把真正的最优解排除在网格外，所以脚本同时打印一个
**解析上界**（各连杆平移长度之和）当交叉验证。

跑法::

    scripts/dev/m7_tool.sh measure_m7_reach.py
"""
import numpy as np
import torch
from kinematics.wrist_ik import RobotIKConfig, WristIK

from web2robot.robots.m7.config import CONFIG

N_RANDOM = 200_000
SEED = 0


def analytic_upper_bound(ik: WristIK) -> float:
    """各连杆平移量之和 —— ‖p_ee‖ 的解析上界（三角不等式，与关节角无关）。

    只作交叉验证：实测值必须 ≤ 这个数，否则说明 FK 或链的搭法出了问题。
    """
    total = 0.0
    for frame_name in ik.chain.get_frame_names():
        frame = ik.chain.find_frame(frame_name)
        if frame is None:
            continue
        total += float(np.linalg.norm(frame.link.offset.get_matrix()[0, :3, 3].numpy()))
        total += float(np.linalg.norm(frame.joint.offset.get_matrix()[0, :3, 3].numpy()))
    return total


def measure(side: str) -> dict:
    ik = WristIK(side=side, robot=RobotIKConfig.m7(side), device="cpu",
                 q_default=np.array(CONFIG["start_config"][side], np.float32))
    lim = ik.limits.cpu().numpy().astype(np.float64)            # (7, 2)
    n_dof = len(lim)

    corners = np.array(np.meshgrid(*[lim[i] for i in range(n_dof)], indexing="ij"))
    corners = corners.reshape(n_dof, -1).T                      # (128, 7)
    rng = np.random.default_rng(SEED)
    rand = rng.uniform(lim[:, 0], lim[:, 1], size=(N_RANDOM, n_dof))
    named = np.stack([np.zeros(n_dof), np.asarray(CONFIG["start_config"][side], float)])
    Q = np.concatenate([corners, rand, named]).astype(np.float32)

    with torch.no_grad():
        pos, _ = ik._fk(torch.tensor(Q))
    r = np.linalg.norm(pos.cpu().numpy(), axis=1)

    best = int(np.argmax(r))
    return {
        "side":      side,
        "r_max":     float(r[best]),
        "r_corners": float(r[:len(corners)].max()),
        "r_random":  float(r[len(corners):len(corners) + N_RANDOM].max()),
        "r_zero":    float(r[-2]),
        "r_start":   float(r[-1]),
        "q_best":    Q[best],
        "p_best":    pos[best].cpu().numpy(),
        "bound":     analytic_upper_bound(ik),
    }


def main() -> None:
    print(f"M7 臂展实测（链根 = {RobotIKConfig.m7('left').root_link_name}，"
          f"末端 = *_hand_frame，seed={SEED}，随机采样 {N_RANDOM}）\n")
    rows = [measure(s) for s in ("left", "right")]
    for m in rows:
        print(f"[{m['side']}] r_max = {m['r_max']:.4f} m"
              f"   （限位角点 {m['r_corners']:.4f} / 随机 {m['r_random']:.4f}）")
        print(f"       零位 {m['r_zero']:.4f}   start_config {m['r_start']:.4f}"
              f"   解析上界 {m['bound']:.4f}")
        print(f"       最远构型 p_ee = {np.round(m['p_best'], 3).tolist()}")
        print(f"                  q = {np.round(m['q_best'], 3).tolist()}")
        assert m["r_max"] <= m["bound"] + 1e-6, "实测超过解析上界 —— FK 或链搭法有问题"
    r_max = max(m["r_max"] for m in rows)
    print(f"\n→ 两侧取大：r_max = {r_max:.4f} m"
          f"（左右差 {abs(rows[0]['r_max'] - rows[1]['r_max'])*1000:.1f} mm）")
    print(f"→ 网格搜索用这个数当横向半径；竖直半径默认取一半 = {r_max/2:.4f} m")


if __name__ == "__main__":
    main()
