#!/usr/bin/env python3
"""独立审计器：拿官方 MuJoCo mesh contacts 复核我方胶囊代理的臂-躯判决。

为什么要有这个脚本（2026-08-11）：我方 ``collision/`` 用的是代理几何（躯干盒 +
手臂胶囊 + 指尖球）的有符号距离，代理**一定**和真实网格有偏差。这个脚本把
``m7.xml`` 里本来就开着的 98 个 mesh 碰撞 geom 拿来当**第二个独立判据**，
只报告、不修改任何轨迹 —— 定位是"体检"，不是流水线的一环。

三件事：
  1. 报告两个 MJCF 各有多少 geom 开着碰撞（``m7.xml`` 开着，``m7_mjx.xml`` 全关，
     后者是我们自己生成的 FK-only 训练模型，别把两者搞混）。
  2. 说明上游 ``models/collision.py`` 为什么在 M7 上查不到东西 —— 不是因为 geom
     关着，而是它的 geom 集合构造把躯干和整只手都排除在外了（见下面打印）。
  3. 逐帧对照 MuJoCo 与我方代理的臂-躯判决，报出**双方分歧的帧**。

``STRUCTURAL`` 是必须排掉的结构性自重叠：URDF 转出来的凸包在静息位就互相插着
（q=0 时 ncon=10，最深 3.1 cm），这些 body 对在每一帧都出现，是模型的自重叠而
不是碰撞信号。不排掉的话"ncon > 0"这个判据恒为真。

    scripts/dev/m7_tool.sh audit_mujoco_contacts.py outputs/retarget/fill_jar
"""
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from _devcli import load_traj, parser                    # noqa: E402
from web2robot.collision import M7CapsuleModel           # noqa: E402
from web2robot.paths import P                            # noqa: E402
from web2robot.robots.m7.env import M7Env                # noqa: E402

#: 每帧都出现的 body 对 = 凸包结构性自重叠，不是碰撞。实测自 fill_jar 216 帧
#: （216/216 帧都在），另有 q=0 静息位复核。
STRUCTURAL = {
    ("waist_yaw_link", "world"),
    ("waist_pitch_link", "waist_yaw_link"),
    ("left_elbow_yaw_link", "left_wrist_roll_link"),
    ("right_elbow_yaw_link", "right_wrist_roll_link"),
    ("left_hand_index_rota_link1", "left_wrist_roll_link"),
    ("right_hand_index_rota_link1", "right_wrist_roll_link"),
}
#: 躯干侧的 body（``waist_pitch_link`` 是 IK 链的根，另两个是它上面的腰关节）
TORSO_BODIES = {"waist_yaw_link", "waist_roll_link", "waist_pitch_link"}


def geom_census():
    for key, tag in (("m7_mjcf", "m7.xml（IK / 渲染 / 碰撞过滤器加载的）"),
                     ("m7_mjx", "m7_mjx.xml（我们生成的 MJX 训练模型，FK only）")):
        m = mujoco.MjModel.from_xml_path(str(P.asset(key)))
        on = [g for g in range(m.ngeom)
              if m.geom_contype[g] or m.geom_conaffinity[g]]
        print(f"  {tag}: ngeom={m.ngeom}  开碰撞={len(on)}")


def upstream_geom_sets(model):
    """复现上游 ``CollisionFilter._build_geom_sets``，看它在 M7 上圈到了什么。"""
    def chain(name):
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        ids = set()
        while bid > 0:
            ids.add(bid)
            bid = int(model.body_parentid[bid])
        return ids

    left, right = chain("left_hand_frame"), chain("right_hand_frame")
    shared = left & right
    lo, ro = left - shared, right - shared
    gset = lambda bs: {g for g in range(model.ngeom)  # noqa: E731
                       if int(model.geom_bodyid[g]) in bs}
    nm = lambda b: mujoco.mj_name2id and mujoco.mj_id2name(  # noqa: E731
        model, mujoco.mjtObj.mjOBJ_BODY, b)
    print(f"  上游圈到 left={len(gset(lo))} geom  right={len(gset(ro))} geom")
    print(f"  被当作 shared 剔掉：{sorted(nm(b) for b in shared)}")
    n_hand = sum(1 for g in range(model.ngeom)
                 if "hand" in (nm(int(model.geom_bodyid[g])) or ""))
    print(f"  ！手/手指的 {n_hand} 个 geom 一个都不在 left/right 里 —— "
          f"chain_to_root 只走 hand_frame 到根的那条链，手指是另一条分支")
    return gset(lo), gset(ro)


def main():
    ap = parser(__doc__)
    ap.add_argument("--margin", type=float, default=0.0,
                    help="我方代理的间隙门槛 [m]，默认 0（只判真穿透）")
    args = ap.parse_args()
    traj, out_dir = load_traj(args)
    ql = traj["q_left"].astype(np.float64)
    qr = traj["q_right"].astype(np.float64)
    n = len(ql)

    print("① 两个 MJCF 的碰撞 geom 数：")
    geom_census()
    env = M7Env()
    env.reset()
    model, data = env.model, env.data
    print("\n② 上游 CollisionFilter 在 M7 上圈到了什么：")
    lg, rg = upstream_geom_sets(model)
    cap = M7CapsuleModel(model)

    def bname(g):
        return mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[g])) or "?"

    mj = {"left": np.zeros(n), "right": np.zeros(n)}   # 穿透深度，正数
    ours = {"left": np.zeros(n), "right": np.zeros(n)}  # 有符号距离，负=穿透
    upstream_cross = 0
    for t in range(n):
        env.set_arm_joints("left", ql[t])
        env.set_arm_joints("right", qr[t])
        mujoco.mj_forward(model, data)
        for i in range(data.ncon):
            c = data.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            if (g1 in lg and g2 in rg) or (g1 in rg and g2 in lg):
                upstream_cross += 1
            b1, b2 = bname(g1), bname(g2)
            if tuple(sorted((b1, b2))) in STRUCTURAL:
                continue
            pair = {b1, b2}
            if not pair & TORSO_BODIES or pair <= TORSO_BODIES:
                continue
            other = (pair - TORSO_BODIES).pop()
            side = ("left" if other.startswith("left")
                    else "right" if other.startswith("right") else None)
            if side:
                mj[side][t] = max(mj[side][t], -float(c.dist))
        p = cap.arm_torso_penetrations(data, margin=args.margin,
                                       include_fingers=True)
        ours["left"][t], ours["right"][t] = p["left"], p["right"]

    print(f"\n③ 逐帧对照（{n} 帧，{args.run_dir.name}）")
    print(f"  上游那套判出的跨臂 contact：{upstream_cross} 个"
          f"{'  ← 印证它在 M7 上是瞎的' if upstream_cross == 0 else ''}")
    disagree = {}
    for side in ("left", "right"):
        mj_bad, our_bad = mj[side] > 0, ours[side] < 0
        both = mj_bad & our_bad
        only_mj = mj_bad & ~our_bad
        only_our = our_bad & ~mj_bad
        print(f"  [{side}] MuJoCo {mj_bad.sum()} 帧（最深 {mj[side].max()*100:.2f} cm）"
              f" / 我方 {our_bad.sum()} 帧（最深 {-ours[side].min()*100:.2f} cm）"
              f" → 都判 {both.sum()}，只有 MuJoCo {only_mj.sum()}，"
              f"只有我方 {only_our.sum()}")
        if only_mj.any():
            idx = np.where(only_mj)[0]
            print(f"        ⚠ 我方漏掉的帧：{idx.tolist()}  "
                  f"MuJoCo 最深 {mj[side][only_mj].max()*100:.2f} cm")
            disagree[side] = idx
        if both.any():
            print(f"        共同判违规的帧上深度差中位 "
                  f"{np.median(np.abs(mj[side][both] + ours[side][both]))*100:.2f} cm"
                  f"（口径不同，看量级不看逐值）")
    np.savez(out_dir / "mujoco_contact_audit.npz",
             mj_left=mj["left"], mj_right=mj["right"],
             ours_left=ours["left"], ours_right=ours["right"],
             missed_left=disagree.get("left", np.array([], int)),
             missed_right=disagree.get("right", np.array([], int)))
    print(f"\n写出 {out_dir / 'mujoco_contact_audit.npz'}")
    print("这个脚本只报告不修改轨迹 —— 它是我方代理的独立复核，不是流水线的一环。")


if __name__ == "__main__":
    main()
