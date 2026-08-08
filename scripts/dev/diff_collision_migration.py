"""迁移验证：旧位置与新位置的碰撞/清洗模块，同一输入必须给出**逐位相同**的输出。

为什么不用"端到端跑一遍看视频差不多"来验收这次迁移：端到端链路前半段有
训练好的根模型 + 锚点搜索 + IK，上游任何抖动都会掩盖掉"迁移是否引入了改动"
这个问题。这里把碰撞纠正和轨迹清洗**从链路里摘出来单测** —— 喂完全相同的
输入轨迹，比两份实现的输出。FD 梯度下降是确定性的、纯 CPU、无随机源，
所以要求是逐位相同（``array_equal``），不是"在容差内"。

这跟第①步质检那边故意放宽到"判决一致 + 不越阈"是两码事：那边有
KeypointRCNN 在 GPU 上的非确定性，逐位相同做不到；这边做得到，就不该放宽。

跑法（在 web2robot 仓库根）::

    envs/rt_env/bin/python scripts/dev/diff_collision_migration.py
"""
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from web2robot.paths import P                                    # noqa: E402

UPSTREAM = P.root("egoinfinity") / "retarget"
# 旧位置的模块 import 的是 `models.*` / `utils.*`，要以 retarget/ 为包根；
# 机器人 config 里的 MJCF 路径也是相对它算的，所以连 cwd 一起切过去。
sys.path.insert(0, str(UPSTREAM))
os.chdir(UPSTREAM)

# 迁移前的轨迹（根模型输出、还没过碰撞纠正）——官方片段，不是 ours_*
BASELINE = UPSTREAM / "examples/fill_jar/m7_new/trajectory.npz"
# 当时归档的、过完 ATF(含指尖)+平滑 的结果
ARCHIVED = UPSTREAM / "examples/fill_jar/m7_b2s/trajectory.npz"


def load_input():
    d = np.load(BASELINE, allow_pickle=True)
    fj = [n.replace("left_", "").replace("_joint", "")
          for n in d["left_finger_joint_names"]]
    return dict(
        q_left=d["q_left"], q_right=d["q_right"],
        q_left_fingers=d["q_left_fingers"], q_right_fingers=d["q_right_fingers"],
        finger_jnames=fj,
    )


def robot_cfg():
    import importlib
    return importlib.import_module("web2robot.robots.m7.config").CONFIG


def compare(name, a, b):
    """逐位比较两份 (q_left, q_right)。返回 True 表示完全一致。"""
    ok = True
    for side, x, y in (("q_left", a[0], b[0]), ("q_right", a[1], b[1])):
        if x.shape != y.shape:
            print(f"  ✗ {name}.{side}: 形状不同 {x.shape} vs {y.shape}")
            ok = False
        elif np.array_equal(x, y):
            print(f"  ✓ {name}.{side}: 逐位相同  shape={x.shape}")
        else:
            d = np.abs(x - y)
            print(f"  ✗ {name}.{side}: 有差异  最大 {d.max():.3e}  "
                  f"不同元素 {int((d > 0).sum())}/{d.size}")
            ok = False
    return ok


def run_pair(label, old_cls, new_cls, cfg, inp, **kw):
    print(f"\n── {label} ─────────────────────────────────────────")
    old = old_cls(cfg, verbose=False, **kw).process(**inp)
    new = new_cls(cfg, verbose=False, **kw).process(**inp)
    return compare(label, old, new)


def main():
    if not BASELINE.exists():
        sys.exit(f"缺少输入轨迹 {BASELINE}")

    inp = load_input()
    cfg = robot_cfg()
    print(f"输入: {BASELINE.relative_to(UPSTREAM)}  "
          f"T={inp['q_left'].shape[0]} 帧  手指 {len(inp['finger_jnames'])} 关节")

    from models.arm_torso_filter import ArmTorsoFilter as OldATF
    from models.dual_hand_filter import DualHandFilter as OldDHF
    from web2robot.collision import ArmTorsoFilter as NewATF
    from web2robot.collision import DualHandFilter as NewDHF

    results = {
        "ArmTorsoFilter": run_pair("ArmTorsoFilter", OldATF, NewATF, cfg, inp),
        "DualHandFilter": run_pair("DualHandFilter", OldDHF, NewDHF, cfg, inp),
    }

    # 轨迹清洗：纯函数。归档的 npz 里没存手腕原始轨迹（存的是解出来的关节角），
    # 所以按构造法造一段 —— 要点是把三条分支都覆盖到：短内部空洞→插值、
    # 长空洞→标 rest、位置爆点→判坏帧、四元数符号跳变→规范化。
    print("\n── traj_cleanup ───────────────────────────────────")
    import utils.traj_cleanup as old_tc
    import web2robot.trajectory.traj_cleanup as new_tc

    T, fps = inp["q_left"].shape[0], 15.0
    rng = np.random.RandomState(0)
    traj = np.zeros((T, 7), dtype=np.float32)
    traj[:, :3] = np.cumsum(rng.randn(T, 3) * 0.01, axis=0)
    traj[:, 3] = 1.0                              # 单位四元数 (w 在第 4 位)
    traj[T // 2:, 3] = -1.0                       # 符号跳变 → canonicalize
    traj[T // 3: T // 3 + 5] = np.nan             # 短空洞(5 帧) → interp
    traj[-20:] = np.nan                           # 长边界空洞(20 帧) → rest
    traj[T // 4, :3] += 3.0                       # 位置爆点 3 m → 判坏帧
    print(f"  输入: 合成 {T} 帧 @ {fps}fps，含短空洞 5 帧 / 长空洞 20 帧 / "
          f"3m 爆点 1 帧 / 四元数符号跳变")

    o = old_tc.clean_wrist_trajectory(traj.copy(), fps, side="left", verbose=False)
    n = new_tc.clean_wrist_trajectory(traj.copy(), fps, side="left", verbose=False)
    same = len(o) == len(n)
    for i, (x, y) in enumerate(zip(o, n)):
        eq = (x == y) if isinstance(x, dict) else np.array_equal(
            np.asarray(x), np.asarray(y), equal_nan=True)
        same &= bool(eq)
        if not eq:
            print(f"  ✗ 返回项 {i} 不同: {x!r:.120} vs {y!r:.120}")
    print(f"  {'✓' if same else '✗'} clean_wrist_trajectory: "
          f"{'4 个返回项(traj/status/cause/report)全部相同' if same else '有差异'}")
    print(f"    报告: 坏帧 {o[3].get('n_bad')} / 插值 {o[3].get('n_interp')} / "
          f"保持 {o[3].get('n_hold')} / rest {o[3].get('n_rest')} / "
          f"四元数翻转 {o[3].get('quat_flips')}  —— 三条分支都被走到了")
    results["traj_cleanup"] = same

    print("\n" + "=" * 52)
    bad = [k for k, v in results.items() if not v]
    if bad:
        print("迁移引入了行为变化: " + ", ".join(bad))
        sys.exit(1)
    print(f"全部一致 ({len(results)}/{len(results)}) —— 迁移是纯移动，行为未变")
    print(f"\n参考: 当时归档的结果在 {ARCHIVED.relative_to(UPSTREAM)}，"
          "端到端复现由 test.py 那一步负责")


if __name__ == "__main__":
    main()
