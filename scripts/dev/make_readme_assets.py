"""生成 README 里引用的图/GIF，落到 ``docs/assets/``（**进 git**）。

为什么产物不落 `outputs/`：`outputs/` 不进 git，README 里的图必须跟着仓库走，
不然别人 clone 下来看到的是一片红叉。这是全工程唯一一个"产物不落 outputs"的例外，
所以单独写一个脚本、并且每张图都记下它是从哪个 run 目录、哪一帧生成的 —— 图一旦
和数据脱钩就会变成宣传物料，这个工程的规矩是画面必须能追回到那次实测。

两个子命令：

    collision <before_dir> <after_dir>   碰撞修复前/后同一帧的对照图
    demo      <run_dir>                  源视频 + 机器人动画并排的 GIF

``before_dir`` / ``after_dir`` 必须是**同一次 IK、只差碰撞过滤开关**的两个 run
（例如同一条命令跑两遍，一遍不加 ``--arm_torso_collision``）。否则图上的差别里
混进了 IK 的随机性，就不再是"碰撞过滤的功劳"。脚本会核对两边 ``ik_rate`` 一致，
不一致直接报错。
"""
import argparse
import sys
from pathlib import Path

import cv2
import imageio.v2 as imageio
import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from web2robot.collision import M7CapsuleModel        # noqa: E402
from web2robot.paths import P                         # noqa: E402
from web2robot.robots.m7.env import M7Env             # noqa: E402

ASSETS = P.repo_root / "docs" / "assets"


def _load(run_dir: Path):
    tr = np.load(run_dir / "trajectory.npz", allow_pickle=True)
    return {
        "ql": tr["q_left"].astype(np.float64), "qr": tr["q_right"].astype(np.float64),
        "fl": tr["q_left_fingers"].astype(np.float64),
        "fr": tr["q_right_fingers"].astype(np.float64),
        "Ln": [str(x) for x in tr["left_finger_joint_names"]],
        "Rn": [str(x) for x in tr["right_finger_joint_names"]],
        "ik": float(np.load(run_dir / "metrics.npz", allow_pickle=True)["ik_rate"]),
    }


def _pose(env, tr, t):
    env.set_arm_joints("left", tr["ql"][t])
    env.set_arm_joints("right", tr["qr"][t])
    env.set_finger_joints(tr["fl"][t], tr["Ln"])
    env.set_finger_joints(tr["fr"][t], tr["Rn"])       # 内部已 mj_forward


def _cam(az, el, dist, tgt):
    c = mujoco.MjvCamera()
    c.azimuth, c.elevation, c.distance = az, el, dist
    c.lookat[:] = tgt
    return c


def _label(img, text, color, y=26):
    """左上角贴一行 ASCII 标签（cv2 的字体没有中文，所以图上一律英文）。"""
    cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 4)
    cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2)
    return img


def _profile(env, cap, tr):
    """逐帧最深穿透（我方代理，含手指），负 = 穿进躯干。"""
    out = np.zeros(len(tr["ql"]))
    for t in range(len(tr["ql"])):
        _pose(env, tr, t)
        p = cap.arm_torso_penetrations(env.data, margin=0.0, include_fingers=True)
        out[t] = min(p["left"], p["right"])
    return out


def _worst_bone(cap, data):
    """当前姿态下离躯干最近（或穿得最深）的那根骨段胶囊 → (a, b, r)。

    画出来的就是过滤器**真正在算的那个东西**（骨段胶囊 vs 躯干盒），比只贴一个数字
    更能说明"代理几何"是什么意思 —— README 里正文讲的也是这一层。
    """
    from web2robot.collision.capsule_collision import _capsule_box_sdf
    center, R = cap._torso_frame(data)
    best, best_d = None, np.inf
    for segs in cap.all_bone_worlds(data).values():
        for a, b, r in segs:
            d = _capsule_box_sdf(a, b, r, center, R, cap.TORSO_HALF)
            if d < best_d:
                best, best_d = (a, b, r), d
    return best, best_d


def _overlay_capsule(scene, a, b, r, rgba):
    """往已经 update 好的 scene 上追加一根半透明胶囊（不改模型，只加渲染 geom）。"""
    if scene.ngeom >= scene.maxgeom:
        return
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3),
                        np.zeros(3), np.zeros(9), np.asarray(rgba, dtype=np.float32))
    mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, r,
                         np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64))
    scene.ngeom += 1


def _overlay_torso_box(scene, cap, data, rgba=(0.25, 0.5, 1.0, 0.15)):
    """把躯干那个各向异性盒也画出来。

    不画盒子这张图会自相矛盾：胶囊看着只挨到胸壳一点点，标注却写"进去 10.5 cm" ——
    因为过滤器量的是**盒**，盒比可见的胸壳大。画出来读者才对得上号。
    """
    if scene.ngeom >= scene.maxgeom:
        return
    center, R = cap._torso_frame(data)
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_BOX,
                        np.asarray(cap.TORSO_HALF, dtype=np.float64),
                        np.asarray(center, dtype=np.float64),
                        np.asarray(R, dtype=np.float64).reshape(9),
                        np.asarray(rgba, dtype=np.float32))
    scene.ngeom += 1


def cmd_collision(args):
    env = M7Env()
    cap = M7CapsuleModel(env.model)
    before, after = _load(args.before_dir), _load(args.after_dir)
    if abs(before["ik"] - after["ik"]) > 1e-6:
        raise SystemExit(
            f"两个 run 的 ik_rate 不一样（{before['ik']:.4f} vs {after['ik']:.4f}）——\n"
            "说明它们不是「同一次 IK 只差碰撞开关」，"
            "对照图会把 IK 的随机性算进碰撞的功劳。")
    if len(before["ql"]) != len(after["ql"]):
        raise SystemExit("两个 run 帧数不同，没法对同一帧。")

    pb, pa = _profile(env, cap, before), _profile(env, cap, after)
    if args.frame is not None:
        t = args.frame
    else:
        # 挑"修好得最多"的那一帧：修复前穿得最深、修复后回到躯干外面。
        # 不挑"修复前最深"那一帧 —— 最深的那些帧过滤器未必修得动（w_ee=60 的保真项
        # 压着推出项），拿修不动的帧当宣传图就是自欺。
        gain = np.where(pb < 0, np.minimum(pa, 0) - np.minimum(pb, 0), 0)
        t = int(gain.argmax())
    print(f"frame {t}: before {pb[t] * 100:+.2f} cm -> after {pa[t] * 100:+.2f} cm "
          f"(全片: 穿透帧 {int((pb < 0).sum())} → {int((pa < 0).sum())} / {len(pb)})")

    H, W = 430, 470
    R = mujoco.Renderer(env.model, height=H, width=W)
    # 版式：列 = BEFORE / AFTER（左右并排最好比），行 = 正面 / 45° 近景
    cols = {}
    for tag, tr, depth, rgba in (
            ("BEFORE  %.1f cm inside torso" % (-pb[t] * 100), before, pb, (1.0, .25, .25, .55)),
            ("AFTER   %+.1f cm  clear" % (pa[t] * 100), after, pa, (.25, .85, .35, .55))):
        _pose(env, tr, t)
        bone, _ = _worst_bone(cap, env.data)
        views = []
        for az, el, dist in ((180, -6, 1.15), (135, -4, 1.05)):
            R.update_scene(env.data, _cam(az, el, dist, args.lookat))
            _overlay_torso_box(R.scene, cap, env.data)   # 躯干代理盒
            _overlay_capsule(R.scene, *bone, rgba)       # 犯规最严重的那根骨段胶囊
            views.append(R.render().copy())
        col = np.concatenate(views, axis=0)
        color = (60, 60, 255) if depth is pb else (40, 170, 60)
        cols[tag] = _label(col, tag, color)
    grid = np.concatenate(list(cols.values()), axis=1)
    bar = np.full((30, grid.shape[1], 3), 255, np.uint8)
    grid = np.concatenate([grid, bar], axis=0)
    grid = _label(grid, "M7 / fill_jar / frame %d   top: front, bottom: 45deg   "
                        "blue box + capsule = proxy geometry the filter measures" % t,
                  (40, 40, 40), y=grid.shape[0] - 10)

    ASSETS.mkdir(parents=True, exist_ok=True)
    out = args.out or ASSETS / "collision_fix_fill_jar.png"
    imageio.imwrite(out, grid)
    print(f"wrote {out}  ({grid.shape[1]}x{grid.shape[0]})")


def _content_box(reader, idx, thresh=245):
    """扫几帧，求"非白像素"的公共外接框 → (y0, y1, x0, x1)。

    ``robot_sim.mp4`` 是 1440×1080 的白底渲染，机器人只占中间一小块，直接缩到 GIF
    里就是一小个人配一大片白。逐帧各自裁会让机器人在 GIF 里跳，所以取**整段的并集框**
    裁一次，机器人在画面里的位置就还是稳的。
    """
    y0, y1, x0, x1 = 10 ** 9, -1, 10 ** 9, -1
    for i in idx:
        reader.set_image_index(i)
        im = np.array(reader.get_next_data())
        mask = im.min(axis=2) < thresh
        if not mask.any():
            continue
        ys, xs = np.where(mask)
        y0, y1 = min(y0, ys.min()), max(y1, ys.max())
        x0, x1 = min(x0, xs.min()), max(x1, xs.max())
    return y0, y1 + 1, x0, x1 + 1


def cmd_demo(args):
    """``input_viz.mp4`` + ``robot_sim.mp4`` 并排 → GIF。

    GIF 而不是 mp4，是因为 GitHub 的 README 不播放 mp4，只显示图片。抽帧步长和
    宽度都留成参数，因为文件大小要压到能进 git 的量级（目标 < 4 MB）。
    """
    ri = imageio.get_reader(args.run_dir / "input_viz.mp4")
    rr = imageio.get_reader(args.run_dir / "robot_sim.mp4")
    n = min(ri.count_frames(), rr.count_frames())
    idx = list(range(args.start, min(n, args.start + args.count * args.step), args.step))
    y0, y1, x0, x1 = _content_box(rr, idx[::4])
    pad = 12
    h = args.height
    frames = []
    for i in idx:
        ri.set_image_index(i)
        rr.set_image_index(i)
        a, b = np.array(ri.get_next_data()), np.array(rr.get_next_data())
        b = b[max(0, y0 - pad):y1 + pad, max(0, x0 - pad):x1 + pad]
        fit = lambda im: cv2.resize(im, (int(im.shape[1] * h / im.shape[0]), h))  # noqa: E731
        a, b = fit(a), fit(b)
        # 标签占一条白边，不压在画面上 —— 压在渲染图上会挡住机器人的头
        strip = lambda im: np.concatenate(                                   # noqa: E731
            [np.full((26, im.shape[1], 3), 255, np.uint8), im], axis=0)
        a, b = strip(a), strip(b)
        frames.append(np.concatenate([_label(a, "INPUT  video + hand 3D", (150, 90, 0), y=19),
                                      _label(b, "OUTPUT  M7 joint trajectory", (0, 110, 190),
                                             y=19)],
                                     axis=1))
    ASSETS.mkdir(parents=True, exist_ok=True)
    out = args.out or ASSETS / f"demo_{args.run_dir.name}.gif"
    imageio.mimsave(out, frames, duration=args.step / 15.0, loop=0)
    print(f"wrote {out}  ({len(frames)} frames, {out.stat().st_size / 1e6:.2f} MB)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("collision", help="碰撞修复前/后对照图")
    c.add_argument("before_dir", type=Path, help="不开碰撞过滤那次的 run 目录")
    c.add_argument("after_dir", type=Path, help="开了碰撞过滤那次的 run 目录")
    c.add_argument("--frame", type=int, default=None, help="指定帧号；默认自动挑修得最多的一帧")
    c.add_argument("--lookat", type=float, nargs=3, default=[0.15, 0.0, 0.15])
    c.add_argument("--out", type=Path, default=None)
    c.set_defaults(fn=cmd_collision)

    d = sub.add_parser("demo", help="源视频 + 机器人动画并排 GIF")
    d.add_argument("run_dir", type=Path, help="一次 s4_retarget.sh 的输出目录")
    d.add_argument("--start", type=int, default=0)
    d.add_argument("--count", type=int, default=60, help="抽多少帧")
    d.add_argument("--step", type=int, default=3, help="每隔几帧抽一帧（源 15 fps）")
    d.add_argument("--height", type=int, default=300)
    d.add_argument("--out", type=Path, default=None)
    d.set_defaults(fn=cmd_demo)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
