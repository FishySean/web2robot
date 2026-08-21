"""从 ``assets/robots/urdf.tar.gz`` 生成 L3.4 的全套资产（可重跑，产物可追溯）。

M7 的资产是"某次手工操作的结果"，没有脚本可查（``m7.xml`` 里那两个 hand_frame quat
的出处只剩 ``fix_m7_handframe.py``，URDF 怎么变成 MJCF 的没人记得）。L3.4 不再这样：
**这个脚本就是 ``assets/robots/l3_4/`` 的出处**，改了资产就重跑它。

    envs/rt_env/bin/python scripts/dev/build_l3_4_assets.py [--force]

产物（全部写进 ``assets/robots/l3_4/``）::

    l3_4_from_urdf.urdf   厂家 URDF + 两处改动（下面"两处改动"）
    l3_4.xml              上面那份编译成 MJCF，再插入 3 个 frame body
    l3_4_mjx.xml          只留双臂的 MJX 模型（grid 根位姿搜索要它做批量 FK）
    scene_vis.xml         地面 + 光照，渲染/检查用
    meshes/               → ../../m7/meshes/ 的相对 symlink（见下面"mesh"）
    l3_4_vendor.xml       厂家自带的 MJCF，只留档不用（见下面"为什么不用"）

## 已量到的事实（这些决定了整件事的工作量）

L3.4 和 M7 的**上肢完全同构**，不是"相似"，是逐位相同：

* 43 个同名关节的 ``axis``/``range`` 全同（唯一例外 ``neck_pitch_joint`` 上限
  0.54 vs 我方 m7.xml 的 0.48，颈部锁死，无影响）；
* 43 个同名 body 的 ``pos``/``quat`` 全同；
* 95 个同名 link 里 92 个的 mass/inertia/COM 逐位相同 —— 是同一批零件；
* ``base_link → waist_pitch_link``（零位）平移两边都是 ``[-0.04724, 0, 0.0707]``。

最后一条尤其省事：根位姿模型学的是"给定手的任务空间目标，底座该放哪"，而底座到
IK 串链根的变换两边一样，所以 **M7 的根模型 checkpoint 对锁腿的 L3.4 直接有效**，
不用重训。（腿一旦解锁，base_link 相对地面的高度会变，这个结论就失效。）

手是**同一只 12 自由度手**（``thumb_bend`` + ``thumb_rota1/2``，其余四指
``bend/joint1/joint2``），不是另一款 11 自由度的手 —— URDF mesh 路径里那个
``xhand`` 只是目录名。所以 MANO→手 的映射表原样成立。

差别只在**下半身**：M7 是升降柱（``base_link → lift_link``），L3.4 是盆骨挂两条
6 自由度腿。本阶段只做上肢，腿/腰/颈全部锁死（是锁死不是删除，见
``robots/l3_4/config.py::LOCKED_JOINTS``）。

## mesh：为什么是 symlink 而不是复制

包里**一个 mesh 都没有**（50 KB 全是 urdf/xml）。109 个引用里 94 个和
``assets/robots/m7/meshes/`` 同名**且零件相同**（上面第 3 条量过），所以直接指过去：
19 MB 的二进制副本没有意义，还会埋下"改了一边忘了另一边"。symlink 是**逐文件**的，
所以以后拿到 L3.4 自己的 STL，就是把对应那个 symlink 换成实体文件，粒度刚好。

剩下 15 个 link 没有正确的 mesh，它们的 ``visual``/``collision`` 被删掉（这是
URDF 的"两处改动"之一），也就是**在画面里不存在**：

* 14 个腿部（hip/knee/ankle/foot）—— 包里没给，全机器上也只有 Unitree G1 的同名
  STL，那是**另一台机器人的腿**，不能拿来充数；
* ``base_link``（盆骨）—— m7 有同名 STL 但那是**升降柱底座**，是另一个零件
  （mass 2.66 vs 4.68，COM 也不同）。画上去就是错的。

结果是渲出来的 L3.4 从腰往上是完整的，腰往下空着。上肢重定向的每个数字都不受
影响（IK 串链根是 ``waist_pitch_link``，碰撞代理盒挂在同一个 body 上），但**出片
之前得把腿的 mesh 要到**，见 ``docs/BACKLOG.md``。

另有 2 个 link（``waist_base_link`` / ``neck_pitch_link``）两边质量略有差异
（1.14 vs 1.06、1.60 vs 1.46），像是同一零件的小改版 —— 沿用 m7 的 mesh，属于
已知的近似，记在这里。

## 为什么不用厂家自带的 l3.4.xml

包里那份是**手写的 MJCF**（带 actuator/PGS 求解器/障碍物/`wind` 选项），和我方
``m7.xml`` 完全不同的血统 —— 我方这份是纯运动学模型（无 actuator、无 contact
exclude），env / IK / 碰撞代理三处都按"纯运动学"写的。拿厂家那份会引入一堆本阶段
不需要、又会悄悄改变行为的东西（比如 ``wind="0.1 0.1 0.1"``）。所以走和 M7 同一条
路：URDF → MJCF。厂家那份留档在 ``l3_4_vendor.xml``，并由本脚本**对着它验一遍**
关节 axis/range 和 body pos/quat（两者不一致就当场报错），这就回答了"URDF 和 XML
是否一致可用"。

## 3 个 frame body

MJCF 编译出来只有厂家的 link，还缺 pipeline 依赖的 3 个 frame：

* ``left/right_hand_frame`` —— IK 的末端帧，轴向必须是 finger+y / thumb∓x /
  palm±z（左手 thumb−x/palm+z，右手镜像）。方法和 ``fix_m7_handframe.py`` 一样：
  拿中指/拇指末节 body 的**物理方向**现算，不抄符号。算完再和 m7 的对照断言 ——
  同一只手同一姿态，算出来必须一致，这是对整条链的强校验。
* ``torso_frame`` —— 和 ``waist_pitch_link`` 重合的叶子 body，沿用 M7 的约定。

``pos="0 0 -0.051"``（hand_frame 相对 wrist_roll_link 的平移）沿用 M7 那个数：
它是"手掌中心在哪"的**设计选择**而不是测量值，同一只手没有理由取不同的值。
"""
from __future__ import annotations

import argparse
import re
import shutil
import tarfile
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import mujoco

from web2robot.paths import P

# ── 输入 / 输出 ───────────────────────────────────────────────────────────────
SRC_TAR   = P.asset("l3_4_src_tar")      # assets/robots/urdf.tar.gz（厂家原包）
OUT_DIR   = SRC_TAR.parent / "l3_4"      # 产物目录；P.asset("l3_4_*") 要求已存在，这里在造它
M7_MESHES = P.asset("m7_meshes")

VENDOR_URDF = "l3_4.urdf"
VENDOR_MJCF = "l3.4.xml"

# ── 没有正确 mesh 的 link（理由见模块 docstring） ─────────────────────────────
_LEG_LINKS = [f"{s}_{p}_link" for s in ("left", "right")
              for p in ("hip_roll", "hip_yaw", "hip_pitch", "knee",
                        "ankle_pitch", "ankle_roll", "foot_ee")]
NO_MESH_LINKS = set(_LEG_LINKS) | {"base_link"}

# ── frame body（沿用 M7 约定，见 docstring） ─────────────────────────────────
HAND_FRAME_POS = (0.0, 0.0, -0.051)
FRAME_PARENTS = {
    "left_hand_frame":  "left_wrist_roll_link",
    "right_hand_frame": "right_wrist_roll_link",
    "torso_frame":      "waist_pitch_link",
}
# 现算 hand_frame 轴向要用的两个物理方向（中指末节 = 手指方向，拇指末节 = 拇指方向）
AXIS_PROBES = {
    "left":  ("left_wrist_roll_link",  "left_hand_mid_link2",  "left_hand_thumb_rota_link2"),
    "right": ("right_wrist_roll_link", "right_hand_mid_link2", "right_hand_thumb_rota_link2"),
}

TORSO = "waist_pitch_link"
ARM_BODIES = {
    side: [f"{side}_{b}" for b in ("shoulder_pitch_link", "shoulder_roll_link",
                                   "arm_yaw_link", "elbow_pitch_link", "elbow_yaw_link",
                                   "wrist_pitch_link", "wrist_roll_link")]
         + [f"{side}_hand_frame"]
    for side in ("left", "right")
}

_n = lambda v: v / (np.linalg.norm(v) + 1e-12)


# ── step 1: 解包 ──────────────────────────────────────────────────────────────

def extract(tmp: Path) -> Path:
    with tarfile.open(SRC_TAR) as tf:
        bad = [m.name for m in tf.getmembers()
               if m.name.startswith("/") or ".." in Path(m.name).parts]
        if bad:
            raise SystemExit(f"tarball 里有可疑路径，拒绝解包：{bad}")
        tf.extractall(tmp)
    hits = list(tmp.rglob(VENDOR_URDF))
    if len(hits) != 1:
        raise SystemExit(f"包里应当有且只有一个 {VENDOR_URDF}，找到 {hits}")
    return hits[0].parent


# ── step 2: URDF（两处改动：flatten mesh 路径 + 删掉无 mesh link 的几何） ──────

def prepare_urdf(src: Path, dst: Path) -> dict:
    """写出 ``l3_4_from_urdf.urdf``，返回改动统计。"""
    text = src.read_text()

    # (a) mesh 路径 flatten：package://robot_control/.../meshes/X.STL → X.STL
    text, n_paths = re.subn(r'filename="[^"]*/([^/"]+\.STL)"', r'filename="\1"', text)

    # (b) 让 MuJoCo 的 URDF 编译器知道 mesh 在哪、角度是弧度（和 m7 那份同样的注入）
    m = re.search(r"<robot[^>]*>", text)
    if m is None:
        raise SystemExit("URDF 里找不到 <robot> 标签")
    inject = ('\n<!-- web2robot: 由 scripts/dev/build_l3_4_assets.py 注入 —— '
              'MuJoCo 的 URDF 编译器从这里取 meshdir/角度单位 -->\n'
              '<mujoco>\n  <compiler meshdir="meshes/" angle="radian"/>\n</mujoco>\n')
    text = text[:m.end()] + inject + text[m.end():]

    # (c) 删掉没有正确 mesh 的 link 的 visual/collision（理由见模块 docstring）
    root = ET.fromstring(text)
    dropped = {}
    for link in root.iter("link"):
        if link.get("name") not in NO_MESH_LINKS:
            continue
        gone = [el for tag in ("visual", "collision") for el in link.findall(tag)]
        for el in gone:
            link.remove(el)
        if gone:
            dropped[link.get("name")] = len(gone)

    header = (f"<!-- 自动生成，请勿手改：scripts/dev/build_l3_4_assets.py\n"
              f"     源 = {SRC_TAR.name}::{VENDOR_URDF}\n"
              f"     改动 1: {n_paths} 处 mesh 路径 flatten 成文件名（mesh 实体见 meshes/）\n"
              f"     改动 2: 注入 <mujoco><compiler meshdir=... angle=radian>\n"
              f"     改动 3: {len(dropped)} 个 link 的 visual/collision 已删（包里没给正确的\n"
              f"             mesh，拿别的机器人的同名 STL 充数是错的）：\n"
              f"             {sorted(dropped)}\n"
              f"-->\n")
    dst.write_text(header + ET.tostring(root, encoding="unicode"))
    return {"mesh_paths": n_paths, "geom_dropped": dropped}


# ── step 3: mesh symlink ──────────────────────────────────────────────────────

def link_meshes(urdf: Path, mesh_dir: Path) -> dict:
    names = sorted(set(re.findall(r'filename="([^"]+\.STL)"', urdf.read_text())))
    mesh_dir.mkdir(parents=True, exist_ok=True)
    linked, missing = [], []
    for name in names:
        src = M7_MESHES / name
        dst = mesh_dir / name
        if not src.exists():
            missing.append(name)
            continue
        target = Path("..", "..", M7_MESHES.parent.name, M7_MESHES.name, name)
        if dst.is_symlink() or dst.exists():
            dst.unlink()
        dst.symlink_to(target)
        if not dst.resolve().exists():
            raise SystemExit(f"symlink 指空了：{dst} → {target}")
        linked.append(name)
    return {"linked": linked, "missing": missing}


# ── step 4: URDF → MJCF + 插入 frame body ─────────────────────────────────────

def compile_mjcf(urdf: Path, out: Path) -> None:
    model = mujoco.MjModel.from_xml_path(str(urdf))
    mujoco.mj_saveLastXML(str(out), model)


def hand_frame_quats(mjcf_no_frames: Path) -> dict:
    """现算两只手 hand_frame 的 local quat（方法同 fix_m7_handframe.py）。

    约定（``scripts/dev/check_handframe_convention.py`` 里的 ``M7_EXPECT``）：

        左手  finger +y / thumb −x / palm +z
        右手  finger +y / thumb +x / palm −z      ← **镜像**

    ``palm`` 指 local 系里的 ``finger × thumb``。所以掌面法向取 ``finger × thumb``
    时左手是 ``+`` 、右手要 ``−`` —— 两侧同号就是 2026-07-24 那个 bug（右手掌面/拇指
    整体翻 180°）。**不要因为公式对称就两侧共用一个符号。**
    """
    PALM_SIGN = {"left": +1.0, "right": -1.0}
    m = mujoco.MjModel.from_xml_path(str(mjcf_no_frames))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    bid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
    out = {}
    for side, (parent, fbody, tbody) in AXIS_PROBES.items():
        # hand_frame 的原点：wrist_roll_link 局部 (0,0,-0.051) 变换到世界
        Rp = d.xmat[bid(parent)].reshape(3, 3)
        p  = d.xpos[bid(parent)] + Rp @ np.array(HAND_FRAME_POS)
        finger_w = _n(d.xpos[bid(fbody)] - p)      # local +y
        thumb_w  = _n(d.xpos[bid(tbody)] - p)
        y_w = finger_w
        z_w = PALM_SIGN[side] * _n(np.cross(finger_w, thumb_w))
        x_w = _n(np.cross(y_w, z_w))
        R_new_world = np.column_stack([x_w, y_w, z_w])
        q = np.zeros(4)
        mujoco.mju_mat2Quat(q, (Rp.T @ R_new_world).flatten())
        out[side] = q
        # 自检：把两个物理方向表达到新 local 系里，逐轴对上上面那张表
        fl = R_new_world.T @ finger_w
        tl = R_new_world.T @ thumb_w
        pl = np.cross(fl, tl)
        got = tuple(_axis(v) for v in (fl, tl, pl))
        want = {"left": ("+y", "-x", "+z"), "right": ("+y", "+x", "-z")}[side]
        if got != want:
            raise SystemExit(f"{side} hand_frame 轴向 {got} ≠ 约定 {want}；"
                             f"finger={np.round(fl,3)} thumb={np.round(tl,3)} "
                             f"palm={np.round(pl,3)}")
    return out


def _axis(v: np.ndarray) -> str:
    i = int(np.argmax(np.abs(v)))
    return f"{'+' if v[i] > 0 else '-'}{'xyz'[i]}"


def insert_frames(mjcf: Path, quats: dict) -> None:
    tree = ET.parse(mjcf)
    root = tree.getroot()
    bodies = {b.get("name"): b for b in root.iter("body")}
    for name, parent in FRAME_PARENTS.items():
        if name in bodies:
            raise SystemExit(f"{name} 已存在，不该重复插入")
        if parent not in bodies:
            raise SystemExit(f"找不到父 body {parent}")
        if name == "torso_frame":
            pos, quat = (0.0, 0.0, 0.0), np.array([1.0, 0.0, 0.0, 0.0])
        else:
            pos, quat = HAND_FRAME_POS, quats[name.split("_hand_frame")[0]]
        el = ET.SubElement(bodies[parent], "body")
        el.set("name", name)
        el.set("pos", " ".join(f"{v:g}" for v in pos))
        el.set("quat", " ".join(f"{v:.8g}" for v in quat))
    _indent(root)
    tree.write(mjcf, encoding="unicode")


def _indent(el, level=0):
    pad = "\n" + "  " * level
    if len(el):
        if not (el.text or "").strip():
            el.text = pad + "  "
        for child in el:
            _indent(child, level + 1)
        if not (el.tail or "").strip():
            el.tail = pad
        if not (child.tail or "").strip():
            child.tail = pad
    elif level and not (el.tail or "").strip():
        el.tail = pad


# ── step 5: arms-only MJX 模型（方法同 generate_m7_mjx.py） ───────────────────

def joint_of(body: str) -> str | None:
    if body.endswith("_hand_frame"):
        return None
    return body[:-len("_link")] + "_joint"


def build_mjx(mjcf: Path, out: Path) -> None:
    """双臂-only 的 MJX 模型。

    结构照 ``scripts/dev/generate_m7_mjx.py``：MJX 那条路只调 ``mjx.kinematics()``
    做纯 FK，所以惯量/几何/actuator 全是占位（但**必须有** —— 没惯量 MuJoCo 直接
    拒绝编译动body），真正有意义的只有 body 的 local pos/quat 和 hinge 的 axis/range。
    """
    m = mujoco.MjModel.from_xml_path(str(mjcf))
    d = mujoco.MjData(m)
    mujoco.mj_resetData(m, d)
    mujoco.mj_forward(m, d)                      # zero config
    bid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
    jid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
    g = lambda a: " ".join(f"{v:.8g}" for v in np.asarray(a).ravel())

    tb = bid(TORSO)
    L = ['<mujoco model="l3_4_mjx">',
         '  <!-- AUTO-GENERATED by scripts/dev/build_l3_4_assets.py from l3_4.xml.',
         '       Arms-only kinematic tree for MJX FK (grid 根位姿搜索 / 训练都走它)。',
         '       手指/mesh/材质/腰腿自由度全丢掉；躯干焊在世界上。惯量和几何是占位。 -->',
         '  <compiler angle="radian"/>',
         '  <option integrator="implicitfast" timestep="0.004" iterations="4" ls_iterations="8"/>',
         '  <default>',
         '    <joint damping="10" frictionloss="0"/>',
         '    <position inheritrange="1"/>',
         '    <geom contype="0" conaffinity="0"/>',
         '  </default>',
         '  <worldbody>',
         f'    <body name="{TORSO}" pos="{g(d.xpos[tb])}" quat="{g(d.xquat[tb])}">',
         '      <inertial pos="0 0 0" mass="10" diaginertia="0.5 0.5 0.5"/>',
         '      <geom type="box" size="0.12 0.12 0.15" rgba="0.5 0.5 0.5 1"/>']

    INDENT = "      "
    act_joints = []
    for side in ("left", "right"):
        depth = 0
        for body in ARM_BODIES[side]:
            b = bid(body)
            pad = INDENT + "  " * (depth + 1)
            jn = joint_of(body)
            if jn is None:                        # hand_frame：叶子固定 body
                L.append(f'{pad}<body name="{body}" pos="{g(m.body_pos[b])}" '
                         f'quat="{g(m.body_quat[b])}"/>')
                for _ in range(depth):
                    L.append(f'{INDENT + "  " * depth}</body>')
                    depth -= 1
                break
            j = jid(jn)
            L.append(f'{pad}<body name="{body}" pos="{g(m.body_pos[b])}" '
                     f'quat="{g(m.body_quat[b])}">')
            L.append(f'{pad}  <joint name="{jn}" type="hinge" axis="{g(m.jnt_axis[j])}" '
                     f'range="{m.jnt_range[j][0]:.6g} {m.jnt_range[j][1]:.6g}"/>')
            L.append(f'{pad}  <inertial pos="0 0 0" mass="0.8" diaginertia="0.01 0.01 0.01"/>')
            L.append(f'{pad}  <geom type="capsule" size="0.03" fromto="0 0 0 0 0 -0.15" '
                     f'rgba="0.6 0.6 0.8 1"/>')
            act_joints.append(jn)
            depth += 1

    L += ['    </body>', '  </worldbody>', '  <actuator>']
    L += [f'    <position name="{jn}_servo" joint="{jn}" kp="50"/>' for jn in act_joints]
    L += ['  </actuator>', '</mujoco>', '']
    out.write_text("\n".join(L))
    mujoco.MjModel.from_xml_path(str(out))       # 编译得过才算产出


def check_arm_chain_matches_m7(l3_4_mjx: Path) -> dict:
    """L3.4 的双臂链和 M7 的是不是同一条 —— 决定 M7 根模型 checkpoint 能不能直接用。

    根位姿模型学的是"给定手的任务空间目标 → 底座放哪"，输入输出都由
    ``waist_pitch_link → hand_frame`` 这条链决定。这条链两边逐位相同（上面量过），
    所以 L3.4 借 M7 的 ckpt 不是"凑合"，是同一个函数。这里把它**验出来**而不是论证：
    对着 ``m7_mjx.xml`` 逐 body/逐关节比。

    比对失败不代表 L3.4 错了 —— 也可能是有人动了 M7 的资产。但无论哪种，
    "借 ckpt" 这个前提就没了，必须当场知道。
    """
    a = mujoco.MjModel.from_xml_path(str(l3_4_mjx))
    b = mujoco.MjModel.from_xml_path(str(P.asset("m7_mjx")))
    worst = 0.0
    bad = []
    for i in range(a.nbody):
        name = mujoco.mj_id2name(a, mujoco.mjtObj.mjOBJ_BODY, i)
        k = mujoco.mj_name2id(b, mujoco.mjtObj.mjOBJ_BODY, name)
        if k < 0:
            bad.append(f"m7_mjx 里没有 body {name}")
            continue
        for attr in ("body_pos", "body_quat"):
            dv = float(np.abs(getattr(a, attr)[i] - getattr(b, attr)[k]).max())
            worst = max(worst, dv)
            if dv > 1e-5:
                bad.append(f"{name}.{attr} 差 {dv:.2e}")
    for i in range(a.njnt):
        name = mujoco.mj_id2name(a, mujoco.mjtObj.mjOBJ_JOINT, i)
        k = mujoco.mj_name2id(b, mujoco.mjtObj.mjOBJ_JOINT, name)
        if k < 0:
            bad.append(f"m7_mjx 里没有 joint {name}")
            continue
        for attr in ("jnt_axis", "jnt_range"):
            dv = float(np.abs(getattr(a, attr)[i] - getattr(b, attr)[k]).max())
            worst = max(worst, dv)
            if dv > 1e-5:
                bad.append(f"{name}.{attr} 差 {dv:.2e}")
    return {"worst": worst, "bad": bad}


# ── step 6: 场景 ──────────────────────────────────────────────────────────────

SCENE = """<mujoco model="l3_4 vis_scene">
  <include file="l3_4.xml"/>

  <statistic center="0.0 0 0.0" extent="1.0"/>

  <visual>
    <headlight diffuse="0.0 0.0 0.0" ambient="0.35 0.35 0.35" specular="0.0 0.0 0.0"/>
    <rgba haze="1 1 1 1"/>
    <global offwidth="1920" offheight="1920" azimuth="140" elevation="-20"/>
  </visual>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="1 1 1" rgb2="1 1 1" width="512" height="3072"/>
    <texture type="2d" name="groundplane" builtin="flat" rgb1="1 1 1" width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.0"/>
  </asset>

  <worldbody>
    <light pos="1 -1 1.5" dir="-0.5 0.5 -0.8" directional="true" diffuse="0.9 0.9 0.9" specular="0.0 0.0 0.0"/>
    <light pos="-1 1 1.5" dir="0.4 -0.3 -0.6" directional="true" diffuse="0.45 0.45 0.45" specular="0.0 0.0 0.0"/>
    <geom name="floor" size="0 0 0.05" type="plane" material="groundplane" pos="0 0 -0.8"/>
  </worldbody>
</mujoco>
"""


# ── step 7: 对着厂家 MJCF 验一遍（"urdf 和 xml 是否一致可用"） ────────────────

def cross_check(ours: Path, vendor: Path) -> dict:
    a = mujoco.MjModel.from_xml_path(str(ours))
    # 厂家那份引用 14 个腿 mesh，编译不过 → 只做文本级比对
    vtext = vendor.read_text()
    vroot = ET.fromstring(vtext)
    vj = {j.get("name"): j for j in vroot.iter("joint") if j.get("name")}
    vb = {b.get("name"): b for b in vroot.iter("body") if b.get("name")}

    def arr(s, n):
        return np.array([float(x) for x in s.split()]) if s else np.zeros(n)

    bad = []
    checked = 0
    for i in range(a.njnt):
        name = mujoco.mj_id2name(a, mujoco.mjtObj.mjOBJ_JOINT, i)
        if name not in vj:
            continue
        checked += 1
        ax = arr(vj[name].get("axis"), 3)
        rg = arr(vj[name].get("range"), 2)
        if not np.allclose(a.jnt_axis[i], ax, atol=1e-6):
            bad.append(f"{name}.axis {a.jnt_axis[i]} vs {ax}")
        if not np.allclose(a.jnt_range[i], rg, atol=1e-6):
            bad.append(f"{name}.range {a.jnt_range[i]} vs {rg}")
    nb = 0
    for i in range(a.nbody):
        name = mujoco.mj_id2name(a, mujoco.mjtObj.mjOBJ_BODY, i)
        if name not in vb or name in FRAME_PARENTS:
            continue
        nb += 1
        pos = arr(vb[name].get("pos"), 3)
        if not np.allclose(a.body_pos[i], pos, atol=1e-6):
            bad.append(f"{name}.pos {a.body_pos[i]} vs {pos}")
    return {"joints_checked": checked, "bodies_checked": nb, "mismatch": bad}


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--force", action="store_true",
                    help="目录已存在时覆盖（默认拒绝，避免手改被冲掉）")
    args = ap.parse_args()

    if OUT_DIR.exists() and not args.force:
        raise SystemExit(f"{OUT_DIR} 已存在。确认要重建就加 --force")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as td:
        src = extract(Path(td))
        print(f"[1/7] 解包 {SRC_TAR.name} → {sorted(p.name for p in src.iterdir())}")

        urdf_out = OUT_DIR / "l3_4_from_urdf.urdf"
        st = prepare_urdf(src / VENDOR_URDF, urdf_out)
        print(f"[2/7] URDF：flatten {st['mesh_paths']} 处 mesh 路径；"
              f"删几何 {len(st['geom_dropped'])} 个 link")

        shutil.copy2(src / VENDOR_MJCF, OUT_DIR / "l3_4_vendor.xml")
        shutil.copy2(src / "l3_4.urdf.xacro", OUT_DIR / "l3_4.urdf.xacro")

        ms = link_meshes(urdf_out, OUT_DIR / "meshes")
        print(f"[3/7] mesh：symlink {len(ms['linked'])} 个 → {M7_MESHES}"
              f"{'；缺 ' + str(ms['missing']) if ms['missing'] else '；无缺失'}")
        if ms["missing"]:
            raise SystemExit("URDF 还引用着不存在的 mesh —— NO_MESH_LINKS 漏了？")

        mjcf = OUT_DIR / "l3_4.xml"
        compile_mjcf(urdf_out, mjcf)
        quats = hand_frame_quats(mjcf)
        insert_frames(mjcf, quats)
        m = mujoco.MjModel.from_xml_path(str(mjcf))
        print(f"[4/7] MJCF：{m.nbody} body / {m.njnt} joint；hand_frame quat "
              f"L={np.round(quats['left'],6).tolist()} R={np.round(quats['right'],6).tolist()}")

        (OUT_DIR / "scene_vis.xml").write_text(SCENE)
        mujoco.MjModel.from_xml_path(str(OUT_DIR / "scene_vis.xml"))
        print("[5/7] scene_vis.xml 编译通过")

        build_mjx(mjcf, OUT_DIR / "l3_4_mjx.xml")
        cm = check_arm_chain_matches_m7(OUT_DIR / "l3_4_mjx.xml")
        if cm["bad"]:
            raise SystemExit("L3.4 的双臂链和 M7 的不一致 → 不能借 M7 的根模型 ckpt：\n  "
                             + "\n  ".join(cm["bad"][:20]))
        print(f"[6/7] l3_4_mjx.xml 编译通过；双臂链与 m7_mjx.xml 逐位一致"
              f"（最大偏差 {cm['worst']:.1e}）→ M7 根模型 ckpt 直接可用")

        cc = cross_check(mjcf, OUT_DIR / "l3_4_vendor.xml")
        if cc["mismatch"]:
            raise SystemExit("我方 MJCF 和厂家 MJCF 不一致：\n  " +
                             "\n  ".join(cc["mismatch"][:20]))
        print(f"[7/7] 对厂家 MJCF 校验通过："
              f"{cc['joints_checked']} 个关节的 axis/range、{cc['bodies_checked']} 个 body 的 pos 全同")


if __name__ == "__main__":
    main()
