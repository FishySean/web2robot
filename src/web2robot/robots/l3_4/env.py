"""
Single L3.4 (rel3_4) humanoid MuJoCo environment.

L3.4 = 双 7-DoF 手臂 + 两只 12-DoF 五指手，挂在 3-DoF 腰 + 盆骨 + 两条 6-DoF 腿上。
**本阶段只做上肢**：腿 / 腰 / 颈全部锁死在 ``config.LOCKED_JOINTS`` 给的值上
（是锁死不是删除 —— MJCF 里那些 hinge 都还在，以后要开就把它们从那张表里拿掉）。

Action:  dict with optional keys "left" and "right", each (7,) joint angles [rad]
         for [shoulder_pitch, shoulder_roll, arm_yaw, elbow_pitch, elbow_yaw,
              wrist_pitch, wrist_roll].

Observation: dict with optional keys "left" and "right", each containing:
    - "pos":  (3,) wrist (hand_frame) position in world frame
    - "quat": (4,) hand_frame orientation quaternion (w, x, y, z) in world frame

## 和 M7 的关系：同构，但**不共用代码**

L3.4 的上肢和 M7 逐位同构（43 个同名关节 axis/range 全同、43 个同名 body 的
pos/quat 全同、``base_link → waist_pitch_link`` 的变换也相同），所以下面这些表和
``robots/m7/`` 里的一模一样。**这不是复制粘贴的疏忽，是刻意的**：一台机器人的定义
不该依赖另一台机器人的模块存在（``robots/l3_4/`` 里没有一处 ``import ...robots.m7``），
否则"删掉 M7"或"改 M7 的关节名"会隔着一台机器人炸。

代价是两份表可能悄悄漂。这一条由 ``tests/test_l3_4_robot.py`` 顶住：它不拿 M7 的表
当参照，而是拿**各自的 MJCF** 当唯一真相 —— 表里每个限位都必须等于
``l3_4.xml`` 里那个关节的 ``range``。真源在资产里，不在另一台机器人的代码里。

## MJCF 是生成的

``assets/robots/l3_4/`` 整个目录由 ``scripts/dev/build_l3_4_assets.py`` 从
``assets/robots/urdf.tar.gz``（厂家原包）生成，**别手改**。那个脚本的 docstring 里
写清了每一处改动和理由（含"包里一个 mesh 都没有"这件事：94 个零件和 M7 相同，直接
symlink 过去；14 个腿部 + 盆骨没有正确的 mesh，几何被删掉 → 渲出来腰以下是空的，
不影响上肢的任何数字）。

## 没有 <actuator>

和 M7 一样，MJCF 是 URDF 转的，只有 joint 没有 actuator，所以
``set_arm_joints`` / ``set_finger_joints`` 都 guard ``aid >= 0``，直接写 qpos。
纯运动学路径（set qpos → mj_forward → 读 xpos）够用。
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import mujoco
import numpy as np

from web2robot.paths import P

# ── robot description ─────────────────────────────────────────────────────────
_MJCF_PATH     = P.asset("l3_4_mjcf")
_MJCF_MJX_PATH = P.asset("l3_4_mjx")     # arms-only MJX model（grid 根位姿搜索用）
_SCENE_PATH    = P.asset("l3_4_scene")
_ROBOT_DIR     = _MJCF_PATH.parent

_ARM_JOINTS = {
    "left": [
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_arm_yaw_joint",
        "left_elbow_pitch_joint",
        "left_elbow_yaw_joint",
        "left_wrist_pitch_joint",
        "left_wrist_roll_joint",
    ],
    "right": [
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_arm_yaw_joint",
        "right_elbow_pitch_joint",
        "right_elbow_yaw_joint",
        "right_wrist_pitch_joint",
        "right_wrist_roll_joint",
    ],
}

_EE_BODY    = {"left": "left_hand_frame", "right": "right_hand_frame"}
# 两条手臂在 MJCF 树里的共同祖先。torso_frame 和它位置重合但是**叶子** body，
# 拿叶子当 IK 串链根会建出空链（M7 那边踩过，见 robots/m7/ik_config.py）。
_TORSO_BODY = "waist_pitch_link"

# ── 锁死的自由度（腿 12 + 腰 3 + 颈 2） ───────────────────────────────────────
# 值全是 0.0 = URDF 的零位（腿直立、腰颈正中）。为什么是"锁死"而不是"删除"：
# 删了以后想做下肢就得重造资产；锁死只是这张表的事，从表里拿掉一行就解锁一个自由度。
# env 在 reset() 里把它们写进 qpos 并在每次 forward 前保持 —— 不是靠"没人去写它们"。
_LOCKED_JOINTS: dict[str, float] = {
    # 腰：3 DoF
    "waist_yaw_joint":   0.0,
    "waist_roll_joint":  0.0,
    "waist_pitch_joint": 0.0,
    # 颈：2 DoF
    "neck_yaw_joint":    0.0,
    "neck_pitch_joint":  0.0,
    # 腿：每条 6 DoF
    **{f"{side}_{j}_joint": 0.0
       for side in ("left", "right")
       for j in ("hip_roll", "hip_yaw", "hip_pitch", "knee",
                 "ankle_pitch", "ankle_roll")},
}

# 重定向器短名 → L3.4 MJCF 手指关节名（**不带** left_/right_ 前缀，env 每次调用时再贴）。
# 和 M7 同一只手（关节名逐字相同）；"xhand" 只是厂家 URDF 里 mesh 路径的目录名，
# 不是另一款手 —— 见 hand_mapping.py 的说明。
_FINGER_JOINT_NAMES = {
    "thumb_bend":  "hand_thumb_bend_joint",
    "thumb_rota1": "hand_thumb_rota_joint1",
    "thumb_rota2": "hand_thumb_rota_joint2",
    "index_abd":   "hand_index_bend_joint",
    "index_mcp":   "hand_index_joint1",
    "index_pip":   "hand_index_joint2",
    "middle_mcp":  "hand_mid_joint1",
    "middle_pip":  "hand_mid_joint2",
    "ring_mcp":    "hand_ring_joint1",
    "ring_pip":    "hand_ring_joint2",
    "pinky_mcp":   "hand_pinky_joint1",
    "pinky_pip":   "hand_pinky_joint2",
}


class L34Env:
    """Single-instance dual-arm L3.4 simulation environment.

    实现的是上游 ``sim.base_env.BaseEnv`` 的接口（reset / set_arm_joints /
    step_joints / get_wrist_pose），但**不继承它** —— 理由和 M7 那边一样：机器人定义
    要能被别的重定向框架直接拿去用，继承上游 ABC 就焊死在 EgoInfinity 的目录结构上。
    保护由 ``tests/test_l3_4_robot.py::TestBaseEnvConformance`` 补（测试里允许
    import 上游，``src/`` 里不允许）。
    """

    def __init__(
        self,
        mjcf_path: Optional[Path] = None,
        start_config: Optional[dict] = None,
    ):
        mjcf_path = Path(mjcf_path) if mjcf_path is not None else _SCENE_PATH
        self.model = mujoco.MjModel.from_xml_path(str(mjcf_path))
        self.data  = mujoco.MjData(self.model)

        self._start_config = start_config

        self._joint_ids:    dict[str, list[int]] = {}
        self._actuator_ids: dict[str, list[int]] = {}
        self._body_ids:     dict[str, int]       = {}

        for side, joints in _ARM_JOINTS.items():
            self._joint_ids[side] = [
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, j)
                for j in joints
            ]
            self._actuator_ids[side] = [
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, j)
                for j in joints
            ]
            self._body_ids[side] = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, _EE_BODY[side]
            )

        self._torso_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, _TORSO_BODY
        )

        # 锁死自由度的 (qpos 地址, 值)。MJX 那个只有双臂的模型里这些关节不存在，
        # 所以缺了不报错 —— 但**存在的必须被写住**。
        self._locked: list[tuple[int, float]] = []
        for jname, val in _LOCKED_JOINTS.items():
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            if jid >= 0:
                self._locked.append((int(self.model.jnt_qposadr[jid]), float(val)))

        self._joint_limits: dict[str, np.ndarray] = {}
        for side, jids in self._joint_ids.items():
            self._joint_limits[side] = np.array(
                [self.model.jnt_range[jid] for jid in jids]
            )

        self.reset()

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def reset(self) -> dict:
        mujoco.mj_resetData(self.model, self.data)
        key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "home")
        if key_id >= 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, key_id)
        self._apply_locked()
        mujoco.mj_forward(self.model, self.data)
        if self._start_config is not None:
            for side, q in self._start_config.items():
                self.set_arm_joints(side, np.asarray(q, dtype=np.float64))
        return self._get_obs()

    def _apply_locked(self) -> None:
        """把腿/腰/颈按住。每次 mj_forward 之前调 —— 一次 reset 里写死不够：
        上层（碰撞过滤、渲染）会直接改 ``data.qpos`` 再 forward，按住才是真锁。"""
        for adr, val in self._locked:
            self.data.qpos[adr] = val

    # ── action ────────────────────────────────────────────────────────────────

    def set_arm_joints(self, side: str, q: np.ndarray):
        assert q.shape == (7,), f"Expected (7,) joint angles, got {q.shape}"
        for i, aid in enumerate(self._actuator_ids[side]):
            if aid >= 0:                        # URDF 转的 MJCF 里没有 actuator
                self.data.ctrl[aid] = q[i]
        for i, jid in enumerate(self._joint_ids[side]):
            self.data.qpos[self.model.jnt_qposadr[jid]] = q[i]
        self._apply_locked()
        mujoco.mj_forward(self.model, self.data)

    def set_finger_joints(self, q: np.ndarray, joint_names: list[str]):
        """Set L3.4 finger joints from retargeter output (left_/right_ prefixed short names)."""
        for i, name in enumerate(joint_names):
            if name.startswith("left_"):
                side, short = "left", name[len("left_"):]
            elif name.startswith("right_"):
                side, short = "right", name[len("right_"):]
            else:
                continue
            if short.endswith("_joint"):
                short = short[:-len("_joint")]
            suffix = _FINGER_JOINT_NAMES.get(short)
            if suffix is None:
                continue
            mjcf_name = f"{side}_{suffix}"
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, mjcf_name)
            if jid < 0:
                continue
            adr = self.model.jnt_qposadr[jid]
            lo, hi = self.model.jnt_range[jid]
            self.data.qpos[adr] = float(np.clip(q[i], lo, hi))
            aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, mjcf_name)
            if aid >= 0:
                self.data.ctrl[aid] = float(np.clip(q[i], lo, hi))
        self._apply_locked()
        mujoco.mj_forward(self.model, self.data)

    def step_joints(self, action: dict) -> dict:
        for side, q in action.items():
            self.set_arm_joints(side, np.asarray(q, dtype=np.float64))
        return self._get_obs()

    # ── observation ───────────────────────────────────────────────────────────

    def _get_obs(self) -> dict:
        return {
            side: {"pos": self.data.xpos[bid].copy(), "quat": self.data.xquat[bid].copy()}
            for side, bid in self._body_ids.items()
        }

    def get_wrist_pose(self, side: str) -> tuple[np.ndarray, np.ndarray]:
        bid = self._body_ids[side]
        return self.data.xpos[bid].copy(), self.data.xquat[bid].copy()
