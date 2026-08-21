"""每台机器人一个 YAML 的参数入口。**这些数字在仓库里只有一份，就在 yaml 里。**

为什么搬出来
------------
IK 权重、碰撞过滤的进入阈值/推出余量、关节限位、静息（IK 种子）姿态，原来散在
``robots/<机器人>/ik_config.py``、``robots/<机器人>/config.py``、
``collision/arm_torso_filter.py``、``collision/dual_hand_filter.py``、
``collision/capsule_collision.py``、``collision/presets.py`` 六个文件的默认值里。
散着的直接后果是：想知道"这台机器人现在到底用的什么参数"得翻六个文件，而且没有
任何地方记着"这个数是标定出来的，那个数只是沿用默认值"。

格式参考 HandUMI（robonet-ai.github.io/handumi-sw）的机器人配置：**一台机器人一个
yaml，装 IK 权重 / 关节限位 / 静息姿态，外加一个 ``verified`` 标志位**说明这份参数
有没有被真的验证过。**只借格式，不借数值** —— 它那些数是给它自己的夹爪用的。

``verified`` 是什么意思（别读歪）
--------------------------------
``verified: true`` = **这组数字是拿我们自己的数据量出来/标定出来的**，改动它需要重新
标定。``verified: false`` = 没有专门用数据验证过，只是沿用默认值或者从资产里读出来的
——**不等于"这个数是错的"**，只是"这个数还没人拿数据钉过，别把它当结论引用"。

现在只有一组是 ``true``：A1 那次针对 grid 路线标定的碰撞参数
（``collision.arm_torso.routes.grid``，见 ``docs/VERIFICATION.md``）。

``source`` 写清出处，``pinned_by`` 写清哪个测试钉住了它 —— 关节限位属于
"``verified: false`` 但有权威出处（MJCF）且被测试钉死"这一类，两个字段合起来才说得清。

哪些参数**刻意不在** yaml 里
----------------------------
* **body 名字**（``torso_body`` / ``wrist_body`` / 骨骼父子链 / 指尖 link 列表）——
  那是**结构事实**，不是可调参数，唯一真相是 MJCF。放进 yaml 等于给同一个名字造第二份
  来源，而改 MJCF 漏改 yaml 不会报错（pytorch_kinematics 会建一条空链，IK 全帧失败但
  不抛异常）。判断线就是这个：**标定脚本能扫的量（= 构造参数）进 yaml，结构事实留在
  代码里挨着模型。**
* **根位姿模型的训练/采样参数**（``robots/*/sample_config.py``）—— 那些是训练超参，
  改了要重训，和"这台机器人是什么样"不是一类东西。
* 路径（在 ``configs/paths.yaml``）。

用法::

    from web2robot.robots.params import robot_params, values

    P = robot_params("m7")
    P["ik"]["joint_limits"]["left"]          # (7,2) list
    values(P["collision"]["arm_torso"]["defaults"])   # 去掉 verified/source 等元信息

    python -m web2robot.robots.params m7     # 打一张 verified 清单
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

from web2robot.paths import P as _PATHS

#: 元信息键：不是参数值，喂给构造函数之前要用 :func:`values` 剔掉。
META_KEYS = frozenset({"verified", "source", "pinned_by", "note", "unit", "order"})

_CACHE: Dict[str, Dict[str, Any]] = {}


def robot_params_path(robot: str) -> Path:
    return _PATHS.repo_root / "configs" / "robots" / f"{robot}.yaml"


def robot_params(robot: str) -> Dict[str, Any]:
    """读 ``configs/robots/<robot>.yaml``，返回**深拷贝**。

    深拷贝不是洁癖：调用方常常要把某个 group 当 kwargs 改一改（``arm_torso_preset``
    就明确承诺返回副本），共享 dict 会让"改了一处污染全局"这种 bug 只在第二次调用时
    才现形。缓存只存解析结果，每次调用拷一份出去。
    """
    if robot not in _CACHE:
        path = robot_params_path(robot)
        if not path.exists():
            known = sorted(p.stem for p in path.parent.glob("*.yaml")) \
                if path.parent.exists() else []
            raise FileNotFoundError(
                f"找不到机器人参数文件 {path}；现有的是 {known}")
        with open(path) as fh:
            cfg = yaml.safe_load(fh)
        if not isinstance(cfg, dict) or cfg.get("robot") != robot:
            raise ValueError(
                f"{path} 里的 robot: 字段是 {cfg.get('robot')!r}，"
                f"和文件名 {robot!r} 不一致 —— 复制文件时最容易漏改这一行")
        _CACHE[robot] = cfg
    return copy.deepcopy(_CACHE[robot])


def values(group: Dict[str, Any]) -> Dict[str, Any]:
    """一个 group 里的**参数值**（剔掉 :data:`META_KEYS`）。

    这样 ``Filter(**values(group))`` 就能直接用，而元信息（``verified`` 等）不会以
    ``TypeError: unexpected keyword argument`` 的形式在跑到一半时炸出来。
    """
    return {k: v for k, v in group.items() if k not in META_KEYS}


def verified_rows(robot: str) -> List[Tuple[str, Any, str]]:
    """``[(点分路径, verified, source)]``，按路径排序。给清单和测试用。"""
    rows: List[Tuple[str, Any, str]] = []

    def walk(node: Any, path: str) -> None:
        if not isinstance(node, dict):
            return
        if "verified" in node:
            rows.append((path, node["verified"], str(node.get("source", ""))))
        for k, v in node.items():
            walk(v, f"{path}.{k}" if path else str(k))

    walk(robot_params(robot), "")
    return sorted(rows)


def _main(argv: List[str]) -> int:
    if len(argv) != 2:
        print(f"用法: python -m web2robot.robots.params <robot>")
        return 2
    robot = argv[1]
    print(f"# {robot_params_path(robot)}")
    for path, ok, src in verified_rows(robot):
        mark = "✅ verified" if ok else "⬜ unverified"
        print(f"{mark}  {path}\n              {src}")
    return 0


if __name__ == "__main__":
    import sys
    raise SystemExit(_main(sys.argv))


__all__ = ["META_KEYS", "robot_params", "robot_params_path", "values",
           "verified_rows"]
