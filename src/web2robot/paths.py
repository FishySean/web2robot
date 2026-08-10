"""路径解析 —— 全工程唯一允许出现绝对路径的模块。

用法::

    from web2robot.paths import P

    P.repo_root                      # 仓库根
    P.weights("hand_detector")       # 第一个存在的候选，都不存在返回 None
    P.asset("m7_mjcf")               # 缺失直接报错（资产缺了应该早失败）
    P.env("retarget")                # venv 的 python 解释器

为什么要有这一层：重构前 40 个 .py 里散着 ``/mnt/vlm/fanshaoheng`` 字面量，
换机器或搬目录就全碎。现在只有 ``configs/paths.yaml`` 一个来源。

根目录的确定顺序：环境变量 ``WEB2ROBOT_ROOT`` → 本文件往上第三层。
装成 editable 包（``pip install -e .``）后 ``src/web2robot/paths.py`` 的
上三层就是仓库根，所以正常情况不需要设环境变量。
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml


def _find_root() -> Path:
    env = os.environ.get("WEB2ROBOT_ROOT")
    if env:
        return Path(env).resolve()
    # src/web2robot/paths.py -> src/web2robot -> src -> repo root
    return Path(__file__).resolve().parents[2]


class Paths:
    """configs/paths.yaml 的只读视图。

    路径一律相对仓库根解析；配置里写绝对路径也支持（``Path.__truediv__``
    遇到绝对路径会直接采用它），方便临时指向别处的数据。
    """

    def __init__(self, root: Optional[Path] = None, config: Optional[Path] = None):
        self.repo_root = (root or _find_root()).resolve()
        self._cfg_path = config or self.repo_root / "configs" / "paths.yaml"
        if not self._cfg_path.exists():
            raise FileNotFoundError(
                f"找不到 {self._cfg_path}；如果仓库不在 "
                f"{self.repo_root} 请设 WEB2ROBOT_ROOT")
        with open(self._cfg_path) as fh:
            self._cfg: Dict[str, Any] = yaml.safe_load(fh)

    # ---- 底层 ----------------------------------------------------------
    def _abs(self, rel: str, follow_symlinks: bool = True) -> Path:
        """相对仓库根转绝对路径。

        ``follow_symlinks=False`` 是给 venv 解释器用的，**不是洁癖**：
        venv 的 ``bin/python`` 本身就是指向基础环境的 symlink，隔离靠的是
        ``pyvenv.cfg`` 所在的目录。跟着 symlink 走会掉回基础环境，包完全是
        另一套 —— 实测 ``envs/rt_env/bin/python`` 的 sys.prefix 是
        ``web2robot/envs/rt_env``（有 ultralytics），resolve 之后变成
        ``gs3dgs_env``（没有 ultralytics）。这种错会以莫名其妙的
        ModuleNotFoundError 形式出现，很难查。
        """
        p = self.repo_root / rel
        return p.resolve() if follow_symlinks else Path(os.path.normpath(p))

    def _lookup(self, section: str, key: str) -> Union[str, List[str]]:
        try:
            return self._cfg[section][key]
        except KeyError:
            keys = sorted(self._cfg.get(section, {}) or {})
            raise KeyError(
                f"paths.yaml 里 {section}: 下没有 '{key}'；有的是 {keys}") from None

    def _candidates(self, section: str, key: str,
                    follow_symlinks: bool = True) -> List[Path]:
        v = self._lookup(section, key)
        return [self._abs(c, follow_symlinks)
                for c in ([v] if isinstance(v, str) else v)]

    def _first_existing(self, section: str, key: str) -> Optional[Path]:
        """候选列表里第一个真实存在的路径；全都不存在返回 None。

        返回 None 而不是抛异常，是因为权重缺失在本工程里是"信号不可用"
        而不是"判为不合格"（质检那边靠这个区分 unknown 和 reject）。
        """
        for p in self._candidates(section, key):
            if p.exists():
                return p
        return None

    def _required(self, section: str, key: str,
                  follow_symlinks: bool = True) -> Path:
        cands = self._candidates(section, key, follow_symlinks)
        for p in cands:
            if p.exists():
                return p
        raise FileNotFoundError(
            f"{section}.{key} 指向的路径都不存在: "
            + ", ".join(str(c) for c in cands))

    # ---- 对外 ----------------------------------------------------------
    def root(self, key: str) -> Path:
        """第三方仓库/归档目录。"""
        return self._required("roots", key)

    def env(self, key: str) -> Path:
        """venv 的 python 解释器。

        必须用绝对路径调用：这台共享机器上 ``conda activate`` 不生效。
        故意**不解析 symlink** —— 见 ``_abs`` 的说明，解析会掉回基础环境。
        """
        return self._required("envs", key, follow_symlinks=False)

    def weights(self, key: str) -> Optional[Path]:
        """模型权重；缺失返回 None（调用方应据此报 unknown，而非 reject）。"""
        return self._first_existing("weights", key)

    def weight_candidates(self, key: str) -> List[Path]:
        """权重的全部候选路径（存在与否都返回），用于报错信息里说清找过哪里。"""
        return self._candidates("weights", key)

    def asset(self, key: str) -> Path:
        """机器人 MJCF/URDF/mesh；缺失直接报错。"""
        return self._required("assets", key)

    def data(self, key: str) -> Path:
        """数据/输出目录；不存在则创建（输出目录第一次跑总是不存在）。"""
        p = self._abs(str(self._lookup("data", key)))
        p.mkdir(parents=True, exist_ok=True)
        return p

    def check_output_dir(self, path: Union[str, Path]) -> Path:
        """校验一个产物目录：解析成绝对路径，并拒绝落在 ``external/`` 里面。

        为什么要有这条硬规矩：**上游 ``test.py`` 的 ``--out`` 默认值是
        ``<clip_parent>/<robot>/``** —— 也就是"把产物写在输入素材旁边"。而我们的
        薄壳必须 ``cd`` 到上游目录（它的 config / checkpoint 路径都是相对自己算的），
        于是任何相对的 ``--out`` 也一起落进 ``external/``。两件事叠起来的结果是实测
        出来的：`external/EgoInfinity/retarget/` 下攒了 408 MB、243 个 mp4/npz，
        其中上游 git 只跟踪 1 个 —— 其余全是我们跑的，而 ``outputs/`` 几乎是空的。

        这很危险，不只是乱：``external/`` 是第三方 checkout，一次 ``git clean -xdf``
        或重新 clone 就把我们的结果全带走；而且产物和输入素材混在一起之后，
        "哪份是官方素材、哪份是我们跑的"只能靠 mtime 猜。

        所以这条和"``src/`` 里不许有绝对路径字面量"同级：写成代码里的检查 + 测试，
        而不是写在文档里 —— 约定会被下一次赶工悄悄破掉，检查不会。
        """
        p = Path(path)
        p = p if p.is_absolute() else (Path.cwd() / p)
        p = Path(os.path.normpath(p))
        for key in ("egoinfinity", "hawor"):
            try:
                ext = self.root(key)
            except FileNotFoundError:
                continue
            if p == ext or str(p).startswith(str(ext) + os.sep):
                raise SystemExit(
                    f"产物目录不能落在第三方仓库里：{p}\n"
                    f"  它在 {ext} 下面 —— external/ 是第三方 checkout，"
                    f"一次 git clean 就会把结果带走。\n"
                    f"  请写成 outputs/ 下的路径，或给绝对路径（/tmp 也行）。")
        return p

    def torch_home(self) -> Path:
        """torchvision 权重缓存目录。

        共享机器上不要用默认的 ``~/.cache``：别人也在用同一个 HOME。
        """
        p = self._abs(str(self._lookup("weights", "torch_home")))
        p.mkdir(parents=True, exist_ok=True)
        return p

    def __repr__(self) -> str:      # pragma: no cover
        return f"Paths(root={self.repo_root})"


P = Paths()
