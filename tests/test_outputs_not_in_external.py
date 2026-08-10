"""产物不许写进 external/ —— 和"src/ 里不许有绝对路径字面量"同级的一条不变量。

为什么需要它（都是量出来的，不是洁癖）：

1. **上游 ``test.py`` 的 ``--out`` 默认值是 ``<clip_parent>/<robot>/``** ——
   把产物写在输入素材旁边。
2. 我们的薄壳**必须** ``cd`` 到上游 ``retarget/``（它的 config / checkpoint
   路径都是相对自己算的），于是任何相对的 ``--out`` 也一起落进 ``external/``。

两件事叠起来，实测结果是 ``external/EgoInfinity/retarget/`` 下攒了 408 MB、
243 个 mp4/npz，而上游 git 只跟踪其中 1 个 —— 其余全是我们跑的；同期
``outputs/`` 里只有一个目录。危害不只是乱：

- ``external/`` 是第三方 checkout，一次 ``git clean -xdf`` 或重新 clone
  就把结果全带走（这正是这次重构最初的动因）；
- 产物和输入素材混在同一棵树里之后，"哪份是官方素材、哪份是我们跑的"
  只能靠 mtime 猜。

所以判据写成代码（``P.check_output_dir``）+ 这个测试，而不是写在 README 里。
"""
import subprocess
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from web2robot.paths import P  # noqa: E402


class TestCheckOutputDir(unittest.TestCase):
    """``P.check_output_dir`` 本身的行为。"""

    def test_rejects_path_inside_external(self):
        inside = P.root("egoinfinity") / "retarget" / "runs" / "m7" / "whatever"
        with self.assertRaises(SystemExit) as cm:
            P.check_output_dir(inside)
        self.assertIn("第三方仓库", str(cm.exception))

    def test_rejects_the_upstream_default_out(self):
        """上游默认值 <clip_parent>/<robot> 必须被拦住 —— 它就是 408 MB 的来源。"""
        default_out = P.root("egoinfinity_clips") / "fill_jar" / "m7"
        with self.assertRaises(SystemExit):
            P.check_output_dir(default_out)

    def test_accepts_outputs_and_tmp(self):
        for ok in (REPO / "outputs" / "retarget" / "fill_jar", Path("/tmp/x")):
            self.assertEqual(P.check_output_dir(ok), ok)

    def test_relative_path_resolved_against_cwd_not_upstream(self):
        """相对路径按当前 cwd 解析。

        这一条钉的是薄壳里那半个修复：``s4_retarget.sh`` 在 ``cd`` 到上游**之前**
        把 ``--out`` 转成绝对路径。要是漏了这一步，``--out myout`` 就会变成
        ``external/EgoInfinity/retarget/myout``。
        """
        got = P.check_output_dir("myout/fill_jar")
        self.assertEqual(got, Path.cwd() / "myout" / "fill_jar")


class TestWrapperOutHandling(unittest.TestCase):
    """薄壳 ``s4_retarget.sh`` 传给上游的 ``--out`` 到底是什么。

    做法是把脚本里 ``cd "$UPSTREAM"`` 那行换成"打印 args 后退出"，这样不用真跑
    重定向（要几分钟 + GPU）就能验参数拼装。测的是意图：**没给 --out 时，
    上游那个"写在素材旁边"的默认值一定被顶掉。**
    """

    @classmethod
    def setUpClass(cls):
        src = (REPO / "scripts" / "s4_retarget.sh").read_text()
        assert 'cd "$UPSTREAM"' in src, "薄壳结构变了，这个测试要跟着改"
        cls.dry = src.replace('cd "$UPSTREAM"',
                              'printf "%s\\n" "${args[@]}"; exit 0')
        # 必须放在 scripts/ 下：脚本用 BASH_SOURCE 往上一层算 ROOT
        cls.path = REPO / "scripts" / "_test_s4_dry.sh"
        cls.path.write_text(cls.dry)

    @classmethod
    def tearDownClass(cls):
        cls.path.unlink(missing_ok=True)

    def _run(self, *argv, cwd="/tmp"):
        out = subprocess.run(["bash", str(self.path), *argv], cwd=cwd,
                             capture_output=True, text=True, timeout=60)
        self.assertEqual(out.returncode, 0, out.stderr)
        return out.stdout.split()

    def test_injects_out_when_missing(self):
        args = self._run("examples/fill_jar", "--robot", "m7")
        self.assertIn("--out", args)
        out = Path(args[args.index("--out") + 1])
        self.assertEqual(out, REPO / "outputs" / "retarget" / "fill_jar")
        P.check_output_dir(out)          # 不抛就说明落点合法

    def test_relative_out_resolved_against_caller_cwd(self):
        args = self._run("examples/fill_jar", "--out", "myout", cwd="/tmp")
        self.assertEqual(args[args.index("--out") + 1], "/tmp/myout")

    def test_equals_form_also_resolved(self):
        args = self._run("examples/fill_jar", "--out=myout", cwd="/tmp")
        self.assertEqual(args[args.index("--out") + 1], "/tmp/myout")

    def test_no_preview_added_once(self):
        self.assertEqual(self._run("examples/fill_jar").count("--no-preview"), 1)
        self.assertEqual(
            self._run("examples/fill_jar", "--no-preview").count("--no-preview"), 1)


class TestNoOursOutputsLeftInExternal(unittest.TestCase):
    """``external/`` 下不该再有我们跑出来的 run 目录。

    判据不是"数一数 mp4"，而是**上游 git 认不认**：凡是含
    ``robot_sim.mp4`` / ``trajectory.npz`` 这类产物标志文件、又不被上游 git
    跟踪的目录，就是我们的产物躺在别人家里。
    """

    OUT_MARK = {"robot_sim.mp4", "trajectory.npz", "metrics.npz",
                "root_frames.npz", "input_viz.mp4"}

    def test_no_untracked_run_dirs_under_upstream(self):
        up = P.root("egoinfinity") / "retarget"
        tracked = set(subprocess.run(
            ["git", "ls-files"], cwd=up, capture_output=True, text=True
        ).stdout.split())
        offenders = []
        for base in ("examples", "runs"):
            root = up / base
            if not root.is_dir():
                continue
            for d in root.rglob("*"):
                if not d.is_dir():
                    continue
                names = {p.name for p in d.iterdir() if p.is_file()}
                hits = names & self.OUT_MARK
                if not hits:
                    continue
                rel = d.relative_to(up)
                if not any(t.startswith(str(rel) + "/") for t in tracked):
                    offenders.append(str(rel))
        self.assertEqual(
            sorted(offenders), [],
            "这些产物目录还躺在第三方 checkout 里，应该在 outputs/ 下：\n  "
            + "\n  ".join(sorted(offenders)))


if __name__ == "__main__":
    unittest.main()
