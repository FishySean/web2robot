"""`docs/PROJECT_LAYOUT.md` 不许和真实目录结构脱钩。

这份布局说明的用途是"不翻文件夹就能定位到东西"，所以它一旦过期就是**误导**——
比只写"看代码"更糟，因为读的人会信。文档烂掉的方式很具体，这里各钉一条：

1. 新建了顶层目录但忘了写进文档（最常发生）；
2. 文档里的路径写错或者东西被搬走了；
3. `outputs/` 下多了一类产物，落点规律没记（于是下一个人又随手建目录）。

`outputs/` `data/` 不进 git，新克隆本来就没有，所以对它们**只查文档有没有记，
不查文件在不在**。跑法::

    envs/rt_env/bin/python -m unittest tests.test_docs_layout -v
"""
import re
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DOC = REPO / "docs" / "PROJECT_LAYOUT.md"

#: 这些顶层目录的内容进 git，文档里提到的路径必须真实存在
TRACKED_ROOTS = ("src/", "tests/", "scripts/", "configs/", "docs/", "assets/",
                 "evidence/", "envs/", "external/patches/")
#: 这些不进 git，只查文档记没记
UNTRACKED_ROOTS = ("outputs/", "data/")


def _expand_braces(p: str):
    """``bench_{ABF12,MC4}.npz`` → 两条真实路径。文档里为了紧凑会这么写。"""
    m = re.search(r"\{([^{}]*)\}", p)
    if not m:
        return [p]
    out = []
    for alt in m.group(1).split(","):
        out += _expand_braces(p[:m.start()] + alt.strip() + p[m.end():])
    return out


def _candidate_paths(text: str):
    """从文档里抠出"看起来像仓库内路径"的字符串。

    两种写法都要认：反引号里的 ``src/foo.py``，和 markdown 链接的
    ``[x](../evidence/README.md)``（链接是相对 docs/ 写的，为了在 GitLab 和
    VSCode 预览里都能点）。
    """
    raw = re.findall(r"`([^`\n]+)`", text) + re.findall(r"\]\(([^)\s]+)\)", text)
    paths = set()
    for s in raw:
        s = s.strip().rstrip(".,;:")
        if s.startswith("../"):          # markdown 链接，相对 docs/
            s = "docs/" + s
            s = str(Path(s).as_posix()).replace("docs/../", "")
        if "<" in s or ">" in s or "*" in s or " " in s:
            continue                     # 占位符 / 命令行片段，不是具体路径
        if not s.startswith(TRACKED_ROOTS + UNTRACKED_ROOTS):
            continue
        paths.update(_expand_braces(s))
    return sorted(paths)


class TestLayoutDocMatchesReality(unittest.TestCase):
    def setUp(self):
        self.assertTrue(DOC.exists(), f"布局说明不见了：{DOC}")
        self.text = DOC.read_text()

    def test_every_top_level_directory_is_documented(self):
        """新建顶层目录必须写进文档 —— 这是这份文档最容易烂的一处。"""
        missing = []
        for d in sorted(REPO.iterdir()):
            if not d.is_dir() or d.name.startswith(".") or d.name == "__pycache__":
                continue
            if f"{d.name}/" not in self.text:
                missing.append(d.name)
        self.assertEqual(
            missing, [],
            "这些顶层目录在 docs/PROJECT_LAYOUT.md 里没有说明（新建目录请补上，"
            "否则下一个人只能靠翻文件夹猜）：" + ", ".join(missing))

    def test_documented_paths_that_should_be_in_git_really_exist(self):
        """文档里指向 src/ evidence/ 之类的路径必须真实存在。

        ``tests/x.py::TestY`` 这种写法（文档里用来指某一组用例）连类名一起验 ——
        只验文件存在的话，测试类被改名或删掉，文档就会指向一个不存在的东西。
        """
        checked, missing = [], []
        for rel in _candidate_paths(self.text):
            if not rel.startswith(TRACKED_ROOTS):
                continue
            checked.append(rel)
            file_part, _, cls = rel.partition("::")
            target = REPO / file_part
            if not target.exists():
                missing.append(rel)
            elif cls and f"class {cls}" not in target.read_text():
                missing.append(f"{file_part} 里没有 class {cls}")
        # 负面控制：抠不出路径的话上面这个循环等于没跑，测试就是空过
        self.assertGreater(len(checked), 15,
                           f"只从文档里抠出 {len(checked)} 条路径，抠取逻辑大概坏了")
        self.assertEqual(missing, [],
                         "文档里这些路径不存在（东西被搬走了？）：" + ", ".join(missing))

    def test_every_outputs_subdirectory_has_a_documented_landing_rule(self):
        """`outputs/` 下多一类产物，落点规律就要记 —— 否则下一个人又随手建目录。

        这里不查文件存在性（`outputs/` 不进 git），查的是反向：**现实里有的，
        文档里得有名字**。
        """
        outputs = REPO / "outputs"
        if not outputs.is_dir():
            self.skipTest("这份克隆里没有 outputs/，没什么可核对的")
        undocumented = [d.name for d in sorted(outputs.iterdir())
                        if d.is_dir() and f"outputs/{d.name}" not in self.text]
        self.assertEqual(
            undocumented, [],
            "outputs/ 下这几类产物在布局文档里没有落点说明："
            + ", ".join(undocumented))

    def test_the_key_evidence_is_pointed_at_by_full_path(self):
        """论文核心材料必须以完整路径出现 —— 这份文档存在的主要理由就是它。

        断言的是"文档里指得到"，而 `tests/test_depth_benchmark.py` 断言"那些数还对"。
        两件事分开：路径对但数被改了，或者数对但没人找得到，都是坏的。
        """
        for needle in ("evidence/depth_benchmark_ho3d/",
                       "outputs/viz/wilor_depth_modes.mp4",
                       "external/patches/README.md",
                       "configs/paths.yaml"):
            with self.subTest(needle=needle):
                self.assertIn(needle, self.text)

    def test_readme_points_at_this_document(self):
        """入口只能有一个：README 必须指过来，不然这份文档没人会发现。"""
        self.assertIn("docs/PROJECT_LAYOUT.md", (REPO / "README.md").read_text())


if __name__ == "__main__":
    unittest.main(verbosity=2)
