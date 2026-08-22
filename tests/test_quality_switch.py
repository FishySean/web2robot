"""质检 / 路由两个模块开关（``--quality_gate`` / ``--routing``，2026-08-21）。

这个测试守的东西按重要性排序：

1. **默认跑法什么都没变。** 两个开关的默认值都是 ``builtin``，``builtin`` 分支里
   ``_route`` 必须逐字返回 ``labels.suggest`` 的原结果 —— 开关是"要不要调用"，
   不是"换一套判断"。端到端的凭据是 ``scripts/dev/check_quality_switch_bytes.sh``
   （真跑 10 段，判决字段零容忍比对），这里守的是单元层面。
2. **skip 就是真的不做。** ``quality_gate=skip`` 时把 probe / run_pose /
   detect_hands / 光流 / ffmpeg 全都换成"一调就炸"，跑完必须不炸 —— 这比读一遍
   代码确认"没调"要硬。判决写 ``skipped`` 而不是 ``accept``：没量过的东西不能算通过。
3. **routing=skip 只少路由字段。** 拿桩件跑完整条判决路径，逐字段比对
   builtin / skip 两份报告，只允许 ``suggested_route`` / ``route_rationale``
   两项不同 —— ``verdict`` / ``reasons`` / ``signals`` 等其余每一项都必须一模一样。
   （``reasons`` 里**故意不放** ``routing_skipped`` 这种记账码：它的契约是"没通过的
   检查，最要紧的在前"，实测塞进去会得到 ``['routing_skipped', 'no_person']``，
   把真正的原因挤到第二位。）
4. ``labels.suggest`` 在 pipeline.py 里只有一个调用点（在 ``_route`` 里）。之前
   有两处（提前退出那条和正常判决那条），漏掉一处就会出现"说了不算路由、被拒的
   片段却还带着路线"这种自相矛盾的输出。用 ast 钉死，不靠人记得。
5. 没有 ``external`` 这一档 —— 现在没有对接对象，先留名字会有人去实现它。

跑法::

    envs/rt_env/bin/python -m unittest tests.test_quality_switch -v
"""
import ast
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from web2robot.common import video_io                                # noqa: E402
from web2robot.quality import cli, hand_gate, motion, pipeline, pose_gate  # noqa: E402
from web2robot.quality.config import GATE_MODES, ROUTING_MODES, QCConfig  # noqa: E402
from web2robot.quality.schema import REASONS, Verdict, ViewClass     # noqa: E402
from web2robot.routing import labels                                 # noqa: E402

PIPELINE_SRC = Path(pipeline.__file__).read_text()


def _texture(h=240, w=320, seed=0):
    """低频纹理图：goodFeaturesToTrack 在这种块状图案上有角点，判据才有值。"""
    import cv2
    rng = np.random.default_rng(seed)
    small = rng.integers(0, 255, size=(h // 8, w // 8, 3), dtype=np.uint8)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


def _boom(*a, **k):
    raise AssertionError("quality_gate=skip 时这一步不该被调用")


def _stub_pose_frames(n):
    """构造"整段都框全了"的姿态结果 -> THIRD_PERSON_BODY -> 有非空建议路线。

    必须是非空路线，否则第 3 条测试（只少路由字段）会在两边都是 None 的情况下
    空转通过。
    """
    return [pose_gate.PoseFrame(frame_idx=i, found=True, det=0.9,
                                box=np.array([10.0, 10.0, 100.0, 200.0]),
                                n_wrist=2, n_elbow=2, n_torso=2, n_head=1,
                                framing_ok=True, box_frac=0.2)
            for i in range(n)]


class _StubStages:
    """把所有要读文件 / 要模型的步骤换成常量，剩下的判决逻辑跑真的。

    换掉的是"测量"，没换"判断"：aggregate / classify_framing / 判决级联 /
    _route 全是原代码。
    """

    def __init__(self, n_frames=16, cuts=()):
        self.frames = [(i * 10, _texture(seed=i)) for i in range(n_frames)]
        self.pfs = _stub_pose_frames(n_frames)
        self.cuts = list(cuts)

    def __enter__(self):
        info = video_io.VideoInfo("stub.mp4", True, 320, 240, 30.0, 900, 30.0)
        self._p = [
            mock.patch.object(video_io, "probe", lambda p: info),
            mock.patch.object(video_io, "sample_frames",
                              lambda p, n, lo, hi: self.frames),
            mock.patch.object(video_io, "sample_pairs",
                              lambda p, n, lo, hi: []),
            mock.patch.object(pose_gate, "run_pose",
                              lambda frames, cfg, **k: self.pfs),
            # 手部检测器"装不上" -> detect_hands 自己返回 None，走的是真代码
            mock.patch.object(hand_gate, "get_hand_model",
                              lambda w, d: (None, "cpu")),
            mock.patch.object(motion, "detect_shot_cuts",
                              lambda *a, **k: self.cuts),
        ]
        for p in self._p:
            p.start()
        return self

    def __exit__(self, *e):
        for p in self._p:
            p.stop()
        return False


class TestModeVocabulary(unittest.TestCase):
    def test_defaults_are_builtin(self):
        cfg = QCConfig()
        self.assertEqual(cfg.quality_gate, "builtin")
        self.assertEqual(cfg.routing, "builtin")

    def test_no_external_mode_yet(self):
        # 用户明确要求现在不加 external：没有对接对象。
        self.assertEqual(GATE_MODES, ("builtin", "skip"))
        self.assertEqual(ROUTING_MODES, ("builtin", "skip"))

    def test_bad_mode_raises(self):
        for kw in (dict(quality_gate="external"), dict(quality_gate="off"),
                   dict(routing="external"), dict(routing="builtin_")):
            with self.assertRaises(ValueError):
                QCConfig(**kw)

    def test_skipped_is_not_accept(self):
        self.assertEqual(Verdict.SKIPPED.value, "skipped")
        self.assertNotEqual(Verdict.SKIPPED.value, Verdict.ACCEPT.value)

    def test_reason_codes_documented(self):
        self.assertIn("quality_gate_skipped", REASONS)

    def test_routing_skip_is_not_a_reason_code(self):
        """关掉路由不是"一项检查没通过"，不许挤进 reasons。

        实测过一版塞进去的：``['routing_skipped', 'no_person']`` —— reasons 的契约
        是"最要紧的在前"，记账码放在第一位会把真正的原因挤下去。
        """
        self.assertNotIn("routing_skipped", REASONS)
        self.assertNotIn('add_reason("routing_skipped")', PIPELINE_SRC)

    def test_cli_choices_come_from_config(self):
        """CLI 的 choices 不许是第二份手抄清单。"""
        src = Path(cli.__file__).read_text()
        self.assertIn("choices=list(GATE_MODES)", src)
        self.assertIn("choices=list(ROUTING_MODES)", src)


class TestQualityGateSkip(unittest.TestCase):
    def test_skip_runs_no_stage_at_all(self):
        """所有测量步骤换成"一调就炸"，仍然要跑通。"""
        with mock.patch.object(video_io, "probe", _boom), \
             mock.patch.object(video_io, "sample_frames", _boom), \
             mock.patch.object(pose_gate, "run_pose", _boom), \
             mock.patch.object(hand_gate, "detect_hands", _boom), \
             mock.patch.object(motion, "detect_shot_cuts", _boom), \
             mock.patch.object(labels, "suggest", _boom):
            rep = pipeline.diagnose_clip("/does/not/exist.mp4",
                                         QCConfig(quality_gate="skip"))
        self.assertEqual(rep.verdict, "skipped")
        self.assertEqual(rep.stages_run, [])
        self.assertEqual(rep.stages_skipped, list(pipeline.ALL_STAGES))
        self.assertEqual(rep.reasons, ["quality_gate_skipped"])
        self.assertEqual(rep.signals, {})
        self.assertIsNone(rep.error)
        self.assertFalse(rep.needs_human_review)

    def test_skip_passes_the_clip_through_untrimmed(self):
        """原样往下传 = 不给 usable_span（下游拿整段），也不给路线。"""
        rep = pipeline.diagnose_clip("/does/not/exist.mp4",
                                     QCConfig(quality_gate="skip"))
        self.assertIsNone(rep.usable_span)
        self.assertIsNone(rep.usable_sec)
        self.assertIsNone(rep.suggested_route)
        self.assertTrue(any("quality_gate=skip" in w for w in rep.route_rationale))
        # 路由的三个输入一个都没测，标签保持"未知"，不许写成默认值假装量过
        self.assertEqual(rep.view_class, ViewClass.UNKNOWN.value)
        self.assertEqual(rep.bg_texture, "unknown")

    def test_skip_marks_routing_too_when_both_off(self):
        rep = pipeline.diagnose_clip("/x.mp4", QCConfig(quality_gate="skip",
                                                        routing="skip"))
        self.assertEqual(rep.reasons, ["quality_gate_skipped"])
        # 两个开关都关的时候，rationale 要把两件事都说出来
        self.assertEqual(len(rep.route_rationale), 2)
        self.assertIn("quality_gate=skip", rep.route_rationale[0])
        self.assertIn("routing=skip", rep.route_rationale[1])

    def test_default_still_measures(self):
        """默认值必须真跑 stage —— 上面那条测试的反面对照。"""
        with self.assertRaises(AssertionError):
            with mock.patch.object(video_io, "probe", _boom):
                pipeline.diagnose_clip("/does/not/exist.mp4", QCConfig())


class TestRoutingSkip(unittest.TestCase):
    def test_builtin_route_is_verbatim_labels_suggest(self):
        cfg = QCConfig()
        args = (ViewClass.THIRD_PERSON_BODY.value, "static", "rich", True)
        self.assertEqual(pipeline._route(cfg, *args), labels.suggest(*args))

    def test_skip_does_not_call_labels_suggest(self):
        cfg = QCConfig(routing="skip")
        with mock.patch.object(labels, "suggest", _boom):
            route, why = pipeline._route(cfg, ViewClass.THIRD_PERSON_BODY.value,
                                        "static", "rich", True)
        self.assertIsNone(route)
        self.assertEqual(len(why), 1)
        self.assertIn("routing=skip", why[0])

    def test_only_the_route_fields_change(self):
        """整条判决路径跑两遍，逐字段比对。"""
        with _StubStages() as st:
            base = pipeline.diagnose_clip("stub.mp4", QCConfig()).to_dict()
            skipped = pipeline.diagnose_clip(
                "stub.mp4", QCConfig(routing="skip")).to_dict()
        # 前提：基线真的给出了一条非空路线，否则这条测试是空转
        self.assertTrue(base["suggested_route"])
        self.assertEqual(base["view_class"], ViewClass.THIRD_PERSON_BODY.value)

        differ = {k for k in base if base[k] != skipped[k]}
        self.assertEqual(differ, {"suggested_route", "route_rationale"})
        self.assertIsNone(skipped["suggested_route"])
        self.assertEqual(base["reasons"], skipped["reasons"])
        # 判决本身一字不变（differ 里没有 verdict 就已经说明了，这里写明白）
        self.assertEqual(base["verdict"], skipped["verdict"])
        self.assertEqual(base["signals"], skipped["signals"])
        self.assertEqual(base["stages_run"], skipped["stages_run"])

    def test_early_exit_path_also_honours_the_switch(self):
        """提前退出（REJECT）那条路也走 _route —— 之前那是第二个调用点。"""
        with _StubStages() as st:
            st.pfs = [pose_gate.PoseFrame(frame_idx=i) for i in range(len(st.pfs))]
            with mock.patch.object(labels, "suggest", _boom):
                rep = pipeline.diagnose_clip("stub.mp4",
                                             QCConfig(routing="skip"))
            base = pipeline.diagnose_clip("stub.mp4", QCConfig())
        self.assertEqual(rep.verdict, Verdict.REJECT.value)      # 走的是提前退出
        self.assertEqual(base.verdict, Verdict.REJECT.value)
        self.assertIn("shot_cuts", rep.stages_skipped)
        self.assertIsNone(rep.suggested_route)
        self.assertEqual(rep.reasons, base.reasons)               # 判决理由不受影响
        self.assertTrue(any("routing=skip" in w for w in rep.route_rationale))

    def test_labels_suggest_has_exactly_one_call_site(self):
        tree = ast.parse(PIPELINE_SRC)
        sites = []
        for fn in ast.walk(tree):
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for node in ast.walk(fn):
                if (isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Attribute)
                        and node.func.attr == "suggest"
                        and isinstance(node.func.value, ast.Name)
                        and node.func.value.id == "labels"):
                    sites.append(fn.name)
        self.assertEqual(sites, ["_route"], f"labels.suggest 出现在 {sites}")


class TestCliWiring(unittest.TestCase):
    """skip 档不需要模型也不需要文件，所以整条 CLI 能在单测里真跑一遍。"""

    def _run(self, out, *extra):
        argv = ["/does/not/exist.mp4", "--out", str(out)] + list(extra)
        rc = cli.main(argv)
        self.assertEqual(rc, 0)
        rows = [json.loads(l) for l in Path(out).read_text().splitlines()]
        md = Path(str(out).replace(".jsonl", ".md")).read_text()
        return rows, md

    def test_skip_end_to_end(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "qc.jsonl"
            rows, md = self._run(out, "--quality_gate", "skip")
        self.assertEqual([r["verdict"] for r in rows], ["skipped"])
        self.assertEqual(rows[0]["stages_run"], [])
        self.assertIn('"quality_gate": "skip"', md)
        self.assertIn("skipped", md)

    def test_hyphen_alias(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "qc.jsonl"
            rows, _ = self._run(out, "--quality-gate", "skip")
        self.assertEqual(rows[0]["verdict"], "skipped")

    def test_routing_flag_reaches_the_config(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "qc.jsonl"
            rows, md = self._run(out, "--quality_gate", "skip",
                                 "--routing", "skip")
        self.assertIn('"routing": "skip"', md)
        self.assertTrue(any("routing=skip" in w
                            for w in rows[0]["route_rationale"]))

    def test_skip_output_is_byte_identical_across_runs(self):
        """skip 档不碰模型，所以这一档的 jsonl 是逐字节可复现的
        （builtin 档不是：KeypointRCNN 在 GPU 上不逐位确定，见
        tests/regression/README.md，那边用 diff_quality_run.py 比）。"""
        import hashlib
        digests = []
        with tempfile.TemporaryDirectory() as d:
            for i in range(2):
                out = Path(d) / f"qc{i}.jsonl"
                self._run(out, "--quality_gate", "skip")
                digests.append(hashlib.md5(out.read_bytes()).hexdigest())
        self.assertEqual(digests[0], digests[1])

    def test_bad_mode_rejected_by_argparse(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(SystemExit):
                cli.main(["/x.mp4", "--out", str(Path(d) / "q.jsonl"),
                          "--quality_gate", "external"])


if __name__ == "__main__":
    unittest.main()
