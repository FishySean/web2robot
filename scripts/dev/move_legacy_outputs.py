"""把 external/ 里的存量产物搬到 outputs/legacy_runs/，路径保持原样。

一次性迁移脚本，2026-08-10 跑过一次（65 项 / 291 文件 / 316 MiB）。留在仓库里
是为了让那次搬迁可复查、可复现 —— 清单在 ``outputs/legacy_runs/MANIFEST.tsv``。
现在再跑应该是"待处理 0 项"，因为 ``P.check_output_dir`` 已经不让新产物落进
``external/`` 了（见 tests/test_outputs_not_in_external.py）。

    envs/rt_env/bin/python scripts/dev/move_legacy_outputs.py --dry   # 只看清单
    envs/rt_env/bin/python scripts/dev/move_legacy_outputs.py         # 真搬

判据不是"数 mp4"，而是上游 git 认不认：含产物标志文件、又不被上游跟踪的目录
就是我们的产物躺在别人家里。

安全规矩（这一版是重写的：第一版用 shutil.move，撞上 root 拥有的 run 目录之后
copy 成功、rmtree 失败，留下了源和目的地各一份）：
  * 先 rename；rename 不行（跨所有者/权限）才 copy
  * **copy 完必须逐文件 md5 比对，比对通过才删源**
  * 源是 root 拥有的才用 sudo -n 删，且只删 upstream retarget/ 之下的路径
  * 可重复跑：目的地已存在就走"校验 + 删源"这条路
"""
import hashlib, os, shutil, subprocess, sys
from pathlib import Path
sys.path.insert(0, "src")
from web2robot.paths import P

UP   = (P.root("egoinfinity") / "retarget").resolve()
DEST = Path("outputs/legacy_runs").resolve()
OUT_MARK = {"robot_sim.mp4","trajectory.npz","metrics.npz","root_frames.npz","input_viz.mp4"}
IN_MARK  = {"depth.mp4","hand_joints.bin","hand_meta.json","scene.json"}
DRY = "--dry" in sys.argv

def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""): h.update(b)
    return h.hexdigest()

def tree(p):
    if p.is_file(): return {"": md5(p)}
    return {str(f.relative_to(p)): md5(f) for f in sorted(p.rglob("*")) if f.is_file()}

def rm_src(p):
    """删源。只允许删 UP 之下的路径；root 拥有的才升权。"""
    assert str(p).startswith(str(UP) + os.sep), f"拒绝删 {p}（不在上游树里）"
    try:
        shutil.rmtree(p) if p.is_dir() else p.unlink()
    except PermissionError:
        subprocess.run(["sudo", "-n", "rm", "-rf", str(p)], check=True)
        subprocess.run(["sudo", "-n", "chown", "-R", f"{os.getuid()}:{os.getgid()}",
                        str(DEST)], check=True)

# ── 收集 ──────────────────────────────────────────────────────────────────────
tracked = set(subprocess.run(["git","ls-files"], cwd=UP,
                             capture_output=True, text=True).stdout.split())
items = []
for base in ("examples","runs"):
    for d in sorted((UP/base).rglob("*")):
        if not d.is_dir(): continue
        names = {p.name for p in d.iterdir() if p.is_file()}
        if not (names & OUT_MARK): continue
        rel = d.relative_to(UP)
        assert not (names & IN_MARK), f"{rel} 输入输出混层，人工处理"
        assert not any(t.startswith(str(rel)+"/") for t in tracked), f"{rel} 被上游跟踪"
        items.append((d, "run"))
def covered(p): return any(p==s or str(p).startswith(str(s)+os.sep) for s,_ in items)
for d in sorted((UP/"examples").iterdir()):
    if d.is_dir():
        items += [(f, "log") for f in sorted(d.glob("*.log")) if not covered(f)]
for extra in (UP/"examples"/"_compare", UP/"runs"/"compare"/"fill_jar_grid_h264.mp4"):
    if extra.exists() and not covered(extra): items.append((extra, "compare"))

print(f"待处理 {len(items)} 项")
if DRY:
    for s,k in items: print(f"  [{k}] {s.relative_to(UP)}")
    sys.exit(0)

# ── 搬 ────────────────────────────────────────────────────────────────────────
DEST.mkdir(parents=True, exist_ok=True)
moved = renamed = copied = resumed = 0
for src, kind in items:
    rel = src.relative_to(UP); dst = DEST / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():                                   # 上一次跑到一半留下的
        assert tree(src) == tree(dst), f"{rel} 源和目的地内容不一致，停下人工看"
        rm_src(src); resumed += 1; moved += 1; continue
    try:
        os.rename(src, dst); renamed += 1
    except OSError:
        (shutil.copytree if src.is_dir() else shutil.copy2)(src, dst)
        assert tree(src) == tree(dst), f"{rel} copy 后 md5 不一致，源保留不动"
        rm_src(src); copied += 1
    moved += 1
print(f"搬完 {moved} 项：rename {renamed}，copy+校验 {copied}，续做上次残留 {resumed}")

# ── 清单 ──────────────────────────────────────────────────────────────────────
rows = []
for p in sorted(DEST.rglob("*")):
    if p.is_dir() or p.name == "MANIFEST.tsv": continue
    rows.append((str(p.relative_to(DEST)), p.stat().st_size))
with open(DEST/"MANIFEST.tsv", "w") as fh:
    fh.write("# 2026-08-10 从 external/EgoInfinity/retarget/ 搬出来的存量产物\n")
    fh.write("# 路径保持原样（相对上游 retarget/），没有做任何重命名或重组：\n")
    fh.write("#   examples/<片段>/m7_*/  = 各次 M7 重定向 run\n")
    fh.write("#   examples/{g1,franka,robonaut2}/ = 没给 --out 时落在上游默认位置的 run\n")
    fh.write("#   examples/_compare/     = 对比片\n")
    fh.write("#   runs/m7/validation/    = 验收用的渲染\n")
    fh.write("# 留在 external/ 里的是输入：片段的 4 个输入文件 + runs/m7/taskspace*/（训练 run）\n")
    fh.write("rel_path\tbytes\n")
    for r in rows: fh.write(f"{r[0]}\t{r[1]}\n")
print(f"清单 {DEST/'MANIFEST.tsv'}：{len(rows)} 个文件，"
      f"{sum(r[1] for r in rows)/2**20:.0f} MiB")
