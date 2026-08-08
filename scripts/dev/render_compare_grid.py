"""2x2 comparison grid for one clip across robots.

Layout:
  top-left  = input (human source hand motion, from a robot's input_viz)
  top-right = M7      bottom-left = Robonaut2      bottom-right = G1

Each robot rendered from a close-up FRONT view (its own configured azimuth/
elevation so we see it from the front; a tightened distance for close-up).
"""
import importlib, numpy as np, mujoco, cv2, os
from utils.viz import render_robot_sim

PANEL = 512
CLIP_FPS = 15.0

# robot -> (config module, out_dir, label, close-up cam (az, elev, dist, lookat))
# Cameras calibrated so each robot's upper body + arms fill a comparable frame.
ROBOTS = [
    ("m7",        "web2robot.robots.m7.config",        "runs/m7/validation/fill_jar",
     "M7 (ours)",  (180, -10, 1.55, [0.0, 0.0, 0.33])),
    ("robonaut2", "sim.robots.robonaut2.config", "runs/compare/fill_jar_r2",
     "Robonaut2",  (60,  -12, 2.65, [0.0, 0.0, 0.66])),
    ("g1",        "sim.robots.g1.config",        "runs/compare/fill_jar_g1",
     "G1",         (140, -12, 1.60, [0.1, 0.0, 0.68])),
]
INPUT_VIZ = "runs/m7/validation/fill_jar/input_viz.mp4"


def render_robot(cfg_key, cfg_mod, out_dir, cam):
    CONFIG = importlib.import_module(cfg_mod).CONFIG
    d = np.load(f"{out_dir}/trajectory.npz", allow_pickle=True)
    qL, qR = d["q_left"], d["q_right"]
    QLf = d["q_left_fingers"] if "q_left_fingers" in d.files else None
    QRf = d["q_right_fingers"] if "q_right_fingers" in d.files else None
    fj = [n.replace("left_", "").replace("_joint", "") for n in d["left_finger_joint_names"]]
    scene = CONFIG.get("scene_path_fingers", CONFIG["scene_path"])
    env = CONFIG["env_cls"](mjcf_path=scene, start_config=CONFIG["start_config"])
    env.reset()
    az, el, di, lo = cam
    cfg = dict(CONFIG)
    cfg["cam_azimuth"] = az; cfg["cam_elevation"] = el
    cfg["cam_distance"] = di; cfg["cam_lookat"] = lo
    frames = render_robot_sim(env, qL, qR, QLf, QRf, fj, cfg,
                              height=PANEL, width=PANEL)
    return frames


def load_video(path, n):
    cap = cv2.VideoCapture(path); fr = []
    while True:
        ok, f = cap.read()
        if not ok: break
        fr.append(f)
    cap.release()
    # pad/truncate to n
    if len(fr) < n and fr: fr += [fr[-1]] * (n - len(fr))
    return fr[:n]


def fit(img, size=PANEL):
    h, w = img.shape[:2]
    s = size / max(h, w)
    r = cv2.resize(img, (int(w*s), int(h*s)))
    canvas = np.zeros((size, size, 3), np.uint8)
    y = (size - r.shape[0]) // 2; x = (size - r.shape[1]) // 2
    canvas[y:y+r.shape[0], x:x+r.shape[1]] = r
    return canvas


def label(img, text, color=(255, 255, 255)):
    cv2.rectangle(img, (0, 0), (img.shape[1], 26), (0, 0, 0), -1)
    cv2.putText(img, text, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return img


def main():
    renders = {}
    for key, mod, out, lab, cam in ROBOTS:
        print(f"rendering {key} ...", flush=True)
        renders[key] = render_robot(key, mod, out, cam)
    T = min(len(v) for v in renders.values())
    inp = load_video(INPUT_VIZ, T)
    print(f"T={T} frames; input={len(inp)}", flush=True)

    tmp = "runs/compare/fill_jar_grid_tmp.mp4"
    out = "runs/compare/fill_jar_grid_h264.mp4"
    vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), CLIP_FPS, (PANEL*2, PANEL*2))
    for t in range(T):
        tl = label(fit(inp[t] if t < len(inp) else inp[-1]), "INPUT (human)", (0, 255, 255))
        tr = label(fit(renders["m7"][t]), "M7 (ours)", (0, 200, 255))
        bl = label(fit(renders["robonaut2"][t]), "Robonaut2", (200, 255, 200))
        br = label(fit(renders["g1"][t]), "G1", (200, 200, 255))
        top = np.hstack([tl, tr]); bot = np.hstack([bl, br])
        vw.write(np.vstack([top, bot]))
    vw.release()
    os.system(f"ffmpeg -y -loglevel error -i {tmp} -c:v libx264 -pix_fmt yuv420p {out}")
    os.remove(tmp)
    print("WROTE", out, f"({T} frames @ {CLIP_FPS:.0f}fps)", flush=True)


if __name__ == "__main__":
    main()
