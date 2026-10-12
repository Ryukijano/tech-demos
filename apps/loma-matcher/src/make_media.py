"""Screenshot grid (PNG) and rotation-sweep video (MP4) for Pages + README."""
from __future__ import annotations

import glob
import json
import os
import shutil
import subprocess
import sys

import cv2
import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(__file__))
import lmx

EXP = os.path.join(lmx.APP, "experiments/exp1_cpu_match")
ART = os.path.join(lmx.APP, "artifacts")
cfg = yaml.safe_load(open(os.path.join(EXP, "config.yaml")))
raw = json.load(open(os.path.join(EXP, "raw.json")))
os.makedirs(ART, exist_ok=True)
lmx.ensure_dirs()

# Ensure assets present
paths = {}
for it in cfg["inputs"]:
    paths[it["id"]] = dict(A=lmx.fetch(it["A"]), B=lmx.fetch(it["B"]))

# ---- grid: LoMa-B matches on all three pairs ----
tiles = []
for pid in [it["id"] for it in cfg["inputs"]]:
    npz = os.path.join(lmx.WORK, f"{pid}_LoMa-B_clean.npz")
    if not os.path.isfile(npz):
        print("missing", npz, "- skip tile")
        continue
    z = np.load(npz)
    canvas = lmx.draw_matches(paths[pid]["A"], paths[pid]["B"], z["kptsA"], z["kptsB"], max_lines=150, seed=0)
    # resize to common height
    h = 360
    scale = h / canvas.shape[0]
    canvas = cv2.resize(canvas, (int(canvas.shape[1] * scale), h), interpolation=cv2.INTER_AREA)
    n = int(raw["clean"][pid]["n_matches"])
    nin = int(raw["clean"][pid]["n_inliers"])
    canvas = lmx.label_bar(canvas, f"{pid}  LoMa-B  matches={n}  MAGSAC inliers={nin}")
    tiles.append(canvas)

if tiles:
    # pad to same width
    wmax = max(t.shape[1] for t in tiles)
    padded = []
    for t in tiles:
        if t.shape[1] < wmax:
            pad = np.zeros((t.shape[0], wmax - t.shape[1], 3), dtype=np.uint8)
            t = np.hstack([t, pad])
        padded.append(t)
    grid = np.vstack(padded)
    foot = np.full((34, grid.shape[1], 3), 20, np.uint8)
    cv2.putText(
        foot,
        "davnords/LoMa (MIT; matcher Apache-2.0 via LightGlue) on CPU | upstream sample pairs | clear-experiment demo",
        (8, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )
    out_png = os.path.join(ART, "loma-matcher-grid.png")
    cv2.imwrite(out_png, np.vstack([grid, foot]))
    print("wrote", out_png)
else:
    print("WARNING: no tiles for grid")

# ---- video: rotation sweep, LoMa-B vs LoMa-R side by side on pair_0015 ----
pid = cfg["breakpoints"]["images"][0]
fr = os.path.join(lmx.WORK, "frames")
shutil.rmtree(fr, ignore_errors=True)
os.makedirs(fr, exist_ok=True)

bp_rows = [r for r in raw.get("breakpoints", []) if r["image"] == pid]
angles = sorted({r["level"] for r in bp_rows}) if bp_rows else [0, 45, 90, 180]
path_a = paths[pid]["A"]
path_b0 = paths[pid]["B"]

frame_i = 0
for ang in angles:
    path_b = path_b0 if ang == 0 else lmx.rotate_image_file(path_b0, ang, os.path.join(lmx.WORK, f"{pid}_B_rot{ang}.jpg"))
    panels = []
    for model in ["LoMa-B", "LoMa-R"]:
        npz = os.path.join(lmx.WORK, f"bp_{pid}_{model}_rot{ang}.npz")
        row = next((r for r in bp_rows if r["model"] == model and r["level"] == ang), None)
        if os.path.isfile(npz):
            z = np.load(npz)
            ka, kb = z["kptsA"], z["kptsB"]
            # pathB inside npz may be bytes
            pb = path_b
            canvas = lmx.draw_matches(path_a, pb, ka, kb, max_lines=120, seed=1)
        else:
            # placeholder if keypoints missing (e.g. skip-breakpoints)
            a = cv2.imread(path_a)
            b = cv2.imread(path_b)
            h = max(a.shape[0], b.shape[0])
            canvas = np.zeros((h, a.shape[1] + b.shape[1], 3), dtype=np.uint8)
            canvas[: a.shape[0], : a.shape[1]] = a
            canvas[: b.shape[0], a.shape[1] :] = b
            ka = np.zeros((0, 2))
        h = 320
        scale = h / canvas.shape[0]
        canvas = cv2.resize(canvas, (int(canvas.shape[1] * scale), h), interpolation=cv2.INTER_AREA)
        if row:
            txt = f"{model}  rot={ang}°  n={row['n_matches']}  inlier_ratio={row['inlier_ratio']:.3f}"
        else:
            txt = f"{model}  rot={ang}°"
        panels.append(lmx.label_bar(canvas, txt))
    # stack B over R, pad widths
    wmax = max(p.shape[1] for p in panels)
    stacked = []
    for p in panels:
        if p.shape[1] < wmax:
            p = np.hstack([p, np.zeros((p.shape[0], wmax - p.shape[1], 3), dtype=np.uint8)])
        stacked.append(p)
    frame = np.vstack(stacked)
    title = np.full((30, frame.shape[1], 3), 15, np.uint8)
    cv2.putText(
        title,
        f"LoMa rotation sweep — {pid} — image B rotated {ang}° (expect LoMa-R to hold past ~45°)",
        (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )
    frame = np.vstack([title, frame])
    for _ in range(4):  # hold each angle
        cv2.imwrite(os.path.join(fr, f"{frame_i:04d}.png"), frame)
        frame_i += 1

out_mp4 = os.path.join(ART, "loma-matcher-rotation.mp4")
if frame_i > 0:
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-framerate",
            "2",
            "-i",
            os.path.join(fr, "%04d.png"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            out_mp4,
        ],
        check=True,
    )
    print("wrote", out_mp4)
else:
    print("WARNING: no frames for video")

# also copy poster naming aliases used by Pages
shutil.copy2(os.path.join(ART, "loma-matcher-grid.png"), os.path.join(ART, "loma-matcher.png")) if os.path.isfile(os.path.join(ART, "loma-matcher-grid.png")) else None
if os.path.isfile(out_mp4):
    shutil.copy2(out_mp4, os.path.join(ART, "loma-matcher.mp4"))
print("wrote artifacts")
