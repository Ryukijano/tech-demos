"""Poster PNG and a short MP4 of one covisible pair and one doppelganger pair."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import fgx

EXP = os.path.join(fgx.APP, "experiments/exp1_cpu_covisibility")
ART = os.path.join(fgx.APP, "artifacts")
raw = json.load(open(os.path.join(EXP, "raw.json"), encoding="utf-8"))
status = json.load(open(os.path.join(EXP, "status.json"), encoding="utf-8"))
os.makedirs(ART, exist_ok=True)


def load_rgb(rel: str) -> np.ndarray:
    return fgx.load_rgb(os.path.join(fgx.APP, rel))


def render(image0: str, image1: str, viz_rel: str, title: str) -> np.ndarray:
    z = np.load(os.path.join(fgx.APP, viz_rel))
    scores = z["scores"] if z["scores"].size else np.zeros(0, np.float32)
    return fgx.draw_matches(load_rgb(image0), load_rgb(image1), z["kpts0"], z["kpts1"], z["matches"], scores, title)


def fit_height(image: np.ndarray, height: int) -> np.ndarray:
    scale = height / image.shape[0]
    width = max(2, int(round(image.shape[1] * scale)))
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def even(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    pad_h = h % 2
    pad_w = w % 2
    if pad_h == 0 and pad_w == 0:
        return image
    return cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(0, 0, 0))


viz = raw.get("viz_index", [])
pos = next((v for v in viz if int(v["label"]) == 1), None)
neg = next((v for v in viz if int(v["label"]) == 0), None)
if pos is None or neg is None:
    raise SystemExit("viz_index is missing a positive or negative pair")

pos_title = (
    f"COVISIBLE  {pos['id']}  score={pos['covisibility']:.3f}  "
    f"FG matches={pos['n_matches']}  inliers={pos['n_inliers']}  SIFT={pos['n_sift']}"
)
neg_title = (
    f"DOPPELGANGER  {neg['id']}  score={neg['covisibility']:.3f}  "
    f"FG matches={neg['n_matches']}  inliers={neg['n_inliers']}  SIFT={neg['n_sift']}"
)
panel_pos = fit_height(render(pos["image0"], pos["image1"], pos["viz"], pos_title), 420)
panel_neg = fit_height(render(neg["image0"], neg["image1"], neg["viz"], neg_title), 420)
width = max(panel_pos.shape[1], panel_neg.shape[1])


def pad_w(image: np.ndarray) -> np.ndarray:
    if image.shape[1] == width:
        return image
    return np.hstack([image, np.zeros((image.shape[0], width - image.shape[1], 3), np.uint8)])


foot = np.full((48, width, 3), 18, np.uint8)
mean_c = status["mean_auroc_covis"]
std_c = status["std_auroc_covis"]
mean_f = status["mean_auroc_fg_matches"]
cv2.putText(
    foot,
    f"FlexGlue CPU ONNX  |  {status['status']}  |  AUROC covis {mean_c:.3f}+/-{std_c:.3f}  vs match-count {mean_f:.3f}",
    (8, 20),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.45,
    (230, 230, 230),
    1,
    cv2.LINE_AA,
)
cv2.putText(
    foot,
    "cvg/FlexGlue MIT + DINOv3 license  |  Doppelgangers test sample, CC BY / CC0  |  not the full benchmark",
    (8, 40),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.4,
    (180, 180, 180),
    1,
    cv2.LINE_AA,
)
poster = even(np.vstack([pad_w(panel_pos), pad_w(panel_neg), foot]))
png_path = os.path.join(ART, "flexglue-doppelganger.png")
cv2.imwrite(png_path, poster)
print("wrote", png_path)

frames_dir = os.path.join(fgx.WORK, "frames")
shutil.rmtree(frames_dir, ignore_errors=True)
os.makedirs(frames_dir, exist_ok=True)
frame_i = 0


def hold(image: np.ndarray, n: int) -> None:
    global frame_i
    image = even(fit_height(image, 480))
    for _ in range(n):
        cv2.imwrite(os.path.join(frames_dir, f"{frame_i:04d}.png"), image)
        frame_i += 1


hold(panel_pos, 8)
hold(panel_neg, 8)

# Rotation sweep: positive over negative when both exist for that angle.
by_angle: dict[str, dict[int, dict]] = {}
for rec in raw.get("breakpoints", []):
    if rec.get("kind") != "rotation" or "viz" not in rec:
        continue
    by_angle.setdefault(str(rec["level"]), {})[int(rec["label"])] = rec

for level in sorted(by_angle, key=lambda s: float(s)):
    tiles = []
    for label in (1, 0):
        rec = by_angle[level].get(label)
        if rec is None:
            continue
        kind = "COVISIBLE" if label == 1 else "DOPPELGANGER"
        title = f"{kind}  rot {level} deg  score={float(rec['covisibility']):.3f}  matches={rec['n_matches']}"
        tiles.append(fit_height(render(rec["image0"], rec["image1"], rec["viz"], title), 300))
    if not tiles:
        continue
    wmax = max(t.shape[1] for t in tiles)
    padded = []
    for tile in tiles:
        if tile.shape[1] < wmax:
            tile = np.hstack([tile, np.zeros((tile.shape[0], wmax - tile.shape[1], 3), np.uint8)])
        padded.append(tile)
    hold(np.vstack(padded), 8)

mp4_path = os.path.join(ART, "flexglue-doppelganger.mp4")
if frame_i == 0:
    raise SystemExit("no frames")
subprocess.run(
    [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-framerate",
        "8",
        "-i",
        os.path.join(frames_dir, "%04d.png"),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        mp4_path,
    ],
    check=True,
)
print("wrote", mp4_path, "frames", frame_i)
docs = os.path.join(fgx.APP, "..", "..", "docs", "assets")
if os.path.isdir(docs):
    shutil.copy2(png_path, os.path.join(docs, "flexglue-doppelganger.png"))
    shutil.copy2(mp4_path, os.path.join(docs, "flexglue-doppelganger.mp4"))
    print("copied into docs/assets")
