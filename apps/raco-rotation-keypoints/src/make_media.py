"""Still frame and a rotation sweep with the keypoints from the measured run."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from geometry import homography, warp_points

APP = Path(__file__).resolve().parents[1]
EXP = APP / "experiments" / "exp1_cpu_rotation"
ART = APP / "artifacts"
FRAME = ART / "_frames"

# RGB, colorblind-friendly.
RACO = (0, 114, 178)
ALIKED = (213, 94, 0)
LINE = (230, 230, 230)


def _font(size: int) -> ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ):
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def _panel(image: np.ndarray, keypoints: np.ndarray, color: tuple[int, int, int], title: str) -> Image.Image:
    canvas = image.copy()
    height, width = canvas.shape[:2]
    for x, y in keypoints:
        cx, cy = int(round(float(x))), int(round(float(y)))
        if 0 <= cx < width and 0 <= cy < height:
            y0, y1 = max(0, cy - 2), min(height, cy + 3)
            x0, x1 = max(0, cx - 2), min(width, cx + 3)
            canvas[y0:y1, x0:x1] = color
    picture = Image.fromarray(canvas, mode="RGB")
    draw = ImageDraw.Draw(picture)
    draw.rectangle((0, 0, width, 22), fill=(12, 16, 22))
    draw.text((8, 4), title, fill=(255, 255, 255), font=_font(13))
    return picture


def _pair(
    view_0: np.ndarray,
    view_1: np.ndarray,
    keypoints_0: np.ndarray,
    keypoints_1: np.ndarray,
    matrix: np.ndarray,
    color: tuple[int, int, int],
    title: str,
    line_budget: int = 40,
) -> Image.Image:
    left = _panel(view_0, keypoints_0, color, title + " · 0°")
    right = _panel(view_1, keypoints_1, color, title + " · rotated")
    gap = 8
    width = left.width + gap + right.width
    height = left.height
    sheet = Image.new("RGB", (width, height), (12, 16, 22))
    sheet.paste(left, (0, 0))
    sheet.paste(right, (left.width + gap, 0))
    warped = warp_points(keypoints_0, matrix)
    order = np.argsort(np.linalg.norm(warped - np.array([view_1.shape[1] / 2, view_1.shape[0] / 2]), axis=1))
    draw = ImageDraw.Draw(sheet)
    drawn = 0
    for index in order:
        x1, y1 = warped[index]
        if not (0 <= x1 < view_1.shape[1] and 0 <= y1 < view_1.shape[0]):
            continue
        distances = np.sqrt(np.sum((keypoints_1 - np.array([x1, y1])) ** 2, axis=1))
        if float(distances.min()) > 3.0:
            continue
        x0, y0 = keypoints_0[index]
        draw.line(
            (float(x0), float(y0), float(x1) + left.width + gap, float(y1)),
            fill=LINE,
            width=1,
        )
        drawn += 1
        if drawn >= line_budget:
            break
    return sheet


def _hstack(images: list[Image.Image], gap: int = 10) -> Image.Image:
    width = sum(image.width for image in images) + gap * (len(images) - 1)
    height = max(image.height for image in images)
    sheet = Image.new("RGB", (width, height), (12, 16, 22))
    x = 0
    for image in images:
        sheet.paste(image, (x, 0))
        x += image.width + gap
    return sheet


def _vstack(images: list[Image.Image], gap: int = 10) -> Image.Image:
    width = max(image.width for image in images)
    height = sum(image.height for image in images) + gap * (len(images) - 1)
    sheet = Image.new("RGB", (width, height), (12, 16, 22))
    y = 0
    for image in images:
        sheet.paste(image, (0, y))
        y += image.height + gap
    return sheet


def _even(image: Image.Image) -> Image.Image:
    width = image.width + (image.width % 2)
    height = image.height + (image.height % 2)
    if (width, height) == image.size:
        return image
    sheet = Image.new("RGB", (width, height), (12, 16, 22))
    sheet.paste(image, (0, 0))
    return sheet


def main() -> None:
    cache = np.load(EXP / "media_cache.npz", allow_pickle=False)
    angles = cache["angles"]
    views = cache["views"]
    side = int(cache["side"][0])
    out_size = int(cache["image_size"][0])
    sequence = str(cache["sequence"])
    ART.mkdir(parents=True, exist_ok=True)
    index_90 = int(np.argmin(np.abs(angles - 90.0)))
    matrix = homography(float(angles[index_90]), out_size, side, scale=1.0)
    raco_sheet = _pair(
        views[index_90, 0],
        views[index_90, 1],
        cache["raco_0"][index_90],
        cache["raco_1"][index_90],
        matrix,
        RACO,
        f"RaCo · {sequence}",
    )
    aliked_sheet = _pair(
        views[index_90, 0],
        views[index_90, 1],
        cache["aliked_0"][index_90],
        cache["aliked_1"][index_90],
        matrix,
        ALIKED,
        f"ALIKED-N16 · {sequence}",
    )
    still = _even(_vstack([raco_sheet, aliked_sheet]))
    still_path = ART / "raco-rotation-keypoints.png"
    still.save(still_path)

    if FRAME.exists():
        shutil.rmtree(FRAME)
    FRAME.mkdir(parents=True)
    for index, angle in enumerate(angles):
        raco = _panel(views[index, 1], cache["raco_1"][index], RACO, f"RaCo · {float(angle):g}°")
        aliked = _panel(views[index, 1], cache["aliked_1"][index], ALIKED, f"ALIKED-N16 · {float(angle):g}°")
        frame = _even(_hstack([raco, aliked]))
        frame.save(FRAME / f"frame_{index:03d}.png")
    video = ART / "raco-rotation-keypoints.mp4"
    command = [
        "ffmpeg",
        "-y",
        "-framerate",
        "2",
        "-i",
        str(FRAME / "frame_%03d.png"),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(video),
    ]
    subprocess.run(command, check=True, capture_output=True)
    (ART / "media_backend.txt").write_text("ffmpeg libx264\n")
    print(f"wrote {still_path.name} and {video.name}")


if __name__ == "__main__":
    main()
