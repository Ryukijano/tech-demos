# SPDX-License-Identifier: Apache-2.0
"""Capture the required screenshot + video of the running viewer.

Screenshot: full page at K=16. Video: sweep the K knob end to end (frames via
headless Chromium, encoded to H.264 MP4 with ffmpeg libx264).
"""

import subprocess
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

APP = Path(__file__).resolve().parents[1]
ART = APP / "artifacts"
FRAMES = Path("/tmp/dvlt-frames")
FRAMES.mkdir(exist_ok=True)
ART.mkdir(exist_ok=True)
URL = "http://127.0.0.1:43130/index.html"

with sync_playwright() as pw:
    browser = pw.chromium.launch()
    page = browser.new_page(viewport={"width": 1280, "height": 800}, device_scale_factor=2)
    page.goto(URL)
    page.wait_for_selector("#resultsTable tr")
    time.sleep(1.5)

    # screenshot at K=16 (the headline configuration)
    page.evaluate("""() => {
        const knob = document.getElementById('knob');
        const ks = [...document.querySelectorAll('#resultsTable tr[data-k]')].map(tr => tr.dataset.k);
        knob.value = ks.indexOf('16');
        knob.dispatchEvent(new Event('input'));
    }""")
    time.sleep(0.8)
    page.screenshot(path=str(ART / "dejaview-loop-knob.png"), full_page=True)

    # video: top of page, sweep knob 0 -> max -> back to 16, ~24 fps output
    page.evaluate("window.scrollTo(0, 0)")
    n_pos = page.evaluate("() => +document.getElementById('knob').max")
    seq = list(range(0, n_pos + 1)) + [n_pos] * 2 + list(range(n_pos, -1, -1))
    frame_idx = 0
    for pos in seq:
        page.evaluate(
            """(pos) => {
                const knob = document.getElementById('knob');
                knob.value = pos;
                knob.dispatchEvent(new Event('input'));
            }""",
            pos,
        )
        time.sleep(0.15)
        for _ in range(10):  # 10 frames per knob position
            page.screenshot(path=str(FRAMES / f"f{frame_idx:05d}.png"), clip={"x": 0, "y": 0, "width": 1280, "height": 800})
            frame_idx += 1
    browser.close()

subprocess.run(
    ["ffmpeg", "-y", "-framerate", "24", "-i", str(FRAMES / "f%05d.png"),
     "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "22", "-movflags", "+faststart",
     str(ART / "dejaview-loop-knob.mp4")],
    check=True, capture_output=True,
)
(ART / "media_backend.txt").write_text("ffmpeg libx264\n")
print("media captured:", sorted(p.name for p in ART.iterdir()))
