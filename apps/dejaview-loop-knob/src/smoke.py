# SPDX-License-Identifier: Apache-2.0
"""Disclosed smoke test: verify the pipeline runs on CPU and measure per-K latency.

Not a gate; must not change thresholds in config.yaml after they are registered.
Used before pre-registration only to size the experiment grid (latency budget).
"""

import json
import time
from pathlib import Path

import numpy as np
import torch
from accelerate import Accelerator
from PIL import Image

from dvlt.model.dvlt.model import DVLT
from dvlt.util.preprocess import preprocess_images

torch.set_num_threads(4)

IMG = 252
scene = Path("data/scenes/scene0")
cams = np.load(scene / "cameras.npz")
views = [0, 3, 7, 10, 14, 17]
frames = [Image.open(scene / f"view_{v:02d}.png") for v in views]

accelerator = Accelerator(cpu=True)  # fp32 on CPU
model = DVLT(img_size=IMG)
model.load_pretrained("nvidia/dvlt", strict=True)
model.setup_test(accelerator)
n_params = sum(p.numel() for p in model.model.parameters())
print(f"params: {n_params/1e6:.1f}M")

batch = preprocess_images(frames, img_size=IMG, patch_size=14, device=accelerator.device)
print({k: tuple(v.shape) for k, v in batch.items() if hasattr(v, 'shape')})

out = {}
for K in [8, 16]:
    model.model.inference_steps = K  # DVLTModel reads this at each predict
    t0 = time.perf_counter()
    with torch.no_grad():
        preds = model.predict(batch, accelerator)
    dt = time.perf_counter() - t0
    cameras = preds["cameras"][0]
    c2w = cameras.camera_to_worlds
    depths = preds["depths"][0]
    out[K] = {"latency_s": round(dt, 2), "c2w": tuple(c2w.shape), "depth": tuple(depths.shape),
              "depth_med": float(depths.median())}
    print(K, out[K], flush=True)

Path("experiments/exp1_cpu_loop_knob").mkdir(parents=True, exist_ok=True)
Path("experiments/exp1_cpu_loop_knob/smoke.json").write_text(
    json.dumps({"params_M": n_params / 1e6, "img_size": IMG, "n_views": len(views),
                "latency": {str(k): v["latency_s"] for k, v in out.items()}}, indent=2)
)
