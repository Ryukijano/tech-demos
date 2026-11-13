# SPDX-License-Identifier: Apache-2.0
"""Disclosed smoke metric check at a single mid-grid K=12 on scene0 only.

Purpose: confirm the GT conventions line up with model output (poses not degenerate,
depth mask sane) BEFORE pre-registering thresholds. It deliberately looks at one K
only, so it cannot inform the K-trend part of the hypothesis. Not a gate.
"""

import json
from pathlib import Path

import numpy as np
import torch
from accelerate import Accelerator
from PIL import Image

from dvlt.metric.depth import compute_depth_metrics
from dvlt.metric.pose import compute_pose_metrics
from dvlt.model.dvlt.model import DVLT
from dvlt.util.preprocess import preprocess_images

torch.set_num_threads(4)

IMG = 252
scene = Path("data/scenes/scene0")
cams = np.load(scene / "cameras.npz")
views = [0, 3, 7, 10, 14, 17]
frames = [Image.open(scene / f"view_{v:02d}.png") for v in views]
gt_c2w = cams["c2w"][views]  # (S, 3, 4)
gt_depth = np.load(scene / "depth.npy")[views]  # (S, H, W)

accelerator = Accelerator(cpu=True)
model = DVLT(img_size=IMG)
model.load_pretrained("nvidia/dvlt", strict=True)
model.setup_test(accelerator)
model.model.inference_steps = 12

batch = preprocess_images(frames, img_size=IMG, patch_size=14, device=accelerator.device)
with torch.no_grad():
    preds = model.predict(batch, accelerator)

pred_c2w = preds["cameras"][0].camera_to_worlds.cpu().numpy()  # (S, 3, 4)


def homog(p34):
    out = np.tile(np.eye(4), (p34.shape[0], 1, 1))
    out[:, :3, :] = p34
    return torch.from_numpy(out).float()

pose_m = compute_pose_metrics(homog(pred_c2w), homog(gt_c2w), thresholds=(5, 15, 30))
pred_depth = preds["depths"][0].cpu()  # (S, H, W)
gt_d = torch.from_numpy(gt_depth).float()
mask = gt_d > 0
depth_m = compute_depth_metrics(pred_depth, gt_d, mask, align="median")

res = {"K": 12, **{k: float(v) for k, v in pose_m.items()},
       **{f"depth_{k}": float(v) for k, v in depth_m.items()}}
print(json.dumps(res, indent=2))
p = Path("experiments/exp1_cpu_loop_knob/smoke.json")
d = json.loads(p.read_text()) if p.exists() else {}
d["metric_smoke_K12_scene0"] = res
p.write_text(json.dumps(d, indent=2))
