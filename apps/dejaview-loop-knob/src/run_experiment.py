# SPDX-License-Identifier: Apache-2.0
"""Measured run for exp1_cpu_loop_knob.

Protocol is frozen in experiments/exp1_cpu_loop_knob/config.yaml (pre-registration
commit). This script records the config SHA-256 into raw.json so any later edit of
the thresholds would be visible.
"""

import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from accelerate import Accelerator
from PIL import Image

from dvlt.metric.depth import compute_depth_metrics
from dvlt.metric.pose import compute_pose_metrics
from dvlt.model.dvlt.model import DVLT
from dvlt.util.preprocess import preprocess_images

APP = Path(__file__).resolve().parents[1]
EXP = APP / "experiments" / "exp1_cpu_loop_knob"
CFG_PATH = EXP / "config.yaml"
CFG = yaml.safe_load(CFG_PATH.read_text())
CFG_SHA = hashlib.sha256(CFG_PATH.read_bytes()).hexdigest()

IMG = CFG["model"]["img_size"]
N_THREADS = CFG["model"]["torch_num_threads"]
SEEDS = CFG["seeds"]
K_MAIN = CFG["k_grid_main"]
K_BREAK = CFG["k_grid_breakpoints"]
N_SCENES = CFG["data"]["n_scenes"]
N_CAND = CFG["data"]["n_candidate_views"]
V_PER = CFG["data"]["views_per_scene"]

torch.set_num_threads(N_THREADS)


def homog(p34: np.ndarray) -> torch.Tensor:
    out = np.tile(np.eye(4), (p34.shape[0], 1, 1))
    out[:, :3, :] = p34
    return torch.from_numpy(out).float()


def load_scene(si: int):
    sdir = APP / "data" / "scenes" / f"scene{si}"
    cams = np.load(sdir / "cameras.npz")
    depth = np.load(sdir / "depth.npy")
    frames = [Image.open(sdir / f"view_{vi:02d}.png") for vi in range(N_CAND)]
    return frames, cams["c2w"], depth


def evaluate(model, accelerator, frames, gt_c2w, gt_depth, view_idx, k, img_size):
    sel_frames = [frames[v] for v in view_idx]
    batch = preprocess_images(sel_frames, img_size=img_size, patch_size=14, device=accelerator.device)
    model.model.inference_steps = int(k)
    t0 = time.perf_counter()
    with torch.no_grad():
        preds = model.predict(batch, accelerator)
    latency = time.perf_counter() - t0

    pred_c2w = preds["cameras"][0].camera_to_worlds.cpu().numpy()
    pose_m = compute_pose_metrics(homog(pred_c2w), homog(gt_c2w[view_idx]), thresholds=(5, 15, 30))

    pred_depth = preds["depths"][0].cpu()
    gt_d = torch.from_numpy(gt_depth[view_idx]).float()
    if pred_depth.shape[-2:] != gt_d.shape[-2:]:  # low-res breakpoint: GT rendered at 252
        gt_d = torch.nn.functional.interpolate(
            gt_d[:, None], size=pred_depth.shape[-2:], mode="nearest"
        )[:, 0]
    mask = gt_d > 0
    depth_m = compute_depth_metrics(pred_depth, gt_d, mask, align="median")

    row = {
        "k": int(k),
        "latency_s": round(latency, 4),
        **{m: float(pose_m[m]) for m in ("Auc_30", "Auc_15", "Auc_5", "R_error", "T_error", "Racc_15", "Tacc_15")},
        "AbsRel": float(depth_m["AbsRel"]),
        "Delta1": float(depth_m["Delta1"]),
    }
    return row, pred_c2w, pred_depth.numpy()


def main():
    accelerator = Accelerator(cpu=True)
    model = DVLT(img_size=IMG)
    model.load_pretrained("nvidia/dvlt", strict=True)
    model.setup_test(accelerator)

    scenes = [load_scene(si) for si in range(N_SCENES)]
    view_sel = {}
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        view_sel[seed] = {si: sorted(rng.choice(N_CAND, V_PER, replace=False).tolist()) for si in range(N_SCENES)}

    # warmup (frozen protocol: one predict at K=8 after model load, untimed row)
    evaluate(model, accelerator, scenes[0][0], scenes[0][1], scenes[0][2], view_sel[SEEDS[0]][0], 8, IMG)

    rows = []
    viewer = {}  # per-K artifacts for the viewer (scene0, seed0)
    all_k = sorted(set(K_MAIN) | set(K_BREAK))
    t_start = time.time()
    for k in all_k:
        for si in range(N_SCENES):
            frames, gt_c2w, gt_depth = scenes[si]
            for seed in SEEDS:
                vi = view_sel[seed][si]
                row, pred_c2w, pred_depth = evaluate(model, accelerator, frames, gt_c2w, gt_depth, vi, k, IMG)
                row.update({"scene": si, "seed": seed, "views": vi, "grid": "main" if k in K_MAIN else "breakpoint"})
                rows.append(row)
                print(json.dumps(row), flush=True)
                if si == 0 and seed == SEEDS[0]:
                    viewer[str(k)] = {
                        "pred_c2w": pred_c2w.tolist(),
                        "gt_c2w": gt_c2w[vi].tolist(),
                        "views": vi,
                        "depth_view0": pred_depth[0][:: 2, :: 2].round(4).tolist(),
                        "row": row,
                    }

    # ---------------- report-only breakpoints beyond the K grid ----------------
    bp_rows = []

    def bp(label, view_count, k, img_size, seed=0):
        for si in range(N_SCENES):
            frames, gt_c2w, gt_depth = scenes[si]
            rng = np.random.default_rng(seed)
            vi = sorted(rng.choice(N_CAND, view_count, replace=False).tolist())
            try:
                row, _, _ = evaluate(model, accelerator, frames, gt_c2w, gt_depth, vi, k, img_size)
                row.update({"scene": si, "seed": seed, "label": label, "n_views": view_count, "img_size": img_size})
            except Exception as e:  # noqa: BLE001 — breakpoints report crashes honestly
                row = {"k": k, "scene": si, "seed": seed, "label": label, "n_views": view_count,
                       "img_size": img_size, "error": repr(e)}
            bp_rows.append(row)
            print(json.dumps(row), flush=True)

    for k in (8, 16):
        bp("two_views", 2, k, IMG)
        bp("low_resolution", V_PER, k, 140)
        bp("many_views", 12, k, IMG)

    gt0_depth_v0 = scenes[0][2][view_sel[SEEDS[0]][0][0]]
    raw = {
        "config_sha256": CFG_SHA,
        "machine": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "threads": N_THREADS,
            "cpu_count": torch.get_num_threads(),
        },
        "wall_time_s": round(time.time() - t_start, 1),
        "view_selection": {str(s): view_sel[s] for s in SEEDS},
        "rows": rows,
        "breakpoint_rows": bp_rows,
    }
    (EXP / "raw.json").write_text(json.dumps(raw, indent=1))
    viewer["gt_depth_scene0_view0"] = gt0_depth_v0[::2, ::2].round(4).tolist()
    (APP / "viewer" / "viewer_data.json").write_text(json.dumps(viewer))
    print(f"done in {raw['wall_time_s']}s; {len(rows)} main rows, {len(bp_rows)} breakpoint rows")


if __name__ == "__main__":
    main()
