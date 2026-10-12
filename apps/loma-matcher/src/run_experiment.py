"""Exp1: CPU latency + determinism (LoMa-B) and rotation breakpoints (LoMa-B vs LoMa-R).

Writes experiments/exp1_cpu_match/raw.json. Intermediates under work/ (git-ignored).
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import resource
import sys
import time

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(__file__))
import lmx

ap = argparse.ArgumentParser()
ap.add_argument("--config", default=os.path.join(lmx.APP, "experiments/exp1_cpu_match/config.yaml"))
ap.add_argument("--skip-breakpoints", action="store_true")
ap.add_argument("--smoke", action="store_true", help="One disclosed LoMa-B match for latency calibration only")
a = ap.parse_args()

cfg = yaml.safe_load(open(a.config))
EXP = os.path.dirname(os.path.abspath(a.config))
torch.set_num_threads(int(cfg["runtime"]["torch_threads"]))
lmx.ensure_dirs()

pairs = {}
for it in cfg["inputs"]:
    pairs[it["id"]] = dict(A=lmx.fetch(it["A"]), B=lmx.fetch(it["B"]))

env = dict(
    python=platform.python_version(),
    torch=torch.__version__,
    threads=torch.get_num_threads(),
    cpu=platform.processor() or platform.machine(),
    nproc=os.cpu_count(),
    cuda=torch.cuda.is_available(),
    loadavg_start=os.getloadavg(),
    started=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
)
raw = dict(env=env, load_s={}, cold_s={}, clean={}, seeds={str(s): {} for s in cfg["seeds"]}, breakpoints=[], smoke=None)

nk = int(cfg["runtime"]["num_keypoints"])
ft = float(cfg["runtime"]["filter_threshold"])
magsac = cfg["magsac"]

# ------------------------- optional disclosed smoke -------------------------
if a.smoke:
    print("SMOKE: loading LoMa-B …", flush=True)
    model, load_s = lmx.load_model("LoMa-B", num_keypoints=nk, filter_threshold=ft)
    pid = cfg["inputs"][0]["id"]
    lmx.set_seed(0)
    ka, kb, t = lmx.match_timed(model, pairs[pid]["A"], pairs[pid]["B"])
    nin, ratio = lmx.magsac_inliers(ka, kb, magsac)
    raw["smoke"] = dict(pair=pid, load_s=load_s, latency_s=t, n_matches=len(ka), n_inliers=nin, inlier_ratio=ratio)
    print(f"SMOKE {pid}: load={load_s:.1f}s match={t:.2f}s n={len(ka)} inliers={nin} ratio={ratio:.3f}", flush=True)
    json.dump(raw, open(os.path.join(EXP, "smoke.json"), "w"), indent=1, default=float)
    print("wrote", os.path.join(EXP, "smoke.json"))
    sys.exit(0)

# ------------------------- phase A: LoMa-B clean + seeds -------------------------
print("Loading LoMa-B …", flush=True)
model_b, raw["load_s"]["LoMa-B"] = lmx.load_model("LoMa-B", num_keypoints=nk, filter_threshold=ft)
print(f"LoMa-B loaded in {raw['load_s']['LoMa-B']:.1f}s", flush=True)

# warmup (excluded from latency)
warm_id = cfg["inputs"][0]["id"]
lmx.set_seed(0)
_, _, raw["cold_s"]["LoMa-B"] = lmx.match_timed(model_b, pairs[warm_id]["A"], pairs[warm_id]["B"])
print(f"warmup LoMa-B {warm_id}: {raw['cold_s']['LoMa-B']:.2f}s", flush=True)

# clean reference match (also used for media)
for pid, paths in pairs.items():
    lmx.set_seed(0)
    ka, kb, t = lmx.match_timed(model_b, paths["A"], paths["B"])
    nin, ratio = lmx.magsac_inliers(ka, kb, magsac)
    raw["clean"][pid] = dict(
        model="LoMa-B",
        latency_s_match=t,
        n_matches=int(len(ka)),
        n_inliers=nin,
        inlier_ratio=ratio,
    )
    np.savez_compressed(os.path.join(lmx.WORK, f"{pid}_LoMa-B_clean.npz"), kptsA=ka, kptsB=kb)
    print(f"clean LoMa-B {pid}: t={t:.2f}s n={len(ka)} inliers={nin}", flush=True)

# seeded clean runs (determinism + latency)
for s in cfg["seeds"]:
    for pid, paths in pairs.items():
        lmx.set_seed(int(s))
        ka, kb, t = lmx.match_timed(model_b, paths["A"], paths["B"])
        nin, ratio = lmx.magsac_inliers(ka, kb, magsac)
        raw["seeds"][str(s)][pid] = dict(
            model="LoMa-B",
            latency_s_match=t,
            n_matches=int(len(ka)),
            n_inliers=nin,
            inlier_ratio=ratio,
        )
        print(f"seed {s} LoMa-B {pid}: t={t:.2f}s n={len(ka)}", flush=True)

# ------------------------- phase B: rotation breakpoints -------------------------
if not a.skip_breakpoints:
    bp = cfg["breakpoints"]
    angles = list(bp["angles_deg"])
    # Keep LoMa-B loaded for its rotation angles, then swap to LoMa-R.
    for model_name in bp["models"]:
        if model_name == "LoMa-B":
            model = model_b
        else:
            del model_b
            gc.collect()
            print(f"Loading {model_name} …", flush=True)
            model, raw["load_s"][model_name] = lmx.load_model(model_name, num_keypoints=nk, filter_threshold=ft)
            print(f"{model_name} loaded in {raw['load_s'][model_name]:.1f}s", flush=True)
            # one warm match
            lmx.set_seed(int(bp["seed"]))
            _, _, raw["cold_s"][model_name] = lmx.match_timed(model, pairs[bp["images"][0]]["A"], pairs[bp["images"][0]]["B"])

        for pid in bp["images"]:
            path_a = pairs[pid]["A"]
            path_b0 = pairs[pid]["B"]
            for ang in angles:
                out_b = os.path.join(lmx.WORK, f"{pid}_B_rot{ang}.jpg")
                if ang == 0:
                    path_b = path_b0
                else:
                    path_b = lmx.rotate_image_file(path_b0, ang, out_b)
                lmx.set_seed(int(bp["seed"]))
                ka, kb, t = lmx.match_timed(model, path_a, path_b)
                nin, ratio = lmx.magsac_inliers(ka, kb, magsac)
                row = dict(
                    image=pid,
                    model=model_name,
                    family="rotation_deg",
                    level=ang,
                    n_matches=int(len(ka)),
                    n_inliers=nin,
                    inlier_ratio=ratio,
                    latency_s_match=t,
                )
                raw["breakpoints"].append(row)
                # save keypoints for media (first breakpoint image, all angles × both models)
                if pid == bp["images"][0]:
                    np.savez_compressed(
                        os.path.join(lmx.WORK, f"bp_{pid}_{model_name}_rot{ang}.npz"),
                        kptsA=ka,
                        kptsB=kb,
                        pathA=np.array(path_a),
                        pathB=np.array(path_b),
                    )
                print(
                    f"bp {model_name} {pid} rot={ang}: n={len(ka)} inliers={nin} ratio={ratio:.3f} t={t:.2f}s",
                    flush=True,
                )

raw["env"]["maxrss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
raw["env"]["loadavg_end"] = os.getloadavg()
raw["env"]["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
json.dump(raw, open(os.path.join(EXP, "raw.json"), "w"), indent=1, default=float)
print("wrote", os.path.join(EXP, "raw.json"))
