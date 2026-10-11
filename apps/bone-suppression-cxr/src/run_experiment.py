"""Exp1: CPU latency + stability of the bone decomposition, seeds from config.yaml.

Loads one model at a time (each TorchScript trace is ~420 MB; peak RSS ~1.2 GB per model on CPU).
Writes raw JSON + .npy work files under work/ and experiments/exp1_cpu_stability/raw_*.json.
"""
import argparse, json, os, platform, sys, time, gc
import numpy as np, yaml, torch
sys.path.insert(0, os.path.dirname(__file__))
import bsx

ap = argparse.ArgumentParser()
ap.add_argument("--config", default=os.path.join(bsx.APP, "experiments/exp1_cpu_stability/config.yaml"))
ap.add_argument("--skip-breakpoints", action="store_true")
a = ap.parse_args()
cfg = yaml.safe_load(open(a.config))
EXP = os.path.dirname(os.path.abspath(a.config))
torch.set_num_threads(int(cfg["runtime"]["torch_threads"]))
os.makedirs(bsx.WORK, exist_ok=True)
sigma = float(cfg["perturbation"]["sigma"])
seeds = cfg["seeds"]

imgs = {}
for i, it in enumerate(cfg["inputs"]):
    imgs[it["id"]] = dict(idx=i, kind=it["kind"], x=bsx.preprocess(bsx.read_gray(bsx.fetch(it["src"]))))

env = dict(python=platform.python_version(), torch=torch.__version__, threads=torch.get_num_threads(),
           cpu=platform.processor() or platform.machine(), nproc=os.cpu_count(),
           loadavg_start=os.getloadavg(), started=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
raw = dict(env=env, clean={}, seeds={str(s): {} for s in seeds}, breakpoints=[])

# Noisy inputs, fixed per (seed, image)
noisy = {s: {k: bsx.add_noise(v["x"], sigma, np.random.default_rng(s * 1000 + v["idx"])) for k, v in imgs.items()} for s in seeds}

# ------------------------- phase A: bone model -------------------------
t0 = time.perf_counter(); bone_m = bsx.load_model("bone"); raw["load_s_bone"] = time.perf_counter() - t0
_, raw["cold_s_bone"] = bsx.run(bone_m, imgs["JPCNN001"]["x"])          # warmup (excluded)
clean_bone, soft_clean = {}, {}
for k, v in imgs.items():
    b, t = bsx.run(bone_m, v["x"]); b2, _ = bsx.run(bone_m, v["x"])
    clean_bone[k] = b; soft_clean[k] = v["x"] - b
    raw["clean"][k] = dict(kind=v["kind"], t_bone=t, deterministic_max_abs=float(np.abs(b - b2).max()),
                           bone_min=float(b.min()), bone_max=float(b.max()), bone_mean=float(b.mean()),
                           looks_inverted=bsx.looks_inverted(v["x"]))
    np.save(os.path.join(bsx.WORK, f"{k}_full.npy"), v["x"]); np.save(os.path.join(bsx.WORK, f"{k}_bone.npy"), b)
    print("clean", k, f"{t:.2f}s", flush=True)

# precondition: reproduce authors' published bone PNGs
import cv2
for k in ["JPCLN030", "JPCNN001"]:
    ref = cv2.imread(bsx.fetch(f"hf:{bsx.MODEL_REPO}/examples/outputs/{k}_bone.png"), cv2.IMREAD_GRAYSCALE).astype(np.float32)
    raw["clean"][k]["ref_mae_8bit"] = float(np.abs(bsx.to_u8(clean_bone[k]).astype(np.float32) - ref).mean())

noisy_bone = {}
for s in seeds:
    noisy_bone[s] = {}
    for k in imgs:
        b, t = bsx.run(bone_m, noisy[s][k]); noisy_bone[s][k] = b
        raw["seeds"][str(s)][k] = dict(t_bone=t, ssim_bone=bsx.ssim(clean_bone[k], b), psnr_bone=bsx.psnr(clean_bone[k], b),
                                       max_abs_diff_bone=float(np.abs(clean_bone[k] - b).max()))
        print("seed", s, k, f"ssim={raw['seeds'][str(s)][k]['ssim_bone']:.4f} t={t:.2f}s", flush=True)

if not a.skip_breakpoints:
    bp = cfg["breakpoints"]; fam = bp["families"]
    ops = ([("noise_sigma", v, lambda x, v=v: bsx.add_noise(x, v, np.random.default_rng(bp["seed"]))) for v in fam["noise_sigma"]]
           + [("lowres_px", v, lambda x, v=v: bsx.lowres(x, v)) for v in fam["lowres_px"]]
           + [("gamma", v, lambda x, v=v: bsx.gamma(x, v)) for v in fam["gamma"]]
           + [("contrast", v, lambda x, v=v: bsx.contrast(x, v)) for v in fam["contrast"]]
           + [("rotation_deg", v, lambda x, v=v: bsx.rotate(x, v)) for v in fam["rotation_deg"]]
           + [("jpeg_quality", v, lambda x, v=v: bsx.jpeg(x, v)) for v in fam["jpeg_quality"]]
           + [("inverted", True, bsx.invert)])
    for k in bp["images"]:
        for f, v, op in ops:
            xp = op(imgs[k]["x"]); b, t = bsx.run(bone_m, xp)
            refb = bsx.rotate(clean_bone[k], v) if f == "rotation_deg" else clean_bone[k]
            row = dict(image=k, family=f, level=v, ssim_bone=bsx.ssim(refb, b), psnr_bone=bsx.psnr(refb, b),
                       bone_mean=float(b.mean()), bone_max=float(b.max()), looks_inverted=bsx.looks_inverted(xp), t_bone=t)
            raw["breakpoints"].append(row)
            print("bp", k, f, v, f"ssim={row['ssim_bone']:.4f}", flush=True)
            if k == bp["images"][1] and f in ("noise_sigma", "rotation_deg", "lowres_px", "inverted", "gamma"):
                np.save(os.path.join(bsx.WORK, f"bp_{k}_{f}_{v}_in.npy"), xp); np.save(os.path.join(bsx.WORK, f"bp_{k}_{f}_{v}_bone.npy"), b)

del bone_m; gc.collect()

# ------------------------- phase B: lung model -------------------------
t0 = time.perf_counter(); lung_m = bsx.load_model("lung"); raw["load_s_lung"] = time.perf_counter() - t0
_, raw["cold_s_lung"] = bsx.run(lung_m, soft_clean["JPCNN001"])
clean_lung = {}
for k in imgs:
    l, t = bsx.run(lung_m, soft_clean[k]); clean_lung[k] = l
    raw["clean"][k]["t_lung"] = t
    np.save(os.path.join(bsx.WORK, f"{k}_lung.npy"), l)
for s in seeds:
    for k in imgs:
        l, t = bsx.run(lung_m, noisy[s][k] - noisy_bone[s][k])
        r = raw["seeds"][str(s)][k]; r["t_lung"] = t; r["ssim_lung"] = bsx.ssim(clean_lung[k], l)
        r["latency_s_pipeline"] = r["t_bone"] + t
        print("lung seed", s, k, f"ssim={r['ssim_lung']:.4f} t={t:.2f}s", flush=True)

import resource
raw["env"]["maxrss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
raw["env"]["loadavg_end"] = os.getloadavg(); raw["env"]["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
json.dump(raw, open(os.path.join(EXP, "raw.json"), "w"), indent=1, default=float)
print("wrote", os.path.join(EXP, "raw.json"))
