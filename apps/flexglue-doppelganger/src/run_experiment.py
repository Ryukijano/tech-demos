"""Exp1: CPU AUROC of FlexGlue's covisibility head vs match-count baselines.

Writes experiments/exp1_cpu_covisibility/raw.json. Images and ONNX weights
stay under work/ (git-ignored). Thresholds are read from config.yaml and
are not edited here.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(__file__))
import fgx

ap = argparse.ArgumentParser()
ap.add_argument("--config", default=os.path.join(fgx.APP, "experiments/exp1_cpu_covisibility/config.yaml"))
ap.add_argument("--smoke", action="store_true", help="One disclosed pair after the pool is built")
ap.add_argument("--skip-breakpoints", action="store_true")
args = ap.parse_args()

cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
EXP = os.path.dirname(os.path.abspath(args.config))
fgx.set_thread_env(int(cfg["runtime"]["intra_op_threads"]))
fgx.self_check()
fgx.ensure_dirs()

print("Selecting Doppelgangers pairs (no matcher yet) …", flush=True)
pool, skip_log = fgx.materialize_pool(cfg)
anchor = fgx.fetch_anchor(cfg)
manifest = {
    "rule": "alphabetical scenes, first license-clean pos and neg that download",
    "n_pairs": len(pool),
    "pairs": [
        {k: v for k, v in item.items() if k != "paths"} | {"paths": [os.path.relpath(p, fgx.APP) for p in item["paths"]]}
        for item in pool
    ],
    "skips": skip_log,
}
with open(os.path.join(EXP, "pool_manifest.json"), "w", encoding="utf-8") as handle:
    json.dump(manifest, handle, indent=1)
print(f"pool: {len(pool)} pairs, manifest written", flush=True)

rt = cfg["runtime"]


def eval_pair(model: fgx.OnnxFlexGlue, rgb0: np.ndarray, rgb1: np.ndarray, seed: int, key0: str, key1: str) -> dict:
    k0, d0, _sift0_now, sift0_s, sift0_cached = model.detect(rgb0, int(rt["max_detect"]))
    k1, d1, _sift1_now, sift1_s, sift1_cached = model.detect(rgb1, int(rt["max_detect"]))
    k0s, d0s = fgx.subsample(k0, d0, int(rt["num_keypoints"]), int(rt["candidate_keypoints"]), seed, key0)
    k1s, d1s = fgx.subsample(k1, d1, int(rt["num_keypoints"]), int(rt["candidate_keypoints"]), seed, key1)
    f0, _d0_now, desc0_s, desc0_cached = model.describe(rgb0)
    f1, _d1_now, desc1_s, desc1_cached = model.describe(rgb1)
    matches, scores, covis, match_s = model.match(
        f0,
        f1,
        k0s,
        k1s,
        (rgb0.shape[1], rgb0.shape[0]),
        (rgb1.shape[1], rgb1.shape[0]),
        float(rt["filter_threshold"]),
        float(rt["max_distance_px"]),
        float(rt["min_covisibility"]),
    )
    if len(matches):
        pts0 = k0s[matches[:, 0]]
        pts1 = k1s[matches[:, 1]]
    else:
        pts0 = np.zeros((0, 2), np.float32)
        pts1 = np.zeros((0, 2), np.float32)
    n_in, in_ratio = fgx.fundamental_inliers(pts0, pts1, float(rt["fundamental_reproj_px"]))
    sift_m = fgx.sift_ratio_matches(d0s, d1s, float(rt["sift_ratio"]))
    if len(sift_m):
        s_in, _s_ratio = fgx.fundamental_inliers(k0s[sift_m[:, 0]], k1s[sift_m[:, 1]], float(rt["fundamental_reproj_px"]))
    else:
        s_in = 0
    describe_s = float(desc0_s + desc1_s + sift0_s + sift1_s)
    return dict(
        covisibility=covis,
        n_matches=int(len(matches)),
        n_inliers=int(n_in),
        inlier_ratio=float(in_ratio),
        n_sift=int(len(sift_m)),
        n_sift_inliers=int(s_in),
        n_kpts0=int(len(k0s)),
        n_kpts1=int(len(k1s)),
        match_s=float(match_s),
        describe_s=describe_s,
        uncached_s=float(match_s + describe_s),
        describe_cached=bool(desc0_cached and desc1_cached),
        sift_cached=bool(sift0_cached and sift1_cached),
        kpts0=k0s,
        kpts1=k1s,
        matches=matches,
        scores=scores,
    )


def public_row(row: dict) -> dict:
    return {k: v for k, v in row.items() if k not in {"kpts0", "kpts1", "matches", "scores"}}


def save_viz(pair_id: str, tag: str, row: dict) -> str:
    path = os.path.join(fgx.WORK, "viz", f"{pair_id}_{tag}.npz")
    np.savez_compressed(
        path,
        kpts0=row["kpts0"],
        kpts1=row["kpts1"],
        matches=row["matches"],
        scores=row["scores"] if len(row["scores"]) else np.zeros(0, np.float32),
    )
    return os.path.relpath(path, fgx.APP)


print("Loading ONNX sessions (CPU) …", flush=True)
t_load = time.perf_counter()
model = fgx.OnnxFlexGlue(cfg)
load_s = time.perf_counter() - t_load
print(f"sessions ready in {load_s:.1f}s, descriptor input {model.width}x{model.height}", flush=True)

env = dict(
    python=platform.python_version(),
    cpu=platform.processor() or platform.machine(),
    nproc=os.cpu_count(),
    intra_op_threads=int(rt["intra_op_threads"]),
    load_s=load_s,
    loadavg_start=list(os.getloadavg()),
    started=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    descriptor_hw=[model.height, model.width],
)

if args.smoke:
    item = next(p for p in pool if p["label"] == 1)
    rgb0, rgb1 = fgx.load_rgb(item["paths"][0]), fgx.load_rgb(item["paths"][1])
    row = eval_pair(model, rgb0, rgb1, 0, item["id"] + "_a", item["id"] + "_b")
    smoke = dict(env=env, pair=item["id"], label=item["label"], **public_row(row))
    with open(os.path.join(EXP, "smoke.json"), "w", encoding="utf-8") as handle:
        json.dump(smoke, handle, indent=1)
    print(
        f"SMOKE {item['id']}: covis={row['covisibility']:.3f} matches={row['n_matches']} "
        f"sift={row['n_sift']} match_s={row['match_s']:.2f} describe_s={row['describe_s']:.2f}",
        flush=True,
    )
    sys.exit(0)

# Warm the sessions on the upstream Berlin anchor so eval timings are not the first graph run.
cold = None
if anchor is not None:
    a0, a1 = fgx.load_rgb(anchor["paths"][0]), fgx.load_rgb(anchor["paths"][1])
    cold_row = eval_pair(model, a0, a1, 0, "anchor_a", "anchor_b")
    cold = dict(pair=anchor["id"], label=anchor["label"], in_metric=False, **public_row(cold_row))
    cold["viz"] = save_viz(anchor["id"], "seed0", cold_row)
    cold["image0"] = os.path.relpath(anchor["paths"][0], fgx.APP)
    cold["image1"] = os.path.relpath(anchor["paths"][1], fgx.APP)
    print(
        f"anchor {anchor['id']}: covis={cold_row['covisibility']:.3f} matches={cold_row['n_matches']} "
        f"match_s={cold_row['match_s']:.2f} describe_s={cold_row['describe_s']:.2f}",
        flush=True,
    )

seeds_out: dict[str, dict] = {}
viz_index = []
for seed in cfg["seeds"]:
    scene_ids = fgx.sample_scene_ids(pool, int(cfg["dataset"]["eval_scenes_per_seed"]), int(seed))
    chosen = [item for item in pool if item["scene"] in scene_ids]
    print(f"seed {seed} scenes: {scene_ids}", flush=True)
    rows = []
    for item in chosen:
        rgb0, rgb1 = fgx.load_rgb(item["paths"][0]), fgx.load_rgb(item["paths"][1])
        row = eval_pair(model, rgb0, rgb1, int(seed), item["id"] + "_a", item["id"] + "_b")
        stored = dict(
            id=item["id"],
            scene=item["scene"],
            label=int(item["label"]),
            image0=os.path.relpath(item["paths"][0], fgx.APP),
            image1=os.path.relpath(item["paths"][1], fgx.APP),
            **public_row(row),
        )
        if int(seed) == 0:
            stored["viz"] = save_viz(item["id"], "seed0", row)
            viz_index.append({k: stored[k] for k in ("id", "label", "scene", "viz", "image0", "image1", "covisibility", "n_matches", "n_inliers", "n_sift")})
        rows.append(stored)
        print(
            f"  {item['id']} y={item['label']} covis={row['covisibility']:.3f} "
            f"fg={row['n_matches']} in={row['n_inliers']} sift={row['n_sift']} "
            f"match={row['match_s']:.2f}s",
            flush=True,
        )
    seeds_out[str(seed)] = {"scenes": scene_ids, "pairs": rows}

breakpoints = []
if not args.skip_breakpoints:
    seed = int(cfg["breakpoints"]["seed"])
    sample = seeds_out[str(seed)]["pairs"]
    pos = next(p for p in sample if p["label"] == 1)
    neg = next(p for p in sample if p["label"] == 0)
    bp_cfg = cfg["breakpoints"]

    def run_modified(base: dict, rgb0: np.ndarray, rgb1: np.ndarray, kind: str, level: str) -> None:
        # Fresh keys so perturbations do not collide with cached clean features.
        row = eval_pair(model, rgb0, rgb1, seed, f"{base['id']}_a_{kind}_{level}", f"{base['id']}_b_{kind}_{level}")
        rss, hwm = fgx.rss_mb()
        rec = dict(
            kind=kind,
            level=level,
            id=base["id"],
            label=int(base["label"]),
            covisibility=row["covisibility"],
            n_matches=row["n_matches"],
            n_inliers=row["n_inliers"],
            n_sift=row["n_sift"],
            match_s=row["match_s"],
            describe_s=row["describe_s"],
            uncached_s=row["uncached_s"],
            rss_mb=rss,
            hwm_mb=hwm,
        )
        if kind == "rotation":
            rec["viz"] = save_viz(base["id"], f"rot_{level}", row)
            # Keep the perturbed image so the video can redraw lines.
            pert = os.path.join(fgx.WORK, "pert", f"{base['id']}_rot_{level}.png")
            fgx._save_png(rgb1, pert)
            rec["image0"] = base["image0"]
            rec["image1"] = os.path.relpath(pert, fgx.APP)
        breakpoints.append(rec)
        print(
            f"  bp {kind} {level} {base['id']} covis={row['covisibility']:.3f} "
            f"fg={row['n_matches']} match={row['match_s']:.2f}s rss={rss:.0f}MB",
            flush=True,
        )

    for base in (pos, neg):
        rgb0 = fgx.load_rgb(os.path.join(fgx.APP, base["image0"]))
        rgb1 = fgx.load_rgb(os.path.join(fgx.APP, base["image1"]))
        for deg in bp_cfg["rotate_deg"]:
            run_modified(base, rgb0, fgx.rotate_image(rgb1, float(deg)), "rotation", str(int(deg)))
        for scale in bp_cfg["downscale"]:
            run_modified(base, rgb0, fgx.downscale_image(rgb1, float(scale)), "downscale", str(scale))
        for sigma in bp_cfg["blur_sigma"]:
            run_modified(base, rgb0, fgx.blur_image(rgb1, float(sigma)), "blur", str(sigma))
        for amount in bp_cfg["texture_mix"]:
            run_modified(base, rgb0, fgx.texture_mix(rgb1, float(amount)), "texture", str(amount))
    # Resolution / memory: positive pair only, both images resized. Descriptor stays 512.
    rgb0 = fgx.load_rgb(os.path.join(fgx.APP, pos["image0"]))
    rgb1 = fgx.load_rgb(os.path.join(fgx.APP, pos["image1"]))
    for side in bp_cfg["long_side_px"]:
        run_modified(
            pos,
            fgx.resize_long_side(rgb0, int(side)),
            fgx.resize_long_side(rgb1, int(side)),
            "resolution",
            str(int(side)),
        )

raw = dict(
    env=env,
    anchor=cold,
    seeds=seeds_out,
    breakpoints=breakpoints,
    viz_index=viz_index,
    rss_end_mb=list(fgx.rss_mb()),
    finished=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
)
with open(os.path.join(EXP, "raw.json"), "w", encoding="utf-8") as handle:
    json.dump(raw, handle, indent=1)
print("wrote", os.path.join(EXP, "raw.json"), flush=True)
