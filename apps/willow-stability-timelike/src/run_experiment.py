"""Official run, breakpoints, and the report-only adaptive-stopping bonus."""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pymatching
import stim
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.circuit import Noise, build_stability_circuit  # noqa: E402
from src.decode import (  # noqa: E402
    build_decoder,
    per_cycle_error,
    rule_of_three_upper,
    sample_correlated,
    sample_pauli,
)
from src.leakage import Leakage, apply_mask, burst_mask  # noqa: E402

EXP = ROOT / "experiments" / "exp1_cpu_stability"


def load_config() -> dict:
    with (EXP / "config.yaml").open() as handle:
        return yaml.safe_load(handle)


def _record(
    study: str,
    variant: str,
    distance: int,
    cycles: int,
    seed: int,
    shots: int,
    fails: int,
    p: float,
    p_start: float,
    tau: float,
    clear_every: int | None,
) -> dict:
    p_shot = fails / shots
    return {
        "study": study,
        "variant": variant,
        "distance": distance,
        "cycles": cycles,
        "seed": seed,
        "shots": shots,
        "fails": fails,
        "p_L": p_shot,
        "eps": per_cycle_error(p_shot, cycles),
        "p": p,
        "p_start": p_start,
        "tau": tau,
        "clear_every": clear_every,
    }


def _stim_seed(seed: int, distance: int, cycles: int, variant: str) -> int:
    tag = 0 if variant == "pauli" else 1
    return 17 + seed * 100_000 + distance * 1_000 + cycles * 10 + tag


def run_one(task: dict) -> dict:
    variant = task["variant"]
    distance = int(task["distance"])
    cycles = int(task["cycles"])
    seed = int(task["seed"])
    shots = int(task["shots"])
    p = float(task["p"])
    p_start = float(task.get("p_start") or 0.0)
    tau = float(task.get("tau") or 0.0)
    clear = task.get("clear_every")
    clear_every = None if clear in (None, 0) else int(clear)
    decoder = build_decoder(distance, cycles, p, close_data=True)
    stim_seed = _stim_seed(seed, distance, cycles, variant)
    if variant == "pauli":
        fails, used = sample_pauli(decoder, shots, stim_seed)
    else:
        fails, used = sample_correlated(
            decoder,
            shots,
            stim_seed,
            Leakage(p_start, tau, clear_every if variant == "walking" else None),
        )
    return _record(
        task["study"], variant, distance, cycles, seed, used, fails, p, p_start, tau,
        2 if variant == "walking" else None,
    )


def _tasks_official(cfg: dict) -> list[dict]:
    p = float(cfg["physical_error_rate"])
    leak = cfg["leakage"]
    shots = cfg["shots"]
    tasks: list[dict] = []
    for seed in cfg["seeds"]:
        for distance in cfg["distances"]:
            tasks.append({
                "study": "lambda",
                "variant": "pauli",
                "distance": distance,
                "cycles": int(cfg["lambda_cycles"][distance]),
                "seed": seed,
                "shots": int(shots["lambda"][distance]),
                "p": p,
            })
            tasks.append({
                "study": "fixed_cycle",
                "variant": "pauli",
                "distance": distance,
                "cycles": int(cfg["fixed_cycles"]),
                "seed": seed,
                "shots": int(shots["fixed_cycle"]),
                "p": p,
            })
        for cycles in cfg["curve_cycles"]:
            for variant, n_shots, p_start, tau in (
                ("pauli", shots["curve_pauli"], 0.0, 0.0),
                ("leakage", shots["curve_leakage"], leak["p_start"], leak["tau"]),
                ("walking", shots["curve_leakage"], leak["p_start"], leak["tau"]),
            ):
                tasks.append({
                    "study": "curve",
                    "variant": variant,
                    "distance": int(cfg["curve_distance"]),
                    "cycles": int(cycles),
                    "seed": seed,
                    "shots": int(n_shots),
                    "p": p,
                    "p_start": p_start,
                    "tau": tau,
                })
    return tasks


def _tasks_breakpoints(cfg: dict) -> list[dict]:
    leak = cfg["leakage"]
    shots = cfg["shots"]
    tasks: list[dict] = []
    seed = int(cfg["seeds"][0])
    for p in cfg["breakpoints"]["p"]:
        for distance in cfg["distances"]:
            tasks.append({
                "study": "breakpoint_p",
                "variant": "pauli",
                "distance": distance,
                "cycles": int(cfg["lambda_cycles"][distance]),
                "seed": seed,
                "shots": int(shots["breakpoint_lambda"]),
                "p": float(p),
            })
    for p_start in cfg["breakpoints"]["p_start"]:
        for variant in ("leakage", "walking"):
            tasks.append({
                "study": "breakpoint_leak",
                "variant": variant,
                "distance": int(cfg["curve_distance"]),
                "cycles": int(cfg["knee"]["longest_cycle"]),
                "seed": seed,
                "shots": int(shots["breakpoint_walk"]),
                "p": float(cfg["physical_error_rate"]),
                "p_start": float(p_start),
                "tau": float(leak["tau"]),
            })
    for tau in cfg["breakpoints"]["tau"]:
        for variant in ("leakage", "walking"):
            tasks.append({
                "study": "breakpoint_tau",
                "variant": variant,
                "distance": int(cfg["curve_distance"]),
                "cycles": int(cfg["knee"]["longest_cycle"]),
                "seed": seed,
                "shots": int(shots["breakpoint_walk"]),
                "p": float(cfg["physical_error_rate"]),
                "p_start": float(leak["p_start"]),
                "tau": float(tau),
            })
    return tasks


def _run_pool(tasks: list[dict], workers: int) -> list[dict]:
    if not tasks:
        return []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        records = list(pool.map(run_one, tasks, chunksize=1))
    return records


def _memory_curve(cycles: list[int], p: float, shots: int, seed: int) -> dict[int, float]:
    out: dict[int, float] = {}
    for cycles_i in cycles:
        circuit = stim.Circuit.generated(
            "surface_code:rotated_memory_z",
            distance=3,
            rounds=cycles_i,
            after_clifford_depolarization=p,
            after_reset_flip_probability=p,
            before_measure_flip_probability=p,
            before_round_data_depolarization=p,
        )
        decoder_dem = circuit.detector_error_model(decompose_errors=True)
        matching = pymatching.Matching.from_detector_error_model(decoder_dem)
        sampler = circuit.compile_detector_sampler(seed=seed)
        detections, observable = sampler.sample(shots, separate_observables=True)
        prediction = matching.decode_batch(detections)
        fails = int(np.sum(prediction[:, 0] != observable[:, 0]))
        out[cycles_i] = fails / shots
    return out


def run_adaptive(cfg: dict) -> dict:
    """Report-only gap stopping on one leaked d=3 shot record per seed."""

    spec = cfg["adaptive"]
    p = float(cfg["physical_error_rate"])
    leak = cfg["leakage"]
    distance = int(spec["distance"])
    cycles = [int(c) for c in spec["cycles"]]
    c_max = max(cycles)
    shots = int(cfg["shots"]["adaptive"])
    threshold = float(spec["gap_threshold"])
    memory = _memory_curve(cycles, p, int(cfg["shots"]["adaptive_memory"]), seed=0)
    per_seed = []
    for seed in cfg["seeds"]:
        long = build_decoder(distance, c_max, p, close_data=False)
        stim_seed = _stim_seed(seed, distance, c_max, "leakage")
        measurements = long.experiment.circuit.compile_sampler(seed=stim_seed).sample(shots)
        rng = np.random.default_rng(stim_seed + 1_000_003)
        mask = burst_mask(
            shots,
            len(long.experiment.z_tile_index),
            c_max,
            Leakage(float(leak["p_start"]), float(leak["tau"]), None),
            rng,
        )
        measurements = apply_mask(measurements, mask, long.experiment)
        gaps = np.zeros((shots, len(cycles)))
        wrong = np.zeros((shots, len(cycles)), dtype=bool)
        for index, cycles_i in enumerate(cycles):
            short = build_decoder(distance, cycles_i, p, close_data=False)
            width = cycles_i * short.experiment.n_tiles
            subset = measurements[:, :width]
            detections, observable = short.experiment.circuit.compile_m2d_converter().convert(
                measurements=subset, separate_observables=True
            )
            prediction, gap = short.complementary_gap(detections, observable)
            gaps[:, index] = gap
            wrong[:, index] = prediction != observable[:, 0]
        per_seed.append({
            "seed": seed,
            "cycles": cycles,
            "gap": gaps,
            "wrong": wrong,
            "memory_p": memory,
            "threshold": threshold,
        })
    return {"per_seed": per_seed, "gap_sweep": [float(t) for t in spec["gap_sweep"]]}


def _adaptive_summary(blob: dict) -> dict:
    """Collapse per-shot gaps into stopping-time and spacetime numbers."""

    sweep_rows = []
    headline = None
    for threshold in blob["gap_sweep"]:
        stop_times = []
        stab_fail = []
        st_err = []
        for seed_blob in blob["per_seed"]:
            cycles = seed_blob["cycles"]
            gaps = seed_blob["gap"]
            wrong = seed_blob["wrong"]
            memory = seed_blob["memory_p"]
            for shot in range(gaps.shape[0]):
                chosen = len(cycles) - 1
                for index, cycles_i in enumerate(cycles):
                    if gaps[shot, index] >= threshold:
                        chosen = index
                        break
                c_stop = cycles[chosen]
                failed = bool(wrong[shot, chosen])
                stop_times.append(c_stop)
                stab_fail.append(failed)
                st_err.append(0.5 * (memory[c_stop] + float(failed)))
        fixed = []
        # Fixed-cycle spacetime uses the pooled stability failure rate.
        for index, cycles_i in enumerate(blob["per_seed"][0]["cycles"]):
            mem = blob["per_seed"][0]["memory_p"][cycles_i]
            fails = 0
            total = 0
            for seed_blob in blob["per_seed"]:
                fails += int(np.sum(seed_blob["wrong"][:, index]))
                total += seed_blob["wrong"].shape[0]
            p_stab = fails / total
            fixed.append({
                "cycles": cycles_i,
                "p_stab": p_stab,
                "p_mem": mem,
                "p_st": 0.5 * (mem + p_stab),
            })
        best = min(fixed, key=lambda row: row["p_st"])
        mean_c = float(np.mean(stop_times))
        p_st = float(np.mean(st_err))
        row = {
            "threshold": threshold,
            "mean_cycles": mean_c,
            "p_stab": float(np.mean(stab_fail)),
            "p_st": p_st,
            "best_fixed_cycles": best["cycles"],
            "best_fixed_p_st": best["p_st"],
            "spacetime_change": (best["p_st"] - p_st) / best["p_st"] if best["p_st"] else None,
            "time_change": (best["cycles"] - mean_c) / best["cycles"],
            "fixed": fixed,
        }
        # time_change is (c_fixed - c_adaptive) / c_fixed: positive means fewer cycles.
        sweep_rows.append(row)
        if abs(threshold - blob["per_seed"][0]["threshold"]) < 1e-9:
            headline = row
    return {"headline": headline, "sweep": sweep_rows}


def _jsonable_adaptive(summary: dict) -> dict:
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--breakpoints", action="store_true")
    parser.add_argument("--skip-adaptive", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    cfg = load_config()
    started = time.perf_counter()
    if args.smoke:
        cfg = copy.deepcopy(cfg)
        cfg["seeds"] = [0]
        cfg["shots"]["lambda"] = {3: 2000, 5: 2000, 7: 2000}
        cfg["shots"]["fixed_cycle"] = 1500
        cfg["shots"]["curve_pauli"] = 1500
        cfg["shots"]["curve_leakage"] = 800
        cfg["shots"]["adaptive"] = 200
        cfg["shots"]["adaptive_memory"] = 2000
        cfg["curve_cycles"] = [2, 4, 8, 16]
        tasks = _tasks_official(cfg)
        records = _run_pool(tasks, args.workers)
        adaptive = None if args.skip_adaptive else _adaptive_summary(run_adaptive(cfg))
        payload = {
            "smoke": True,
            "seconds": time.perf_counter() - started,
            "records": records,
            "adaptive": adaptive,
        }
        (EXP / "smoke.json").write_text(json.dumps(payload, indent=2) + "\n")
        print(f"smoke wrote {len(records)} records in {payload['seconds']:.1f}s")
        return

    if args.breakpoints:
        tasks = _tasks_breakpoints(cfg)
        print(f"breakpoint tasks {len(tasks)}", flush=True)
        records = _run_pool(tasks, args.workers)
        payload = {"seconds": time.perf_counter() - started, "records": records}
        (EXP / "breakpoints_raw.json").write_text(json.dumps(payload, indent=2) + "\n")
        print(f"breakpoints wrote {len(records)} records in {payload['seconds']:.1f}s")
        return

    tasks = _tasks_official(cfg)
    print(f"official tasks {len(tasks)}", flush=True)
    records = _run_pool(tasks, args.workers)
    adaptive = None if args.skip_adaptive else _adaptive_summary(run_adaptive(cfg))
    payload = {
        "smoke": False,
        "seconds": time.perf_counter() - started,
        "config_thresholds": cfg["thresholds"],
        "records": records,
        "adaptive": _jsonable_adaptive(adaptive) if adaptive else None,
        "quoted": {
            "mu_T": "1.47 ± 0.25",
            "mu_T_pauli_model": "1.97 ± 0.44",
            "walking_floor": "≤ 3.20e-8 at 60 cycles",
            "paper_vs_standard": "two orders of magnitude below the standard construction",
            "post_vs_literature": "four orders of magnitude lower than any experiment in the literature",
            "adaptive_spacetime": "38% lower spacetime logical error",
            "adaptive_time": "34% lower mean stopping time",
        },
    }
    (EXP / "raw.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(f"official wrote {len(records)} records in {payload['seconds']:.1f}s")


if __name__ == "__main__":
    main()
