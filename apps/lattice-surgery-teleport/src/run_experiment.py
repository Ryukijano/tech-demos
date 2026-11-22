"""Official seeds, then the breakpoint grid. Reads config.yaml; does not edit it."""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.circuit import Basis  # noqa: E402
from src.decode import count_failures  # noqa: E402

EXP = ROOT / "experiments" / "exp1_cpu_teleport"


def _load() -> dict:
    with (EXP / "config.yaml").open() as handle:
        cfg = yaml.safe_load(handle)
    return cfg


def _shots(table: dict, distance: int) -> int:
    if distance in table:
        return int(table[distance])
    return int(table[str(distance)])


def _run(spec: dict) -> dict:
    basis: Basis = spec["basis"]
    out = count_failures(
        distance=int(spec["distance"]),
        basis=basis,
        p=float(spec["p"]),
        init_rounds=int(spec["init_rounds"]),
        merge_rounds=int(spec["merge_rounds"]),
        shots=int(spec["shots"]),
        seed=int(spec["seed"]),
        chunk_size=int(spec["chunk_size"]),
    )
    row = dict(spec)
    row.update(out)
    rate = out["fails"] / out["shots"]
    print(
        f"{row['study']} d={row['distance']} {row['basis']} p={row['p']} "
        f"init={row['init_rounds']} merge={row['merge_rounds']} seed={row['seed']} "
        f"fails={out['fails']}/{out['shots']} ({rate:.5f}) {out['seconds']}s",
        flush=True,
    )
    return row


def _scaling_jobs(cfg: dict) -> list[dict]:
    jobs = []
    for distance in cfg["distances"]:
        for basis in cfg["bases"]:
            for seed in cfg["seeds"]:
                jobs.append(
                    {
                        "study": "scaling",
                        "distance": int(distance),
                        "basis": basis,
                        "p": float(cfg["physical_error_rate"]),
                        "init_rounds": int(distance),
                        "merge_rounds": int(distance),
                        "shots": _shots(cfg["shots"], int(distance)),
                        "seed": int(seed),
                        "chunk_size": int(cfg["chunk_size"]),
                    }
                )
    return jobs


def _breakpoint_jobs(cfg: dict) -> list[dict]:
    bp = cfg["breakpoints"]
    seed = int(bp["seed"])
    chunk = int(cfg["chunk_size"])
    jobs: list[dict] = []
    for distance in cfg["distances"]:
        for basis in cfg["bases"]:
            for p in bp["p"]:
                jobs.append(
                    {
                        "study": "p_sweep",
                        "distance": int(distance),
                        "basis": basis,
                        "p": float(p),
                        "init_rounds": int(distance),
                        "merge_rounds": int(distance),
                        "shots": _shots(bp["shots"], int(distance)),
                        "seed": seed,
                        "chunk_size": chunk,
                    }
                )
    for distance in (3, 5):
        for basis in cfg["bases"]:
            for p in bp["merge_p"]:
                for merge_rounds in list(bp["merge_rounds"]) + [distance]:
                    jobs.append(
                        {
                            "study": "merge_rounds",
                            "distance": distance,
                            "basis": basis,
                            "p": float(p),
                            "init_rounds": distance,
                            "merge_rounds": int(merge_rounds),
                            "shots": _shots(bp["shots"], distance),
                            "seed": seed,
                            "chunk_size": chunk,
                        }
                    )
            for p in bp["few_rounds_p"]:
                jobs.append(
                    {
                        "study": "few_rounds",
                        "distance": distance,
                        "basis": basis,
                        "p": float(p),
                        "init_rounds": 1,
                        "merge_rounds": 1,
                        "shots": _shots(bp["shots"], distance),
                        "seed": seed,
                        "chunk_size": chunk,
                    }
                )
    paper = bp["paper_d3"]
    for basis in cfg["bases"]:
        for p in paper["p"]:
            jobs.append(
                {
                    "study": "paper_d3_rounds",
                    "distance": 3,
                    "basis": basis,
                    "p": float(p),
                    "init_rounds": int(paper["init_rounds"]),
                    "merge_rounds": int(paper["merge_rounds"]),
                    "shots": _shots(bp["shots"], 3),
                    "seed": seed,
                    "chunk_size": chunk,
                }
            )
    return jobs


def _smoke_jobs(cfg: dict) -> list[dict]:
    jobs = []
    for basis in cfg["bases"]:
        jobs.append(
            {
                "study": "smoke",
                "distance": 3,
                "basis": basis,
                "p": float(cfg["physical_error_rate"]),
                "init_rounds": 3,
                "merge_rounds": 3,
                "shots": 400,
                "seed": 0,
                "chunk_size": 400,
            }
        )
    return jobs


def _execute(jobs: list[dict], workers: int) -> list[dict]:
    # Largest distances first so the pool is not stuck behind them at the end.
    ordered = sorted(jobs, key=lambda job: (-int(job["distance"]), job["basis"], float(job["p"])))
    ctx_workers = max(1, workers)
    if ctx_workers == 1:
        return [_run(job) for job in ordered]
    with ProcessPoolExecutor(max_workers=ctx_workers) as pool:
        return list(pool.map(_run, ordered))


def _dump(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--breakpoints", action="store_true")
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()
    cfg = _load()
    workers = int(args.workers if args.workers is not None else cfg["workers"])
    if args.smoke:
        rows = _execute(_smoke_jobs(cfg), workers)
        _dump(EXP / "smoke.json", {"records": rows, "note": "code-path smoke, not the official seeds"})
        print(f"wrote {EXP / 'smoke.json'}")
        return
    if args.breakpoints:
        rows = _execute(_breakpoint_jobs(cfg), workers)
        _dump(EXP / "breakpoints_raw.json", {"records": rows})
        print(f"wrote {EXP / 'breakpoints_raw.json'} ({len(rows)} rows)")
        return
    rows = _execute(_scaling_jobs(cfg), workers)
    _dump(
        EXP / "raw.json",
        {
            "records": rows,
            "physical_error_rate": cfg["physical_error_rate"],
            "hypothesis": cfg["hypothesis"],
            "thresholds": cfg["thresholds"],
        },
    )
    print(f"wrote {EXP / 'raw.json'} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
