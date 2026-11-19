"""Pre-registered CPU measurement of stable-envs state storage vs stored frames.

The pass bars live in experiments/exp1_cpu_state_replay/config.yaml and are
applied by src/analyze.py. This script only collects them.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# Thread pools are fixed in config.yaml. Set them before NumPy / Torch / Numba import.
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("NUMBA_NUM_THREADS", "4")

import numpy as np
import torch
import yaml
from PIL import Image

APP = Path(__file__).resolve().parents[1]
EXP = APP / "experiments" / "exp1_cpu_state_replay"
WORK = EXP / "work"


def load_config() -> dict:
    path = EXP / "config.yaml"
    text = path.read_text()
    cfg = yaml.safe_load(text)
    cfg["_sha256"] = hashlib.sha256(text.encode()).hexdigest()
    cfg["_path"] = str(path)
    return cfg


def write_png(task: tuple[str, np.ndarray, int]) -> None:
    path, frame, level = task
    Image.fromarray(frame, mode="RGB").save(path, format="PNG", compress_level=level)


def median(values: list[float]) -> float:
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    if n % 2 == 1:
        return float(ordered[mid])
    return float(0.5 * (ordered[mid - 1] + ordered[mid]))


def sample_actions(spec: dict, num_envs: int, steps: int, seed: int) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    shape = (num_envs, steps, *spec["action_shape"])
    if spec["discrete"]:
        values = rng.integers(0, int(spec["n_actions"]), size=shape)
        return torch.as_tensor(values, dtype=torch.int64)
    values = rng.uniform(float(spec["low"]), float(spec["high"]), size=shape)
    return torch.as_tensor(np.ascontiguousarray(values), dtype=torch.float32)


def open_env(snv, name: str, num_envs: int):
    cls = getattr(snv, name)
    return cls(
        num_envs=num_envs,
        device="cpu",
        backend="auto",
        simulation_backend="auto",
    )


def env_info(env) -> dict:
    frame = getattr(env, "frame_shape", None)
    return {
        "simulation_backend": str(getattr(env, "simulation_backend", "")),
        "render_backend_arg": str(env.backend),
        "size": int(env.size),
        "native_size": int(getattr(env, "native_size", env.size)),
        "frame_shape": [int(x) for x in frame] if frame is not None else None,
    }


def collect(snv, name: str, spec: dict, num_envs: int, steps: int, seed: int):
    env = open_env(snv, name, num_envs)
    try:
        info = env_info(env)
        env.reset(seed=seed)
        actions = sample_actions(spec, num_envs, steps, seed)
        out = env.step(actions, frames=None)
        states = out["states"].detach().clone()
        reference = env.render(states.reshape(-1)).cpu().contiguous()
        return states, reference, info
    finally:
        env.close()


def replay_frames(snv, name: str, states) -> torch.Tensor:
    env = open_env(snv, name, 1)
    try:
        return env.render(states.reshape(-1)).cpu().contiguous()
    finally:
        env.close()


def save_pngs(frames: np.ndarray, directory: Path, level: int, workers: int) -> int:
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True)
    tasks = [(str(directory / f"{i:05d}.png"), frames[i], level) for i in range(frames.shape[0])]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(write_png, tasks, chunksize=8))
    return sum(path.stat().st_size for path in directory.glob("*.png"))


def time_call(fn, reps: int) -> list[float]:
    fn()
    samples = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return samples


def lossless_stats(reference: torch.Tensor, replay: torch.Tensor) -> dict:
    ref = reference.numpy()
    got = replay.numpy()
    if ref.shape != got.shape:
        return {
            "lossless_fraction": 0.0,
            "n_frames": int(ref.shape[0]),
            "n_frames_mismatch": int(ref.shape[0]),
            "n_pixels_mismatch": -1,
            "max_abs": -1,
            "shape_reference": list(ref.shape),
            "shape_replay": list(got.shape),
        }
    diff = np.abs(got.astype(np.int16) - ref.astype(np.int16))
    bad = diff.any(axis=(1, 2, 3))
    n = int(ref.shape[0])
    n_bad = int(bad.sum())
    return {
        "lossless_fraction": float(n - n_bad) / float(n),
        "n_frames": n,
        "n_frames_mismatch": n_bad,
        "n_pixels_mismatch": int((diff > 0).sum()),
        "max_abs": int(diff.max()),
        "shape_reference": list(ref.shape),
        "shape_replay": list(got.shape),
    }


def png_roundtrip_ok(paths: list[Path], reference: np.ndarray) -> bool:
    for index, path in enumerate(paths):
        with Image.open(path) as image:
            got = np.asarray(image)
        if got.shape != reference[index].shape or not np.array_equal(got, reference[index]):
            return False
    return True


def measure_cell(snv, cfg: dict, name: str, spec: dict, num_envs: int, steps: int, seed: int, dest: Path) -> dict:
    states, reference, info = collect(snv, name, spec, num_envs, steps, seed)
    expected = spec["expected_hw"]
    got_hw = [int(reference.shape[1]), int(reference.shape[2])]
    if got_hw != expected:
        raise RuntimeError(f"{name} frame is {got_hw}, config expected {expected}")
    if int(reference.shape[0]) < int(cfg["main"]["min_render_rows"]):
        raise RuntimeError(f"{name} rendered {reference.shape[0]} rows, below min_render_rows")

    dest.mkdir(parents=True, exist_ok=True)
    state_path = dest / "states.pt"
    npz_path = dest / "frames.npz"
    png_dir = dest / "png"
    torch.save(states, state_path)
    ref_np = reference.numpy()
    np.savez(npz_path, frames=ref_np)
    level = int(cfg["storage"]["png_compress_level"])
    workers = int(cfg["timing"]["png_write_threads"])
    png_bytes = save_pngs(ref_np, png_dir, level, workers)
    png_paths = sorted(png_dir.glob("*.png"))

    replay = replay_frames(snv, name, states)
    stats = lossless_stats(reference, replay)
    roundtrip = png_roundtrip_ok(png_paths, ref_np)

    reps = int(cfg["timing"]["reps"])

    def load_and_render() -> None:
        loaded = torch.load(state_path, weights_only=False)
        env = load_and_render.env
        frames = env.render(loaded.reshape(-1))
        load_and_render.touch = int(frames.reshape(-1)[0])

    load_and_render.env = open_env(snv, name, 1)
    try:
        replay_samples = time_call(load_and_render, reps)
    finally:
        load_and_render.env.close()

    def load_npz() -> None:
        loaded = np.load(npz_path)["frames"]
        load_npz.touch = int(np.ascontiguousarray(loaded).reshape(-1)[0])

    def load_png() -> None:
        touch = 0
        for path in png_paths:
            with Image.open(path) as image:
                touch += int(np.asarray(image).reshape(-1)[0])
        load_png.touch = touch

    npz_samples = time_call(load_npz, reps)
    png_samples = time_call(load_png, reps)
    n_frames = int(reference.shape[0])
    replay_fps = n_frames / median(replay_samples)
    npz_fps = n_frames / median(npz_samples)
    png_fps = n_frames / median(png_samples)
    state_bytes = int(state_path.stat().st_size)
    npz_bytes = int(npz_path.stat().st_size)
    return {
        "env": name,
        "seed": int(seed),
        "num_envs": int(num_envs),
        "steps": int(steps),
        "n_frames": n_frames,
        "info": info,
        "state_bytes": state_bytes,
        "png_bytes": int(png_bytes),
        "npz_bytes": npz_bytes,
        "disk_saving_vs_png_pct": 100.0 * (1.0 - state_bytes / png_bytes),
        "disk_saving_vs_npz_pct": 100.0 * (1.0 - state_bytes / npz_bytes),
        "png_roundtrip_ok": bool(roundtrip),
        "replay_seconds": replay_samples,
        "npz_seconds": npz_samples,
        "png_seconds": png_samples,
        "state_replay_fps": replay_fps,
        "npz_load_fps": npz_fps,
        "png_load_fps": png_fps,
        "frame_load_fps": max(npz_fps, png_fps),
        "frame_sha256": hashlib.sha256(ref_np.tobytes()).hexdigest(),
        **stats,
    }


def try_non_native(snv, name: str, spec: dict, size: int) -> dict:
    env = open_env(snv, name, 1)
    try:
        env.reset(seed=0)
        actions = sample_actions(spec, 1, 1, 0)
        out = env.step(actions, frames=None)
        states = out["states"].detach().clone()
        try:
            frames = env.render(states.reshape(-1), size=size)
        except ValueError as exc:
            return {"env": name, "size": size, "accepted": False, "error": str(exc)}
        return {
            "env": name,
            "size": size,
            "accepted": True,
            "shape": [int(x) for x in frames.shape],
        }
    finally:
        env.close()


def upsample_bytes(frames_npz: Path) -> dict:
    frames = np.load(frames_npz)["frames"]
    up = np.repeat(np.repeat(frames, 2, axis=1), 2, axis=2)
    dest = frames_npz.with_name("frames_x2.npz")
    np.savez(dest, frames=up)
    return {
        "native_shape": list(frames.shape),
        "up_shape": list(up.shape),
        "native_npz_bytes": int(frames_npz.stat().st_size),
        "up_npz_bytes": int(dest.stat().st_size),
        "note": "Nearest-neighbour upsample of native frames, not a stable-envs render.",
    }


def long_or_many(snv, cfg: dict, name: str, spec: dict, num_envs: int, steps: int, label: str) -> dict:
    dest = WORK / "breakpoints" / label
    cell = measure_cell(snv, cfg, name, spec, num_envs, steps, 0, dest)
    stuck = None
    try:
        states = torch.load(dest / "states.pt", weights_only=False)
        pos = states["agent", "position"]
        arr = pos.detach().cpu().numpy()
        if arr.ndim == 3 and arr.shape[0] == 1:
            delta = np.abs(np.diff(arr[0], axis=0)).sum(axis=1)
            stuck = float((delta == 0).mean()) if delta.size else 0.0
    except Exception as exc:
        stuck = f"unavailable: {type(exc).__name__}: {exc}"
    cell["repeated_position_fraction"] = stuck
    cell["label"] = label
    return cell


def variation_breakpoint(snv, cfg: dict) -> dict:
    spec_cfg = cfg["breakpoints_protocol"]["variation_omitted"]
    name = spec_cfg["env"]
    spec = cfg["envs"][name]
    num_envs = int(spec_cfg["num_envs"])
    steps = int(spec_cfg["steps"])
    color = spec_cfg["agent_color"]
    env = open_env(snv, name, num_envs)
    try:
        env.set_variation({"agent.color": color})
        env.reset(seed=0)
        actions = sample_actions(spec, num_envs, steps, 0)
        out = env.step(actions, frames=None)
        states = out["states"].detach().clone()
        variation = env.get_variation().detach().clone()
        reference = env.render(states.reshape(-1)).cpu().contiguous()
    finally:
        env.close()

    fresh = open_env(snv, name, 1)
    try:
        default_frames = fresh.render(states.reshape(-1)).cpu().contiguous()
        repeated = variation.repeat_interleave(steps, dim=0)
        restored = fresh.render(states.reshape(-1), variation=repeated).cpu().contiguous()
    finally:
        fresh.close()

    dest = WORK / "breakpoints" / "variation"
    dest.mkdir(parents=True, exist_ok=True)
    state_path = dest / "states.pt"
    both_path = dest / "states_and_variation.pt"
    torch.save(states, state_path)
    torch.save({"states": states, "variation": variation}, both_path)
    ref_np = reference.numpy()
    return {
        "env": name,
        "agent_color": color,
        "n_frames": int(reference.shape[0]),
        "default_replay": lossless_stats(reference, default_frames),
        "variation_restored": lossless_stats(reference, restored),
        "state_bytes": int(state_path.stat().st_size),
        "state_plus_variation_bytes": int(both_path.stat().st_size),
        "reference_sha256": hashlib.sha256(ref_np.tobytes()).hexdigest(),
    }


def third_family(snv, cfg: dict) -> dict:
    spec_cfg = cfg["breakpoints_protocol"]["third_family"]
    name = spec_cfg["env"]
    spec = {
        "expected_hw": spec_cfg["expected_hw"],
        "discrete": True,
        "action_shape": [1],
        "n_actions": int(spec_cfg["n_actions"]),
    }
    dest = WORK / "breakpoints" / name
    return measure_cell(
        snv,
        cfg,
        name,
        spec,
        int(spec_cfg["num_envs"]),
        int(spec_cfg["steps"]),
        0,
        dest,
    )


def fresh_process_hash(states_path: Path) -> str:
    code = (
        "import hashlib, sys\n"
        "import tensordict\n"
        "import torch\n"
        "import stable_envs as snv\n"
        "states = torch.load(sys.argv[1], weights_only=False)\n"
        "env = snv.TwoRoom(num_envs=1, device='cpu', backend='auto', simulation_backend='auto')\n"
        "try:\n"
        "    frames = env.render(states.reshape(-1)).cpu().numpy()\n"
        "finally:\n"
        "    env.close()\n"
        "print(hashlib.sha256(frames.tobytes()).hexdigest())\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code, str(states_path)],
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr[-2000:])
    return proc.stdout.strip().splitlines()[-1]


def png_level_extreme(cfg: dict, frames_npz: Path, state_bytes: int) -> dict:
    frames = np.load(frames_npz)["frames"]
    level = int(cfg["breakpoints_protocol"]["png_compress_level_extreme"])
    dest = WORK / "breakpoints" / f"png_level_{level}"
    nbytes = save_pngs(frames, dest, level, int(cfg["timing"]["png_write_threads"]))
    return {
        "compress_level": level,
        "png_bytes": int(nbytes),
        "state_bytes": int(state_bytes),
        "disk_saving_vs_png_pct": 100.0 * (1.0 - state_bytes / nbytes),
        "n_frames": int(frames.shape[0]),
    }


def machine_block(snv) -> dict:
    import stable_envs

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "torch": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "numpy": np.__version__,
        "stable_envs": stable_envs.__version__,
        "stable_envs_file": stable_envs.__file__,
    }


def main() -> None:
    cfg = load_config()
    threads = int(cfg["timing"]["torch_num_threads"])
    if threads != 4 or os.environ.get("NUMBA_NUM_THREADS") != "4":
        raise RuntimeError("timing.torch_num_threads is frozen at 4 and must be set before import")
    torch.set_num_threads(threads)

    import stable_envs as snv

    if WORK.exists():
        shutil.rmtree(WORK)
    WORK.mkdir(parents=True)

    cells = []
    t0 = time.perf_counter()
    for name, spec in cfg["envs"].items():
        for seed in cfg["seeds"]:
            print(f"main {name} seed {seed}", flush=True)
            cell = measure_cell(
                snv,
                cfg,
                name,
                spec,
                int(cfg["main"]["num_envs"]),
                int(cfg["main"]["steps"]),
                int(seed),
                WORK / name / f"seed{seed}",
            )
            cells.append(cell)
            print(
                f"  saving png {cell['disk_saving_vs_png_pct']:.2f}%  "
                f"npz {cell['disk_saving_vs_npz_pct']:.2f}%  "
                f"lossless {cell['lossless_fraction']:.3f}  "
                f"replay {cell['state_replay_fps']:.1f} fps  "
                f"load {cell['frame_load_fps']:.1f} fps",
                flush=True,
            )

    print("breakpoints", flush=True)
    non_native = []
    for name, sizes in cfg["breakpoints_protocol"]["non_native_resolution"].items():
        for size in sizes:
            non_native.append(try_non_native(snv, name, cfg["envs"][name], int(size)))
            print(f"  size {name} {size} accepted={non_native[-1]['accepted']}", flush=True)

    tworoom_seed0 = WORK / "TwoRoom" / "seed0"
    upsample = upsample_bytes(tworoom_seed0 / "frames.npz")
    long_ep = long_or_many(snv, cfg, "TwoRoom", cfg["envs"]["TwoRoom"], 1, 512, "long_episode")
    print(f"  long lossless {long_ep['lossless_fraction']}", flush=True)
    many = long_or_many(snv, cfg, "TwoRoom", cfg["envs"]["TwoRoom"], 64, 8, "many_envs")
    print(f"  many lossless {many['lossless_fraction']}", flush=True)
    varied = variation_breakpoint(snv, cfg)
    print(
        f"  variation default match {varied['default_replay']['lossless_fraction']} "
        f"restored {varied['variation_restored']['lossless_fraction']}",
        flush=True,
    )
    acrobot = third_family(snv, cfg)
    print(f"  acrobot lossless {acrobot['lossless_fraction']}", flush=True)
    parent_hash = next(cell["frame_sha256"] for cell in cells if cell["env"] == "TwoRoom" and cell["seed"] == 0)
    child_hash = fresh_process_hash(tworoom_seed0 / "states.pt")
    extreme = png_level_extreme(cfg, tworoom_seed0 / "frames.npz", next(
        cell["state_bytes"] for cell in cells if cell["env"] == "TwoRoom" and cell["seed"] == 0
    ))

    raw = {
        "exp_id": cfg["exp_id"],
        "config_sha256": cfg["_sha256"],
        "hypothesis": cfg["hypothesis"].strip(),
        "machine": machine_block(snv),
        "wall_seconds": time.perf_counter() - t0,
        "cells": cells,
        "breakpoints": {
            "non_native_resolution": non_native,
            "upsample_tworoom_seed0": upsample,
            "long_episode": long_ep,
            "many_envs": many,
            "variation_omitted": varied,
            "third_family": acrobot,
            "fresh_process": {
                "parent_sha256": parent_hash,
                "child_sha256": child_hash,
                "match": parent_hash == child_hash,
            },
            "png_compress_level_extreme": extreme,
        },
    }
    out = EXP / "raw.json"
    out.write_text(json.dumps(raw, indent=2) + "\n")
    print(f"wrote {out} in {raw['wall_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
