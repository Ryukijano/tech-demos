"""CPU rollout of the released Sudoku-Extreme EBR checkpoint.

Seeds reseed the latent-init and Langevin noise. The checkpoint itself is one
training run. Segment t is the state after t blocks of 21 energy microsteps,
which is the depth axis of the official evaluator.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
EXP = ROOT / "experiments" / "exp1_cpu_sudoku_extreme"
WORK = ROOT / "work"
EBR = WORK / "ebr"
if not (EBR / "models").is_dir():
    raise SystemExit("work/ebr is missing. Run ./run_all.sh first.")
sys.path.insert(0, str(EBR))

from dataset.common import read_metadata, write_hrm_split  # noqa: E402
from dataset.puzzle_dataset import create_dataloader  # noqa: E402
from evaluators.common import eval_config, load_checkpoint, load_model, measure  # noqa: E402
from models.losses import IGNORE_LABEL_ID  # noqa: E402
from utils.runtime import seed_everything, tree_to_device  # noqa: E402


def load_cfg() -> dict:
    with (EXP / "config.yaml").open() as handle:
        return yaml.safe_load(handle)


def set_threads(n: int) -> None:
    torch.set_num_threads(max(1, int(n)))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def load_released(cfg: dict, manifest: dict):
    device = torch.device("cpu")
    ckpt = load_checkpoint(manifest["checkpoint"], map_location=device)
    config = eval_config(ckpt, device=device)
    halt = int(config.arch.halt_max_steps)
    if halt != int(cfg["train_segments"]):
        raise SystemExit(f"checkpoint halt_max_steps is {halt}, config expects {cfg['train_segments']}")
    config.global_batch_size = int(cfg["batch_size"])
    return ckpt, config, device


def build_model(ckpt, config, data_path: str, device: torch.device):
    config.dataset.data_path = data_path
    # Build consumes RNG. The caller reseeds again before the rollout.
    model, _applied = load_model(ckpt, config, _metadata(config), device=device, build_device=device)
    model.eval()
    return model


def _metadata(config):
    return read_metadata(config.dataset.data_path, "test")


def apply_overrides(model, *, init_std: float | None, noise_scale: float | None) -> None:
    arch = model.model.config
    if init_std is not None:
        arch.init_std = float(init_std)
    if noise_scale is not None:
        arch.noise_cosine = False
        arch.noise_scale = float(noise_scale)
        arch.noise_settle_hot = float(noise_scale)


def rollout(model, config, *, seed: int, depth: int, device: torch.device) -> dict:
    """One trajectory per board. Returns per-segment exact masks and scores."""

    seed_everything(seed)
    loader, _metadata_obj = create_dataloader(
        config,
        "test",
        test_set_mode=True,
        epochs_per_iter=1,
        global_batch_size=int(config.global_batch_size),
    )
    train_depth = int(config.arch.halt_max_steps)
    exact_rows: list[torch.Tensor] = []
    cell_rows: list[torch.Tensor] = []
    energy_rows: list[torch.Tensor] = []
    gtan_rows: list[torch.Tensor] = []
    gnorm_rows: list[torch.Tensor] = []
    pred_rows: list[torch.Tensor] = []
    n_seen = 0
    inner = model.model.inner
    with torch.inference_mode():
        for _name, data, _rows in loader:
            data = tree_to_device(data, device)
            # The loader pads the last batch with ignore labels up to the batch size.
            keep = data["labels"].ne(IGNORE_LABEL_ID).any(-1)
            if int(keep.sum()) == 0:
                continue
            if int(keep.sum()) != int(keep.numel()):
                data = {key: value[keep] for key, value in data.items()}
            labels = data["labels"].to(torch.long)
            cells = labels.ne(IGNORE_LABEL_ID)
            x = inner.input_embeddings(data["inputs"])
            cos_sin = inner.cos_sin()
            carry = model.initial_carry(data)
            per_exact = []
            per_cell = []
            per_energy = []
            per_gtan = []
            per_gnorm = []
            per_pred = []
            for segment in range(depth):
                if segment >= train_depth:
                    carry.halted = torch.zeros_like(carry.halted)
                    carry.steps = torch.full_like(carry.steps, train_depth - 1)
                if segment and bool(carry.halted.any()):
                    raise RuntimeError("a row halted before the requested depth")
                carry, _loss, _metrics, outputs, _all = model(
                    carry=carry, batch=data, return_keys=["preds", "convergence_logit"]
                )
                preds = outputs["preds"].to(torch.long)
                energy, gnorm, gtan = measure(inner.level, carry.inner_carry.z, x, cos_sin)
                match = preds.eq(labels) | ~cells
                per_exact.append(match.all(-1).cpu())
                per_cell.append(match.float().mean(-1).cpu())
                per_energy.append(energy.cpu())
                per_gtan.append(gtan.cpu())
                per_gnorm.append(gnorm.cpu())
                per_pred.append(preds.cpu())
            exact_rows.append(torch.stack(per_exact, dim=0))
            cell_rows.append(torch.stack(per_cell, dim=0))
            energy_rows.append(torch.stack(per_energy, dim=0))
            gtan_rows.append(torch.stack(per_gtan, dim=0))
            gnorm_rows.append(torch.stack(per_gnorm, dim=0))
            pred_rows.append(torch.stack(per_pred, dim=0))
            n_seen += int(labels.shape[0])
    if n_seen == 0:
        raise RuntimeError("loader produced no boards")
    return {
        "exact_mask": torch.cat(exact_rows, dim=1).numpy(),
        "cell": torch.cat(cell_rows, dim=1).numpy(),
        "energy": torch.cat(energy_rows, dim=1).numpy(),
        "gtan": torch.cat(gtan_rows, dim=1).numpy(),
        "gnorm": torch.cat(gnorm_rows, dim=1).numpy(),
        "preds": torch.cat(pred_rows, dim=1).numpy(),
    }


def pack_run(seed: int, rolled: dict, *, keep_preds: bool) -> dict:
    exact = rolled["exact_mask"].astype(bool)
    record = {
        "seed": seed,
        "depth": int(exact.shape[0]),
        "n": int(exact.shape[1]),
        "exact": exact.mean(axis=1).astype(float).tolist(),
        "n_correct": exact.sum(axis=1).astype(int).tolist(),
        "cell": rolled["cell"].mean(axis=1).astype(float).tolist(),
        "exact_mask": exact.tolist(),
        "energy": rolled["energy"].astype(float).tolist(),
        "gtan": rolled["gtan"].astype(float).tolist(),
        "gnorm": rolled["gnorm"].astype(float).tolist(),
    }
    if keep_preds:
        record["preds"] = rolled["preds"].astype(int).tolist()
    return record


def choose_example(exact_mask: np.ndarray) -> int:
    """First board wrong at the training halt and right eight segments later."""

    wrong4 = ~exact_mask[3]
    right8 = exact_mask[7]
    hit = np.flatnonzero(wrong4 & right8)
    if hit.size:
        return int(hit[0])
    stuck = np.flatnonzero(wrong4)
    if stuck.size:
        return int(stuck[0])
    return 0


def run_main(cfg: dict, manifest: dict) -> None:
    set_threads(int(cfg["threads"]))
    _ckpt, config, device = load_released(cfg, manifest)
    # Rebuild per seed so init-time RNG draws cannot leak across seeds.
    runs = []
    example = None
    t_all = time.perf_counter()
    for seed in cfg["seeds"]:
        depth = int(cfg["extreme_segments"]) if int(seed) == 0 else int(cfg["eval_segments"])
        model = build_model(_ckpt, config, str(WORK / "subset-hard"), device)
        t0 = time.perf_counter()
        rolled = rollout(model, config, seed=int(seed), depth=depth, device=device)
        seconds = time.perf_counter() - t0
        keep = int(seed) == 0
        record = pack_run(int(seed), rolled, keep_preds=keep)
        record["seconds"] = seconds
        if keep:
            idx = choose_example(rolled["exact_mask"])
            inputs = np.load(WORK / "subset-hard" / "test" / "all__inputs.npy")
            labels = np.load(WORK / "subset-hard" / "test" / "all__labels.npy")
            example = {
                "board_index": idx,
                "test_index": int(manifest["hard_indices"][idx]),
                "rating": int(manifest["hard_ratings"][idx]),
                "clues": inputs[idx].astype(int).tolist(),
                "labels": labels[idx].astype(int).tolist(),
                "preds": rolled["preds"][:, idx, :].astype(int).tolist(),
                "exact": rolled["exact_mask"][:, idx].astype(bool).tolist(),
                "energy": rolled["energy"][:, idx].astype(float).tolist(),
                "gtan": rolled["gtan"][:, idx].astype(float).tolist(),
            }
            record.pop("preds", None)
        else:
            record.pop("preds", None)
        # Drop per-board masks from the JSON for segments we still need.
        # exact_mask, energy and gtan stay: the bars read them.
        runs.append(record)
        acc16 = record["exact"][int(cfg["eval_segments"]) - 1]
        print(f"seed {seed} depth {depth} acc@16 {acc16:.3f} in {seconds:.1f}s", flush=True)
        del model
    payload = {
        "checkpoint_sha256": manifest["checkpoint_sha256"],
        "hard_ratings": manifest["hard_ratings"],
        "hard_indices": manifest["hard_indices"],
        "seconds_total": time.perf_counter() - t_all,
        "runs": runs,
        "example": example,
    }
    (EXP / "raw.json").write_text(json.dumps(payload))
    print(f"wrote {EXP / 'raw.json'}")


def run_breakpoints(cfg: dict, manifest: dict) -> None:
    set_threads(int(cfg["threads"]))
    _ckpt, config, device = load_released(cfg, manifest)
    bp = cfg["breakpoints"]
    n = int(bp["breakpoint_boards"])
    depth = int(bp["breakpoint_segments"])
    band_depth = int(bp["band_segments"])
    # A width-n prefix dataset, same order as the hard subset.
    prefix = WORK / "subset-hard-prefix"
    _write_prefix(n, prefix)
    studies = []
    for init_std in bp["init_std"]:
        model = build_model(_ckpt, config, str(prefix), device)
        apply_overrides(model, init_std=float(init_std), noise_scale=None)
        rolled = rollout(model, config, seed=0, depth=depth, device=device)
        studies.append({
            "kind": "init_std",
            "value": float(init_std),
            "exact": rolled["exact_mask"].mean(axis=1).astype(float).tolist(),
            "n": int(rolled["exact_mask"].shape[1]),
        })
        print(f"init_std {init_std} acc {studies[-1]['exact']}", flush=True)
        del model
    for sigma in bp["noise_scale"]:
        model = build_model(_ckpt, config, str(prefix), device)
        apply_overrides(model, init_std=None, noise_scale=float(sigma))
        rolled = rollout(model, config, seed=0, depth=depth, device=device)
        studies.append({
            "kind": "noise_scale",
            "value": float(sigma),
            "exact": rolled["exact_mask"].mean(axis=1).astype(float).tolist(),
            "n": int(rolled["exact_mask"].shape[1]),
        })
        print(f"noise_scale {sigma} acc {studies[-1]['exact']}", flush=True)
        del model
    for name, folder in (
        ("easiest", WORK / "subset-easy"),
        ("median", WORK / "subset-median"),
        ("hardest", prefix),
    ):
        model = build_model(_ckpt, config, str(folder), device)
        rolled = rollout(model, config, seed=0, depth=band_depth, device=device)
        studies.append({
            "kind": "band",
            "name": name,
            "exact": rolled["exact_mask"].mean(axis=1).astype(float).tolist(),
            "n": int(rolled["exact_mask"].shape[1]),
        })
        print(f"band {name} acc@1 {studies[-1]['exact'][0]:.3f} acc@end {studies[-1]['exact'][-1]:.3f}", flush=True)
        del model
    (EXP / "breakpoints_raw.json").write_text(json.dumps({
        "studies": studies,
        "median_rating": manifest["median_rating"],
        "hard_rating_min": min(manifest["hard_ratings"][:n]),
        "hard_rating_max": max(manifest["hard_ratings"][:n]),
        "easy_rating": 0,
    }))
    print(f"wrote {EXP / 'breakpoints_raw.json'}")


def _write_prefix(n: int, dest: Path) -> None:
    src = WORK / "subset-hard" / "test"
    inputs = np.load(src / "all__inputs.npy")[:n]
    labels = np.load(src / "all__labels.npy")[:n]
    arrays = {
        "inputs": inputs,
        "labels": labels,
        "group_indices": np.arange(n + 1, dtype=np.int32),
        "puzzle_indices": np.arange(n + 1, dtype=np.int32),
        "puzzle_identifiers": np.zeros(n, dtype=np.int32),
    }
    metadata = {
        "pad_id": 0,
        "ignore_label_id": 0,
        "blank_identifier_id": 0,
        "vocab_size": 11,
        "seq_len": 81,
        "num_puzzle_identifiers": 1,
        "total_groups": n,
        "mean_puzzle_examples": 1.0,
        "sets": ["all"],
        "total_samples": n,
    }
    write_hrm_split(str(dest), "test", arrays, metadata)


def run_smoke(cfg: dict, manifest: dict) -> None:
    """Two boards, two segments, a seed outside the official set. Not a result."""

    set_threads(int(cfg["threads"]))
    _ckpt, config, device = load_released(cfg, manifest)
    dest = WORK / "subset-smoke"
    _write_prefix(2, dest)
    model = build_model(_ckpt, config, str(dest), device)
    rolled = rollout(model, config, seed=99, depth=2, device=device)
    exact = rolled["exact_mask"]
    energy = rolled["energy"]
    if exact.shape != (2, 2):
        raise SystemExit(f"smoke shape {exact.shape}")
    if not np.isfinite(energy).all():
        raise SystemExit("smoke energy is not finite")
    payload = {
        "seed": 99,
        "depth": 2,
        "n": 2,
        "exact": exact.mean(axis=1).astype(float).tolist(),
        "energy_finite": True,
        "official": False,
    }
    (EXP / "smoke.json").write_text(json.dumps(payload, indent=2))
    print(f"smoke ok exact {payload['exact']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--breakpoints", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    cfg = load_cfg()
    manifest = json.loads((WORK / "manifest.json").read_text())
    if args.smoke and args.breakpoints:
        raise SystemExit("choose one of --smoke or --breakpoints")
    if args.smoke:
        run_smoke(cfg, manifest)
    elif args.breakpoints:
        run_breakpoints(cfg, manifest)
    else:
        run_main(cfg, manifest)


if __name__ == "__main__":
    main()
