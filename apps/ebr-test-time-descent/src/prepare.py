"""Clone the pinned EBR repo and build the Sudoku-Extreme rating subsets.

The full test CSV is downloaded from the revision named in config.yaml.
Arrays are cached under work/ and are not committed.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parent.parent
EXP = ROOT / "experiments" / "exp1_cpu_sudoku_extreme"
WORK = ROOT / "work"
EBR = WORK / "ebr"
if not (EBR / "models").is_dir():
    raise SystemExit("work/ebr is missing. Run ./run_all.sh so the pinned checkout exists before prepare.py.")
sys.path.insert(0, str(EBR))

from dataset.build_sudoku_dataset import DataProcessConfig, convert_subset  # noqa: E402
from dataset.common import write_hrm_split  # noqa: E402


def load_cfg() -> dict:
    with (EXP / "config.yaml").open() as handle:
        return yaml.safe_load(handle)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_repo(cfg: dict) -> None:
    commit = cfg["upstream_commit"]
    if (EBR / "models" / "energy.py").is_file():
        head = subprocess.check_output(["git", "-C", str(EBR), "rev-parse", "HEAD"], text=True).strip()
        if head != commit:
            raise SystemExit(f"work/ebr is {head}, expected {commit}")
        return
    WORK.mkdir(parents=True, exist_ok=True)
    if EBR.exists():
        raise SystemExit(f"{EBR} exists but is not a complete checkout")
    subprocess.check_call(
        ["git", "clone", "--filter=blob:none", "--no-checkout", "https://github.com/wwillchen/ebr.git", str(EBR)]
    )
    subprocess.check_call(["git", "-C", str(EBR), "fetch", "--depth", "1", "origin", commit])
    subprocess.check_call(["git", "-C", str(EBR), "checkout", "--detach", "FETCH_HEAD"])
    head = subprocess.check_output(["git", "-C", str(EBR), "rev-parse", "HEAD"], text=True).strip()
    if head != commit:
        raise SystemExit(f"checked out {head}, expected {commit}")


def ensure_checkpoint(cfg: dict) -> Path:
    dest = WORK / "checkpoints" / "sudoku.pth"
    if dest.is_file() and sha256_file(dest) == cfg["checkpoint_sha256"]:
        return dest
    path = hf_hub_download(
        cfg["checkpoint_repo"],
        cfg["checkpoint_file"],
        revision=cfg["checkpoint_revision"],
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    # Keep a stable path. hf_hub_download already cached the bytes; copy if needed.
    if Path(path).resolve() != dest.resolve():
        dest.write_bytes(Path(path).read_bytes())
    digest = sha256_file(dest)
    if digest != cfg["checkpoint_sha256"]:
        raise SystemExit(f"checkpoint sha256 {digest} != {cfg['checkpoint_sha256']}")
    return dest


def ensure_full_test(cfg: dict) -> Path:
    full = WORK / "sudoku-extreme-full"
    meta = full / "test" / "dataset.json"
    if meta.is_file():
        info = json.loads(meta.read_text())
        if int(info["total_samples"]) == 422786:
            return full
    os.environ.setdefault("HF_HOME", str(WORK / "hf"))
    full.mkdir(parents=True, exist_ok=True)
    convert_subset("test", DataProcessConfig(output_dir=str(full), num_aug=0))
    (full / "identifiers.json").write_text(json.dumps(["<blank>"]))
    return full


def read_ratings(cfg: dict) -> np.ndarray:
    os.environ.setdefault("HF_HOME", str(WORK / "hf"))
    csv_path = hf_hub_download(
        cfg["data_repo"],
        "test.csv",
        repo_type="dataset",
        revision=cfg["data_revision"],
    )
    ratings = []
    with open(csv_path, newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        if header[3] != "rating":
            raise SystemExit(f"unexpected test.csv header {header}")
        for row in reader:
            ratings.append(int(row[3]))
    return np.asarray(ratings, dtype=np.int32)


def write_subset(full: Path, indices: np.ndarray, dest: Path) -> None:
    inputs = np.load(full / "test" / "all__inputs.npy", mmap_mode="r")
    labels = np.load(full / "test" / "all__labels.npy", mmap_mode="r")
    n = int(indices.size)
    arrays = {
        "inputs": np.asarray(inputs[indices]),
        "labels": np.asarray(labels[indices]),
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
    (dest / "identifiers.json").write_text(json.dumps(["<blank>"]))


def main() -> None:
    cfg = load_cfg()
    ensure_repo(cfg)
    ckpt = ensure_checkpoint(cfg)
    full = ensure_full_test(cfg)
    ratings = read_ratings(cfg)
    if ratings.shape[0] != 422786:
        raise SystemExit(f"expected 422786 test rows, found {ratings.shape[0]}")
    order = np.argsort(-ratings, kind="mergesort")
    n_boards = int(cfg["n_boards"])
    hard = order[:n_boards]
    band_n = int(cfg["breakpoints"]["breakpoint_boards"])
    easiest_pool = np.flatnonzero(ratings == 0)
    if easiest_pool.size < band_n:
        raise SystemExit(f"only {easiest_pool.size} rating-0 puzzles")
    easiest = easiest_pool[:band_n]
    median = int(np.median(ratings))
    median_pool = np.flatnonzero(ratings == median)
    if median_pool.size < band_n:
        raise SystemExit(f"only {median_pool.size} puzzles at median rating {median}")
    median_idx = median_pool[:band_n]
    write_subset(full, hard, WORK / "subset-hard")
    write_subset(full, easiest, WORK / "subset-easy")
    write_subset(full, median_idx, WORK / "subset-median")
    manifest = {
        "checkpoint": str(ckpt),
        "checkpoint_sha256": sha256_file(ckpt),
        "n_test": int(ratings.shape[0]),
        "hard_indices": hard.astype(int).tolist(),
        "hard_ratings": ratings[hard].astype(int).tolist(),
        "easy_indices": easiest.astype(int).tolist(),
        "easy_ratings": ratings[easiest].astype(int).tolist(),
        "median_rating": median,
        "median_indices": median_idx.astype(int).tolist(),
        "median_ratings": ratings[median_idx].astype(int).tolist(),
        "rating_min": int(ratings.min()),
        "rating_max": int(ratings.max()),
    }
    (WORK / "manifest.json").write_text(json.dumps(manifest))
    print(
        f"hard n={n_boards} ratings {int(ratings[hard[-1]])}..{int(ratings[hard[0]])} "
        f"median={median} checkpoint ok"
    )


if __name__ == "__main__":
    main()
