"""Lock the scoring rule and, when the subset exists, the checkpoint hash.

Does not roll out the official seeds.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.metrics import auroc, bars_from_runs  # noqa: E402

EXP = ROOT / "experiments" / "exp1_cpu_sudoku_extreme"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check_auroc() -> None:
    perfect = auroc(np.array([0.0, 1.0, 2.0, 3.0]), np.array([0, 0, 1, 1]))
    if perfect != 1.0:
        raise SystemExit(f"perfect AUROC is {perfect}")
    tied = auroc(np.array([1.0, 1.0, 1.0, 1.0]), np.array([0, 1, 0, 1]))
    if tied is None or abs(tied - 0.5) > 1e-9:
        raise SystemExit(f"tied AUROC is {tied}")
    if auroc(np.array([1.0, 2.0]), np.array([1, 1])) is not None:
        raise SystemExit("one-class AUROC should be None")


def _synthetic_run(seed: int, n_correct_at: dict[int, int], n: int = 96) -> dict:
    exact = np.zeros((16, n), dtype=bool)
    energy = np.zeros((16, n), dtype=np.float64)
    gtan = np.ones((16, n), dtype=np.float64)
    # Fill each segment, carrying the previous pattern forward.
    current = 0
    for segment in range(1, 17):
        if segment in n_correct_at:
            current = n_correct_at[segment]
        exact[segment - 1, :current] = True
        energy[segment - 1, :current] = -300.0
        energy[segment - 1, current:] = -100.0
        gtan[segment - 1, :current] = 0.1
        gtan[segment - 1, current:] = 4.0
    return {
        "seed": seed,
        "exact": exact.mean(axis=1).tolist(),
        "exact_mask": exact.tolist(),
        "energy": energy.tolist(),
        "gtan": gtan.tolist(),
    }


def _check_status_rule() -> None:
    passing = [
        _synthetic_run(0, {4: 40, 8: 55, 16: 70}),
        _synthetic_run(1, {4: 42, 8: 56, 16: 72}),
        _synthetic_run(2, {4: 38, 8: 54, 16: 68}),
    ]
    got = bars_from_runs(
        passing, n_boards=96, min_class=5, gain_min=0.02,
        regression_boards_max=1, auroc_min=0.70, gtan_ratio_min=2.0,
    )
    if got["status"] != "PASS":
        raise SystemExit(f"synthetic pass returned {got['status']} {got['bars']}")
    flat_exact = np.zeros((16, 96), dtype=bool)
    flat_exact[:, :48] = True
    flat_energy = np.zeros((16, 96))
    flat_gtan = np.ones((16, 96))
    failing = []
    for seed in (0, 1, 2):
        failing.append({
            "seed": seed,
            "exact": flat_exact.mean(axis=1).tolist(),
            "exact_mask": flat_exact.tolist(),
            "energy": flat_energy.tolist(),
            "gtan": flat_gtan.tolist(),
        })
    got = bars_from_runs(
        failing, n_boards=96, min_class=5, gain_min=0.02,
        regression_boards_max=1, auroc_min=0.70, gtan_ratio_min=2.0,
    )
    if got["status"] != "FAIL":
        raise SystemExit(f"synthetic fail returned {got['status']} {got['bars']}")


def _check_files() -> None:
    with (EXP / "config.yaml").open() as handle:
        cfg = yaml.safe_load(handle)
    need = ("gain_16_minus_4_min", "regression_boards_max", "energy_auroc_min", "gtan_ratio_min")
    missing = [key for key in need if key not in cfg["thresholds"]]
    if missing:
        raise SystemExit(f"config missing {missing}")
    manifest_path = ROOT / "work" / "manifest.json"
    ckpt = ROOT / "work" / "checkpoints" / "sudoku.pth"
    if manifest_path.is_file() and ckpt.is_file():
        manifest = json.loads(manifest_path.read_text())
        digest = _sha256(ckpt)
        if digest != cfg["checkpoint_sha256"]:
            raise SystemExit("checkpoint hash does not match config")
        if len(manifest["hard_indices"]) != int(cfg["n_boards"]):
            raise SystemExit("subset size does not match config")
        if manifest["hard_ratings"][0] < manifest["hard_ratings"][-1]:
            raise SystemExit("hard subset is not in descending rating order")


def main() -> None:
    _check_auroc()
    _check_status_rule()
    _check_files()
    print("verify ok")


if __name__ == "__main__":
    main()
