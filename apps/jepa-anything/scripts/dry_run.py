#!/usr/bin/env python3
"""Dry-run for JEPA-Anything: print planned commands; optionally clone/install.

This path is intentionally honest: it does not download multi-GB domain
weights unless you opt in via run_light_inference.py --download.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "work"
UPSTREAM = "https://github.com/Gen-Verse/JEPA-Anything.git"
LIGHT_CKPT_REPO = "Gen-Verse/jepa-anything"
LIGHT_CKPT_FILE = (
    "05_Ten_Task_Dynamics/Synthetic_Causal_Dynamics/checkpoints/"
    "Synthetic_Causal_Dynamics_final_checkpoint.pt"
)


def run(cmd: list[str], *, cwd: Path | None = None, check: bool = True) -> int:
    print("+", " ".join(cmd))
    if cwd:
        print(f"  (cwd={cwd})")
    proc = subprocess.run(cmd, cwd=cwd)
    if check and proc.returncode != 0:
        raise SystemExit(proc.returncode)
    return proc.returncode


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--clone",
        action="store_true",
        help="Actually clone JEPA-Anything and install the core package",
    )
    parser.add_argument(
        "--run-recipe",
        action="store_true",
        help="After --clone, run the synthetic-linear-dynamics structural recipe",
    )
    args = parser.parse_args()

    print("=== JEPA-Anything dry-run ===")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Work dir: {WORK}")
    print()
    print("Planned install:")
    print(f"  git clone {UPSTREAM}")
    print("  python3 -m pip install -e './jepa-anything-core[dev]'")
    print("  python3 recipes/synthetic-linear-dynamics/run_recipe.py --quiet")
    print()
    print("Lightest domain checkpoint (~1.9 MB):")
    print(f"  HF dataset: {LIGHT_CKPT_REPO}")
    print(f"  file: {LIGHT_CKPT_FILE}")
    print("  infer: python test/infer_dynamics.py --checkpoint <pt> --output <out>/predictions.npz")
    print()

    if not args.clone:
        print("OK dry-run only (pass --clone to fetch upstream).")
        print("For checkpoint smoke: python3 scripts/run_light_inference.py --download")
        return

    WORK.mkdir(parents=True, exist_ok=True)
    repo = WORK / "JEPA-Anything"
    if not repo.exists():
        run(["git", "clone", "--depth", "1", UPSTREAM, str(repo)])
    else:
        print(f"Reuse existing clone at {repo}")

    # Prefer editable core install when present
    core = repo / "jepa-anything-core"
    if core.exists():
        run([sys.executable, "-m", "pip", "install", "-e", f"{core}[dev]"], check=False)
    else:
        print("WARN: jepa-anything-core/ not found — layout may have changed; see upstream README.")

    recipe = repo / "recipes" / "synthetic-linear-dynamics" / "run_recipe.py"
    if args.run_recipe and recipe.exists():
        run([sys.executable, str(recipe), "--quiet"], cwd=repo, check=False)
    elif args.run_recipe:
        print(f"WARN: recipe missing at {recipe}")

    print()
    print("Clone path complete. Next: python3 scripts/run_light_inference.py --download")


if __name__ == "__main__":
    if shutil.which("git") is None and "--clone" in sys.argv:
        print("git is required for --clone", file=sys.stderr)
        raise SystemExit(1)
    main()
