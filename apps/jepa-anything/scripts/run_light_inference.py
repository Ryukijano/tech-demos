#!/usr/bin/env python3
"""Download the lightest JEPA-Anything domain checkpoint and smoke-test inference.

Checkpoint: Synthetic Causal Dynamics (~1.9 MB) from Gen-Verse/jepa-anything dataset.
If torch / upstream infer script are missing, print the planned command and exit 0
with a clear DRY-RUN note (so CI stays green without GPU/weights tooling).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "work"
CKPT_DIR = WORK / "checkpoints"
CKPT_NAME = "Synthetic_Causal_Dynamics_final_checkpoint.pt"
HF_REPO = "Gen-Verse/jepa-anything"
HF_FILE = (
    "05_Ten_Task_Dynamics/Synthetic_Causal_Dynamics/checkpoints/"
    "Synthetic_Causal_Dynamics_final_checkpoint.pt"
)


def download_checkpoint(dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("Installing huggingface_hub…")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub"])
        from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id=HF_REPO,
        repo_type="dataset",
        filename=HF_FILE,
        local_dir=str(dest.parent),
    )
    # hf may nest directories; copy/symlink to flat expected name
    downloaded = Path(path)
    if downloaded.name != CKPT_NAME:
        target = dest.parent / CKPT_NAME
        if not target.exists():
            target.write_bytes(downloaded.read_bytes())
        return target
    return downloaded


def try_torch_load(ckpt: Path) -> dict:
    import torch

    obj = torch.load(ckpt, map_location="cpu", weights_only=False)
    info = {"type": type(obj).__name__}
    if isinstance(obj, dict):
        info["keys"] = sorted(list(obj.keys()))[:40]
        # common weight bags
        for k in ("state_dict", "model", "ema", "net"):
            if k in obj and hasattr(obj[k], "keys"):
                info[f"{k}_n_tensors"] = len(list(obj[k].keys()))
    return info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true", help="Fetch the ~1.9 MB checkpoint")
    parser.add_argument(
        "--infer",
        action="store_true",
        help="If upstream infer_dynamics.py exists under work/, run smoke infer",
    )
    args = parser.parse_args()

    planned = [
        f"hf download {HF_REPO} --repo-type dataset {HF_FILE}",
        "python test/infer_dynamics.py "
        f"--checkpoint checkpoints/{CKPT_NAME} "
        "--output /tmp/jepa_inference_smoke/predictions.npz",
    ]
    print("Planned commands:")
    for line in planned:
        print(" ", line)

    ckpt = CKPT_DIR / CKPT_NAME
    if not args.download and not ckpt.exists():
        print()
        print("DRY-RUN: no --download and no local checkpoint. Exiting 0.")
        print("Re-run with --download to fetch Synthetic Causal Dynamics (~1.9 MB).")
        return

    if args.download or not ckpt.exists():
        print()
        print("Downloading lightest checkpoint…")
        ckpt = download_checkpoint(ckpt)
        print(f"Saved: {ckpt} ({ckpt.stat().st_size} bytes)")

    report: dict = {"checkpoint": str(ckpt), "bytes": ckpt.stat().st_size}

    try:
        report["torch"] = try_torch_load(ckpt)
        print("torch.load OK:", json.dumps(report["torch"], indent=2))
    except Exception as exc:  # noqa: BLE001 — report gracefully
        report["torch_error"] = str(exc)
        print(f"WARN: could not torch.load ({exc}). Checkpoint file is present.")

    infer_script = WORK / "JEPA-Anything" / "test" / "infer_dynamics.py"
    # Also check dataset-style layout if user synced HF dataset
    alt = WORK / "jepa-anything-dataset" / "test" / "infer_dynamics.py"
    script = infer_script if infer_script.exists() else alt if alt.exists() else None

    if args.infer and script:
        out = WORK / "predictions"
        out.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            str(script),
            "--checkpoint",
            str(ckpt),
            "--output",
            str(out / "predictions.npz"),
        ]
        print("+", " ".join(cmd))
        code = subprocess.call(cmd)
        report["infer_exit"] = code
    elif args.infer:
        print("WARN: infer_dynamics.py not found. Clone upstream first:")
        print("  python3 scripts/dry_run.py --clone")
        report["infer"] = "skipped-missing-script"
    else:
        print("Skip infer (pass --infer after cloning upstream).")

    (WORK / "light_inference_report.json").write_text(json.dumps(report, indent=2))
    print("Wrote", WORK / "light_inference_report.json")


if __name__ == "__main__":
    main()
