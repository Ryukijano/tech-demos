#!/usr/bin/env python3
"""PixelUMM smoke test — fails gracefully without GPU / weights.

Exit codes:
  0 — environment reported honestly (missing pieces are OK)
  1 — unexpected failure inside this script
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "artifacts" / "smoke_report.json"


def main() -> int:
    report: dict = {
        "app": "pixelumm",
        "task": "image-vlm",
        "guardrails_default": "enabled for T2V; image-vlm path does not disable them",
        "no_guardrails": "opt-in only via upstream --no-guardrails (do not use by default)",
        "checks": {},
        "status": "incomplete",
        "next_steps": [],
    }

    try:
        import torch  # type: ignore

        report["checks"]["torch"] = {
            "ok": True,
            "version": getattr(torch, "__version__", "?"),
            "cuda_available": bool(torch.cuda.is_available()),
            "device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        }
        if torch.cuda.is_available():
            report["checks"]["torch"]["device0"] = torch.cuda.get_device_name(0)
    except Exception as exc:  # noqa: BLE001
        report["checks"]["torch"] = {"ok": False, "error": str(exc)}
        report["next_steps"].append("Install CUDA PyTorch per upstream ENVIRONMENT.md")

    ckpt = os.environ.get("PIXELUMM_CKPT")
    qwen = os.environ.get("PIXELUMM_QWEN_DIR")
    report["checks"]["PIXELUMM_CKPT"] = {
        "set": bool(ckpt),
        "exists": bool(ckpt and Path(ckpt).exists()),
        "path": ckpt,
    }
    report["checks"]["PIXELUMM_QWEN_DIR"] = {
        "set": bool(qwen),
        "exists": bool(qwen and Path(qwen).exists()),
        "path": qwen,
    }

    upstream = ROOT / "work" / "PixelUMM"
    report["checks"]["upstream_clone"] = {
        "exists": upstream.exists(),
        "path": str(upstream),
        "inference_py": (upstream / "inference.py").exists() if upstream.exists() else False,
    }

    has_cuda = report["checks"].get("torch", {}).get("cuda_available")
    has_ckpt = report["checks"]["PIXELUMM_CKPT"]["exists"]
    has_qwen = report["checks"]["PIXELUMM_QWEN_DIR"]["exists"]
    has_infer = report["checks"]["upstream_clone"]["inference_py"]

    if has_cuda and has_ckpt and has_qwen and has_infer:
        report["status"] = "ready"
        report["message"] = "Environment looks ready for image-vlm. Run scripts/run_image_vlm.sh"
    else:
        report["status"] = "graceful-skip"
        report["message"] = (
            "PixelUMM image understanding is NOT runnable here yet "
            "(missing GPU and/or weights and/or upstream clone). "
            "This is expected in CPU-only CI. See README.md."
        )
        if not has_cuda:
            report["next_steps"].append("Allocate an NVIDIA GPU with CUDA 13 + FlashAttention build")
        if not has_ckpt or not has_qwen:
            report["next_steps"].append(
                "Download S8-F22-R05 + Qwen config/tokenizer per upstream CHECKPOINT.md; "
                "export PIXELUMM_CKPT and PIXELUMM_QWEN_DIR"
            )
        if not has_infer:
            report["next_steps"].append("bash scripts/setup_env.sh  # clones nv-tlabs/PixelUMM")

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print()
    print(report["message"])
    # Graceful: always 0 when we successfully diagnosed missing deps
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001
        print(f"smoke_test internal error: {exc}", file=sys.stderr)
        raise SystemExit(1)
