"""Download the HPatches reference frames used as the image pool."""

from __future__ import annotations

import zipfile
from pathlib import Path

from huggingface_hub import hf_hub_download

HF_REPO = "honggyuAn/hpatches-sequences-release"
HF_FILE = "hpatches-sequences-release.zip"


def ensure_reference_pool(cache_dir: Path, pool_size: int) -> list[Path]:
    """Return ``pool_size`` reference images, archive order, extracted once."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    archive = Path(
        hf_hub_download(repo_id=HF_REPO, filename=HF_FILE, repo_type="dataset")
    )
    with zipfile.ZipFile(archive) as bundle:
        names = sorted(name for name in bundle.namelist() if name.endswith("/1.ppm"))
        if len(names) < pool_size:
            raise RuntimeError(f"archive has {len(names)} reference frames, need {pool_size}")
        chosen = names[:pool_size]
        paths: list[Path] = []
        for name in chosen:
            sequence = name.split("/")[-2]
            target = cache_dir / f"{sequence}.ppm"
            if not target.exists() or target.stat().st_size == 0:
                target.write_bytes(bundle.read(name))
            paths.append(target)
    return paths
