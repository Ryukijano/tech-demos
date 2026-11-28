"""Replace result files in one step so a reader never sees a partial write."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


def _temporary(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="wb",
        prefix=f".{path.name}.",
        dir=path.parent,
        delete=False,
    )
    name = Path(handle.name)
    handle.close()
    return name


def write_text(path: Path, text: str) -> None:
    tmp = _temporary(path)
    try:
        # newline="\n" keeps CSV records from gaining a second CR on Windows.
        tmp.write_text(text, encoding="utf-8", newline="\n")
        os.replace(tmp, path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def write_bytes(path: Path, data: bytes) -> None:
    tmp = _temporary(path)
    try:
        tmp.write_bytes(data)
        os.replace(tmp, path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def replace_file(tmp: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    os.replace(tmp, dest)


def publish(staging: Path, dest: Path, names: list[str]) -> None:
    """Move a finished set of files into dest. Nothing is moved if a name is missing."""
    missing = [name for name in names if not (staging / name).is_file()]
    if missing:
        raise FileNotFoundError(f"staged outputs missing: {', '.join(missing)}")
    for name in names:
        os.replace(staging / name, dest / name)
