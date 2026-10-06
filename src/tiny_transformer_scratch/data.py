"""Download the TinyStories text files into ./data (not committed to git).

The files come from one fixed commit of the dataset and must match the SHA-256 and size that
Hugging Face publishes for them, so a changed or tampered file is refused instead of used.

Usage:
    python -m tiny_transformer_scratch.data            # validation split, ~19 MB
    python -m tiny_transformer_scratch.data --split train   # ~1.9 GB
"""

from __future__ import annotations

import argparse
import hashlib
import urllib.request
from pathlib import Path
from typing import NamedTuple

REVISION = "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64"  # dataset commit of 2024-08-12
BASE_URL = f"https://huggingface.co/datasets/roneneldan/TinyStories/resolve/{REVISION}/"


class DataFile(NamedTuple):
    """One file of the dataset and what it must contain."""

    name: str
    sha256: str
    size: int


FILES = {
    "valid": DataFile(
        "TinyStories-valid.txt",
        "94e431816c4cce81ff71e4408ff8d3bda9a42e8d2663986697c3954288cb38b4",
        19_447_282,
    ),
    "train": DataFile(
        "TinyStories-train.txt",
        "c5cf5e22ff13614e830afbe61a99fbcbe8bcb7dd72252b989fa1117a368d401f",
        1_924_281_556,
    ),
}


class IntegrityError(ValueError):
    """The bytes do not match the published size or SHA-256."""


def _sha256(path: Path) -> str:
    """SHA-256 of a file, read in 1 MiB chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _fetch(url: str, partial: Path, want: DataFile) -> None:
    """Stream url into partial while hashing; refuse extra bytes or a wrong hash."""
    digest, size = hashlib.sha256(), 0
    with urllib.request.urlopen(url, timeout=60) as resp, open(partial, "wb") as out:  # noqa: S310 (fixed https URL)
        while chunk := resp.read(1 << 20):
            size += len(chunk)
            if size > want.size:
                raise IntegrityError(f"{want.name}: more than the expected {want.size} bytes")
            digest.update(chunk)
            out.write(chunk)
    if size != want.size or digest.hexdigest() != want.sha256:
        raise IntegrityError(f"{want.name}: got {size} bytes with SHA-256 {digest.hexdigest()}")


def download(split: str = "valid", dest: Path = Path("data")) -> Path:
    """Fetch and verify one split unless a verified copy is present; return its path."""
    want = FILES[split]
    target = dest / want.name
    if target.exists():
        if target.stat().st_size != want.size or _sha256(target) != want.sha256:
            raise IntegrityError(
                f"{target} does not match the published SHA-256; delete it and download again"
            )
        return target
    dest.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(".part")
    try:
        _fetch(BASE_URL + want.name, partial, want)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    partial.replace(target)
    return target


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Download TinyStories")
    parser.add_argument("--split", choices=sorted(FILES), default="valid")
    parser.add_argument("--dest", type=Path, default=Path("data"))
    args = parser.parse_args()
    path = download(args.split, args.dest)
    print(f"{path} ({path.stat().st_size / 1e6:.1f} MB, SHA-256 verified)")


if __name__ == "__main__":
    main()
