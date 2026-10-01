"""Download the TinyStories text files into ./data (not committed to git).

Usage:
    python -m tiny_transformer_scratch.data            # validation split, ~19 MB
    python -m tiny_transformer_scratch.data --split train   # ~1.9 GB
"""

from __future__ import annotations

import argparse
import shutil
import urllib.request
from pathlib import Path

BASE_URL = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/"
FILES = {"valid": "TinyStories-valid.txt", "train": "TinyStories-train.txt"}


def download(split: str = "valid", dest: Path = Path("data")) -> Path:
    """Fetch one split unless it is already present; return its path."""
    name = FILES[split]
    target = dest / name
    if target.exists():
        return target
    dest.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(".part")
    url = BASE_URL + name  # fixed https URL, never user input
    with urllib.request.urlopen(url, timeout=60) as resp, open(partial, "wb") as out:  # noqa: S310
        shutil.copyfileobj(resp, out, length=1 << 20)
    partial.replace(target)
    return target


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Download TinyStories")
    parser.add_argument("--split", choices=sorted(FILES), default="valid")
    parser.add_argument("--dest", type=Path, default=Path("data"))
    args = parser.parse_args()
    path = download(args.split, args.dest)
    print(f"{path} ({path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
