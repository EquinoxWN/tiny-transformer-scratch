"""Byte-level byte-pair-encoding (BPE) tokenizer: training, encoding, decoding, saving.

Usage:
    python -m tiny_transformer_scratch.bpe train --input data/TinyStories-valid.txt \
        --vocab-size 1024 --max-bytes 5000000 --out artifacts/tokenizer-1024.json
    python -m tiny_transformer_scratch.bpe stats --tokenizer artifacts/tokenizer-1024.json \
        --input data/TinyStories-valid.txt --max-bytes 1000000
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter, defaultdict
from itertools import pairwise
from pathlib import Path

# GPT-2 style pre-tokenisation (ASCII classes, so it needs only the standard `re` module):
# contractions, words with their leading space, numbers, punctuation runs, whitespace.
PATTERN = r"""'(?:s|t|re|ve|m|ll|d)| ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+(?!\S)|\s+"""
END_OF_TEXT = "<|endoftext|>"

Pair = tuple[int, int]


def _merge_ids(ids: list[int], pair: Pair, new_id: int) -> list[int]:
    """Replace every non-overlapping occurrence of pair in ids with new_id."""
    out, i, n = [], 0, len(ids)
    while i < n:
        if i + 1 < n and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class BPETokenizer:
    """Byte-level BPE: ids 0-255 are raw bytes, then one id per merge, then special tokens."""

    def __init__(
        self,
        merges: list[Pair] | None = None,
        special_tokens: list[str] | None = None,
        pattern: str = PATTERN,
    ) -> None:
        self.pattern = pattern
        self._regex = re.compile(pattern)
        self.merges: list[Pair] = list(merges or [])
        self.ranks: dict[Pair, int] = {pair: i for i, pair in enumerate(self.merges)}
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        for i, (a, b) in enumerate(self.merges):
            self.vocab[256 + i] = self.vocab[a] + self.vocab[b]
        self.special: dict[str, int] = {}
        for tok in special_tokens or []:
            self.special[tok] = len(self.vocab) + len(self.special)
        self._special_ids = {v: k for k, v in self.special.items()}
        self._special_re = (
            re.compile("(" + "|".join(map(re.escape, self.special)) + ")") if self.special else None
        )
        self._cache: dict[str, list[int]] = {}

    @property
    def vocab_size(self) -> int:
        return len(self.vocab) + len(self.special)

    # ---- training ---------------------------------------------------------

    @classmethod
    def train(
        cls, text: str, vocab_size: int, special_tokens: list[str] | None = None, pattern: str = PATTERN
    ) -> BPETokenizer:
        """Learn vocab_size - 256 - len(special) merges from text."""
        special_tokens = special_tokens or []
        n_merges = vocab_size - 256 - len(special_tokens)
        if n_merges < 0:
            raise ValueError("vocab_size must leave room for 256 byte tokens and the special tokens")
        regex = re.compile(pattern)
        chunks = re.split("|".join(map(re.escape, special_tokens)), text) if special_tokens else [text]

        # Count each distinct pre-token once; merges then work on (word, frequency) pairs.
        freq: Counter[str] = Counter()
        for chunk in chunks:
            freq.update(regex.findall(chunk))
        words: list[list[int]] = [list(w.encode("utf-8")) for w in freq]
        counts: list[int] = list(freq.values())

        pair_counts: Counter[Pair] = Counter()
        where: defaultdict[Pair, set[int]] = defaultdict(set)
        for wi, ids in enumerate(words):
            for pair in pairwise(ids):
                pair_counts[pair] += counts[wi]
                where[pair].add(wi)

        merges: list[Pair] = []
        for step in range(n_merges):
            if not pair_counts:
                break
            # Highest count wins; ties go to the smallest pair so training is deterministic.
            best = min(pair_counts, key=lambda p: (-pair_counts[p], p))
            if pair_counts[best] < 2:
                break
            new_id = 256 + step
            merges.append(best)
            for wi in list(where[best]):
                ids = words[wi]
                c = counts[wi]
                for pair in pairwise(ids):
                    pair_counts[pair] -= c
                    if pair_counts[pair] <= 0:
                        del pair_counts[pair]
                merged = _merge_ids(ids, best, new_id)
                words[wi] = merged
                for pair in pairwise(merged):
                    pair_counts[pair] += c
                    where[pair].add(wi)
            del where[best]
        return cls(merges, special_tokens, pattern)

    # ---- encoding ---------------------------------------------------------

    def _encode_chunk(self, chunk: str) -> list[int]:
        """Apply merges to one pre-token, lowest rank first."""
        cached = self._cache.get(chunk)
        if cached is not None:
            return cached
        ids = list(chunk.encode("utf-8"))
        while len(ids) > 1:
            pair = min(pairwise(ids), key=lambda p: self.ranks.get(p, len(self.ranks)))
            rank = self.ranks.get(pair)
            if rank is None:
                break
            ids = _merge_ids(ids, pair, 256 + rank)
        if len(self._cache) < 100_000:
            self._cache[chunk] = ids
        return ids

    def encode(self, text: str) -> list[int]:
        """Turn text into token ids; special tokens map to their own ids."""
        parts = self._special_re.split(text) if self._special_re else [text]
        out: list[int] = []
        for part in parts:
            if part in self.special:
                out.append(self.special[part])
                continue
            for chunk in self._regex.findall(part):
                out.extend(self._encode_chunk(chunk))
        return out

    def decode(self, ids: list[int]) -> str:
        """Turn token ids back into text."""
        buf = bytearray()
        for i in ids:
            if i in self._special_ids:
                buf.extend(self._special_ids[i].encode("utf-8"))
            elif i in self.vocab:
                buf.extend(self.vocab[i])
            else:
                raise KeyError(f"unknown token id {i}")
        return buf.decode("utf-8", errors="replace")

    # ---- persistence ------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Write the tokenizer as JSON."""
        doc = {
            "version": 1,
            "pattern": self.pattern,
            "special_tokens": self.special,
            "merges": [list(m) for m in self.merges],
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(doc) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> BPETokenizer:
        """Read a tokenizer written by save()."""
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
        if doc.get("version") != 1:
            raise ValueError(f"unsupported tokenizer version {doc.get('version')!r}")
        specials = sorted(doc["special_tokens"], key=doc["special_tokens"].get)
        return cls([tuple(m) for m in doc["merges"]], specials, doc["pattern"])


def _read(path: str, max_bytes: int, skip_bytes: int = 0) -> str:
    """Read at most max_bytes of UTF-8 text after skip_bytes, dropping split characters."""
    with open(path, "rb") as f:
        f.seek(skip_bytes)
        raw = f.read(max_bytes)
    return raw.decode("utf-8", errors="ignore")


def main(argv: list[str] | None = None) -> None:
    """Command-line entry point: train or report stats."""
    parser = argparse.ArgumentParser(description="Byte-level BPE tokenizer")
    sub = parser.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train", help="learn merges from a text file")
    t.add_argument("--input", required=True)
    t.add_argument("--vocab-size", type=int, default=1024)
    t.add_argument("--max-bytes", type=int, default=5_000_000)
    t.add_argument("--out", required=True)
    s = sub.add_parser("stats", help="measure compression on a text file")
    s.add_argument("--tokenizer", required=True)
    s.add_argument("--input", required=True)
    s.add_argument("--max-bytes", type=int, default=1_000_000)
    s.add_argument("--skip-bytes", type=int, default=0, help="measure on held-out text past this offset")
    args = parser.parse_args(argv)

    text = _read(args.input, args.max_bytes, getattr(args, "skip_bytes", 0))
    if args.cmd == "train":
        start = time.perf_counter()
        tok = BPETokenizer.train(text, args.vocab_size, special_tokens=[END_OF_TEXT])
        tok.save(args.out)
        print(
            f"trained {len(tok.merges)} merges on {len(text.encode()):,} bytes "
            f"in {time.perf_counter() - start:.1f}s -> {args.out}"
        )
    else:
        tok = BPETokenizer.load(args.tokenizer)
        start = time.perf_counter()
        ids = tok.encode(text)
        elapsed = time.perf_counter() - start
        n_bytes = len(text.encode())
        print(
            f"bytes={n_bytes:,} tokens={len(ids):,} bytes_per_token={n_bytes / len(ids):.2f} "
            f"vocab={tok.vocab_size} roundtrip={'ok' if tok.decode(ids) == text else 'FAILED'} "
            f"encode_mb_per_s={n_bytes / 1e6 / elapsed:.2f}"
        )


if __name__ == "__main__":
    main()
