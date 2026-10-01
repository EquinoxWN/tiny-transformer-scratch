.PHONY: setup lint test data tokenizer bench ci audit

PY ?= python
TEXT = data/TinyStories-valid.txt

setup:
	$(PY) -m pip install -e ".[dev]"

lint:
	$(PY) -m ruff check .
	$(PY) -m ruff format --check .

test:
	$(PY) -m pytest -q

# Download the TinyStories validation split (~19 MB) into data/ (git-ignored).
data:
	$(PY) -m tiny_transformer_scratch.data

# Train the 1024-token tokenizer on the first 5 MB and measure it on the next 5 MB.
tokenizer: data
	$(PY) -m tiny_transformer_scratch.bpe train --input $(TEXT) --vocab-size 1024 --max-bytes 5000000 --out artifacts/tokenizer-1024.json
	$(PY) -m tiny_transformer_scratch.bpe stats --tokenizer artifacts/tokenizer-1024.json --input $(TEXT) --skip-bytes 5000000 --max-bytes 5000000

# Held-out compression for every committed tokenizer (numbers in docs/results/m1.md).
bench: data
	for v in 512 1024 4096; do $(PY) -m tiny_transformer_scratch.bpe stats --tokenizer artifacts/tokenizer-$$v.json --input $(TEXT) --skip-bytes 5000000 --max-bytes 5000000; done

# Known vulnerabilities in the installed dependencies.
audit:
	$(PY) -m pip_audit --skip-editable

ci: setup lint test
