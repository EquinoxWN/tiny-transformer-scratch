# RFC 0001: tiny-transformer-scratch design

- **Status:** Accepted (M1 implemented)
- **Author:** AUTHOR_NAME
- **Created:** 2026

## Problem

Most people who use large language models cannot explain what happens between typing a prompt
and reading the answer: how text becomes numbers, how a network learns from gradients, and how
attention picks the next word. Frameworks hide all of it. This project builds a small GPT-style
model end to end with nothing but NumPy, then proves each piece is correct.

## Goals

- **Autograd (M1):** a reverse-mode engine whose every gradient is checked against finite
  differences.
- **Tokenizer (M1):** byte-level BPE trained on TinyStories, lossless on any Unicode input.
- **Model (M2):** multi-head causal self-attention, MLP, LayerNorm and residual blocks on the
  engine, trained with AdamW and a warmup + cosine schedule.
- **Generation (M3):** KV cache, temperature and top-k sampling, and output parity with a PyTorch
  re-implementation.

## Non-goals

- Competitive model quality. The target is a model that writes coherent short TinyStories.
- GPU kernels or mixed precision. PyTorch appears only as the parity reference in M3.
- Running as a hosted production service.

## Proposed design

![architecture](../architecture.png)

### Autograd engine (`autograd.py`)

- `Tensor` wraps a NumPy array and records `(parents, backward closure)` when any input needs
  gradients. `no_grad()` turns recording off for inference.
- `backward()` builds a topological order with an iterative DFS (deep graphs must not hit
  Python's recursion limit), then runs each closure once, after every consumer has added its
  gradient.
- Broadcasting is undone in the backward pass by summing over broadcast axes.
- Operations in M1: `+ - * /`, `**`, batched `@`, `sum`, `mean`, `reshape`, `transpose`,
  indexing (with `np.add.at`, so repeated embedding rows accumulate correctly), `exp`, `log`,
  `tanh`, `relu`, `gelu`, `softmax`, `log_softmax`, and `cross_entropy`.
- `softmax` and `log_softmax` subtract the row maximum, so logits of ±1000 do not overflow.

### Gradient checking (`gradcheck.py`)

The output is contracted with fixed random weights to a scalar, which exercises the whole
Jacobian. Each input element is nudged by ±1e-6 in float64, and the central difference must match
backprop (`atol=1e-6`, `rtol=1e-5`).

### Tokenizer (`bpe.py`)

- GPT-2 style pre-tokenisation (contractions, words with their leading space, numbers,
  punctuation, whitespace), then byte-level BPE inside each pre-token. Ids 0 to 255 are raw bytes,
  so any string round-trips.
- Training counts distinct pre-tokens once and keeps an incremental pair-count index, so each
  merge only touches the words that contain the merged pair. Ties break on the smaller pair, so
  training is deterministic.
- `<|endoftext|>` (TinyStories' story separator) is a single special token that never merges.
- Tokenizers are saved as small JSON files (`artifacts/tokenizer-*.json`).

## Alternatives considered

| Option | Why not (yet) |
|---|---|
| Use PyTorch autograd from the start | Hides exactly what the project exists to explain. PyTorch returns in M3 as the parity oracle. |
| Scalar (micrograd-style) autograd | Easy to read but thousands of times slower; a transformer needs vectorised array ops. |
| Hugging Face `tokenizers` | Fast, but a black box. The from-scratch BPE is small enough to read and test. |
| Character-level tokens | No training step, but sequences are 3 to 4 times longer, so attention costs far more for the same text. |
| `regex` package for Unicode-aware pre-tokenisation | Adds a C dependency. The ASCII classes work because TinyStories is almost all English, and byte fallback still round-trips any Unicode. |

## Measurement plan

- M1: gradient error per operation; tokenizer compression (bytes per token) and round-trip on
  held-out text, at vocabulary sizes 512, 1024 and 4096.
- M2: training and validation loss curves on TinyStories.
- M3: sample generations, attention maps, tokens per second with and without the KV cache, and
  maximum absolute difference against the PyTorch version.

## Milestones

- **M1 (done):** autograd engine plus gradient checks; BPE tokenizer trained on TinyStories.
- **M2:** transformer blocks, AdamW, learning-rate schedule, training loop.
- **M3:** KV-cache generation, sampling, PyTorch parity, published proof.

## Risks and open questions

- NumPy on CPU limits model size to a few million parameters. That is enough for TinyStories.
- float64 is needed for gradient checks, but training should use float32 for speed; M2 must keep
  both paths.
