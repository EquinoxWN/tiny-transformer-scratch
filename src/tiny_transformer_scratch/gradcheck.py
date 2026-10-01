"""Finite-difference gradient checking for the autograd engine."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from tiny_transformer_scratch.autograd import Tensor, no_grad


def gradcheck(
    fn: Callable[..., Tensor],
    inputs: Sequence[Tensor],
    eps: float = 1e-6,
    atol: float = 1e-6,
    rtol: float = 1e-5,
    seed: int = 0,
) -> float:
    """Compare backprop against central differences; return the max abs error or raise."""
    for x in inputs:
        if x.data.dtype != np.float64:
            raise TypeError("gradcheck needs float64 inputs for accurate differences")
        x.data = np.ascontiguousarray(x.data)  # so reshape(-1) below is a view
        x.requires_grad = True
        x.grad = None

    probe = fn(*inputs)
    # Random weights turn any output into a scalar that exercises the whole Jacobian.
    weights = np.random.default_rng(seed).standard_normal(probe.shape)

    def scalar() -> float:
        with no_grad():
            return float((fn(*inputs).data * weights).sum())

    (probe * weights).sum().backward()

    worst = 0.0
    for n, x in enumerate(inputs):
        analytic = np.zeros_like(x.data) if x.grad is None else x.grad
        numeric = np.zeros_like(x.data)
        flat = x.data.reshape(-1)  # a view: edits below change x.data in place
        for i in range(flat.size):
            original = flat[i]
            flat[i] = original + eps
            plus = scalar()
            flat[i] = original - eps
            minus = scalar()
            flat[i] = original
            numeric.reshape(-1)[i] = (plus - minus) / (2 * eps)
        err = np.abs(analytic - numeric)
        worst = max(worst, float(err.max(initial=0.0)))
        if not np.allclose(analytic, numeric, atol=atol, rtol=rtol):
            i = int(err.argmax())
            raise AssertionError(
                f"input {n}: gradient mismatch at flat index {i}: "
                f"analytic={analytic.reshape(-1)[i]:.8g} numeric={numeric.reshape(-1)[i]:.8g}"
            )
    return worst
