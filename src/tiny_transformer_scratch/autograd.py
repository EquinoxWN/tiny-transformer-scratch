"""A small reverse-mode autograd engine on NumPy arrays."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from typing import Any

import numpy as np

_grad_enabled = True


@contextmanager
def no_grad() -> Iterator[None]:
    """Disable graph recording inside the block, e.g. for inference."""
    global _grad_enabled
    previous, _grad_enabled = _grad_enabled, False
    try:
        yield
    finally:
        _grad_enabled = previous


def _unbroadcast(grad: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Sum grad over the axes NumPy broadcast, so it matches shape."""
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for axis, size in enumerate(shape):
        if size == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad


def _as_array(data: Any) -> np.ndarray:
    """Convert input to a floating-point array."""
    arr = np.asarray(data)
    return arr if np.issubdtype(arr.dtype, np.floating) else arr.astype(np.float64)


class Tensor:
    """An array that records the operations applied to it for backpropagation."""

    __array_priority__ = 100  # make `ndarray <op> Tensor` dispatch to Tensor

    def __init__(self, data: Any, requires_grad: bool = False) -> None:
        self.data = _as_array(data)
        self.requires_grad = requires_grad
        self.grad: np.ndarray | None = None
        self._parents: tuple[Tensor, ...] = ()
        self._backward: Callable[[np.ndarray], None] | None = None
        self._op = ""

    # ---- basics -----------------------------------------------------------

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    def item(self) -> float:
        """Return the value of a one-element tensor."""
        return float(self.data.item())

    def __repr__(self) -> str:
        return f"Tensor(shape={self.shape}, op={self._op or 'leaf'}, requires_grad={self.requires_grad})"

    def _accumulate(self, grad: np.ndarray) -> None:
        """Add grad into self.grad."""
        if self.requires_grad:
            self.grad = grad.copy() if self.grad is None else self.grad + grad

    @staticmethod
    def _result(
        data: np.ndarray, parents: Sequence[Tensor], op: str, backward: Callable[[np.ndarray], None]
    ) -> Tensor:
        """Wrap data as an op output, recording the graph when needed."""
        out = Tensor(data)
        if _grad_enabled and any(p.requires_grad for p in parents):
            out.requires_grad = True
            out._parents = tuple(parents)
            out._backward = backward
            out._op = op
        return out

    def backward(self, grad: Any = None) -> None:
        """Backpropagate from this tensor through the recorded graph."""
        if grad is None:
            if self.data.size != 1:
                raise ValueError("backward() without a gradient needs a scalar output")
            grad = np.ones_like(self.data)
        order: list[Tensor] = []
        seen: set[int] = set()
        stack: list[tuple[Tensor, bool]] = [(self, False)]
        while stack:  # iterative post-order DFS; deep graphs must not hit the recursion limit
            node, expanded = stack.pop()
            if expanded:
                order.append(node)
                continue
            if id(node) in seen:
                continue
            seen.add(id(node))
            stack.append((node, True))
            stack.extend((p, False) for p in node._parents if id(p) not in seen)
        self.grad = _as_array(grad) if self.grad is None else self.grad + _as_array(grad)
        for node in reversed(order):
            if node._backward is not None and node.grad is not None:
                node._backward(node.grad)

    # ---- elementwise arithmetic ------------------------------------------

    def __add__(self, other: Any) -> Tensor:
        other = _lift(other)

        def backward(g: np.ndarray) -> None:
            self._accumulate(_unbroadcast(g, self.shape))
            other._accumulate(_unbroadcast(g, other.shape))

        return Tensor._result(self.data + other.data, (self, other), "add", backward)

    def __mul__(self, other: Any) -> Tensor:
        other = _lift(other)

        def backward(g: np.ndarray) -> None:
            self._accumulate(_unbroadcast(g * other.data, self.shape))
            other._accumulate(_unbroadcast(g * self.data, other.shape))

        return Tensor._result(self.data * other.data, (self, other), "mul", backward)

    def __truediv__(self, other: Any) -> Tensor:
        other = _lift(other)

        def backward(g: np.ndarray) -> None:
            self._accumulate(_unbroadcast(g / other.data, self.shape))
            other._accumulate(_unbroadcast(-g * self.data / other.data**2, other.shape))

        return Tensor._result(self.data / other.data, (self, other), "div", backward)

    def __neg__(self) -> Tensor:
        return Tensor._result(-self.data, (self,), "neg", lambda g: self._accumulate(-g))

    def __sub__(self, other: Any) -> Tensor:
        return self + (-_lift(other))

    def __rsub__(self, other: Any) -> Tensor:
        return _lift(other) + (-self)

    def __rtruediv__(self, other: Any) -> Tensor:
        return _lift(other) / self

    __radd__ = __add__
    __rmul__ = __mul__

    def __pow__(self, exponent: float) -> Tensor:
        if isinstance(exponent, Tensor):
            raise TypeError("only constant exponents are supported")
        e = float(exponent)
        return Tensor._result(
            self.data**e, (self,), "pow", lambda g: self._accumulate(g * e * self.data ** (e - 1))
        )

    # ---- linear algebra and shape ----------------------------------------

    def __matmul__(self, other: Any) -> Tensor:
        other = _lift(other)
        if self.ndim < 2 or other.ndim < 2:
            raise ValueError("matmul needs operands with at least 2 dimensions")

        def backward(g: np.ndarray) -> None:
            self._accumulate(_unbroadcast(g @ np.swapaxes(other.data, -1, -2), self.shape))
            other._accumulate(_unbroadcast(np.swapaxes(self.data, -1, -2) @ g, other.shape))

        return Tensor._result(self.data @ other.data, (self, other), "matmul", backward)

    def sum(self, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> Tensor:
        """Sum over axis (all axes by default)."""

        def backward(g: np.ndarray) -> None:
            if axis is not None and not keepdims:
                g = np.expand_dims(g, axis)
            self._accumulate(np.broadcast_to(g, self.shape))

        return Tensor._result(self.data.sum(axis=axis, keepdims=keepdims), (self,), "sum", backward)

    def mean(self, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> Tensor:
        """Average over axis (all axes by default)."""
        count = self.data.size if axis is None else int(np.prod([self.shape[a] for a in np.atleast_1d(axis)]))
        return self.sum(axis=axis, keepdims=keepdims) / count

    def reshape(self, *shape: int) -> Tensor:
        """Return the same values with a new shape."""
        return Tensor._result(
            self.data.reshape(*shape), (self,), "reshape", lambda g: self._accumulate(g.reshape(self.shape))
        )

    def transpose(self, *axes: int) -> Tensor:
        """Permute axes (reverse them by default)."""
        order = axes or tuple(reversed(range(self.ndim)))
        inverse = tuple(np.argsort(order))
        return Tensor._result(
            self.data.transpose(order), (self,), "transpose", lambda g: self._accumulate(g.transpose(inverse))
        )

    def __getitem__(self, index: Any) -> Tensor:
        def backward(g: np.ndarray) -> None:
            full = np.zeros_like(self.data)
            np.add.at(full, index, g)  # correct even when an index repeats (embedding lookups)
            self._accumulate(full)

        return Tensor._result(self.data[index], (self,), "index", backward)

    # ---- nonlinearities ---------------------------------------------------

    def exp(self) -> Tensor:
        """Elementwise e**x."""
        out = np.exp(self.data)
        return Tensor._result(out, (self,), "exp", lambda g: self._accumulate(g * out))

    def log(self) -> Tensor:
        """Elementwise natural log."""
        return Tensor._result(np.log(self.data), (self,), "log", lambda g: self._accumulate(g / self.data))

    def tanh(self) -> Tensor:
        """Elementwise tanh."""
        out = np.tanh(self.data)
        return Tensor._result(out, (self,), "tanh", lambda g: self._accumulate(g * (1 - out**2)))

    def relu(self) -> Tensor:
        """Elementwise max(x, 0)."""
        return Tensor._result(
            np.maximum(self.data, 0), (self,), "relu", lambda g: self._accumulate(g * (self.data > 0))
        )

    def gelu(self) -> Tensor:
        """GELU with the tanh approximation used by GPT-2."""
        x = self.data
        k = np.sqrt(2 / np.pi)
        t = np.tanh(k * (x + 0.044715 * x**3))
        out = 0.5 * x * (1 + t)

        def backward(g: np.ndarray) -> None:
            dt = (1 - t**2) * k * (1 + 3 * 0.044715 * x**2)
            self._accumulate(g * (0.5 * (1 + t) + 0.5 * x * dt))

        return Tensor._result(out, (self,), "gelu", backward)

    def softmax(self, axis: int = -1) -> Tensor:
        """Numerically stable softmax along axis."""
        e = np.exp(self.data - self.data.max(axis=axis, keepdims=True))
        s = e / e.sum(axis=axis, keepdims=True)
        return Tensor._result(
            s, (self,), "softmax", lambda g: self._accumulate(s * (g - (g * s).sum(axis=axis, keepdims=True)))
        )

    def log_softmax(self, axis: int = -1) -> Tensor:
        """Numerically stable log-softmax along axis."""
        shifted = self.data - self.data.max(axis=axis, keepdims=True)
        out = shifted - np.log(np.exp(shifted).sum(axis=axis, keepdims=True))

        def backward(g: np.ndarray) -> None:
            self._accumulate(g - np.exp(out) * g.sum(axis=axis, keepdims=True))

        return Tensor._result(out, (self,), "log_softmax", backward)


def _lift(value: Any) -> Tensor:
    """Wrap constants as non-trainable tensors."""
    return value if isinstance(value, Tensor) else Tensor(value)


def cross_entropy(logits: Tensor, targets: np.ndarray) -> Tensor:
    """Mean negative log-likelihood of integer targets under row-wise softmax."""
    targets = np.asarray(targets)
    if logits.ndim != 2 or targets.shape != (logits.shape[0],):
        raise ValueError("expected logits (N, C) and integer targets (N,)")
    picked = logits.log_softmax(axis=-1)[np.arange(len(targets)), targets]
    return -picked.mean()
