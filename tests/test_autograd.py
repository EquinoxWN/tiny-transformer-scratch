"""Every operation's gradient is checked against central finite differences."""

import numpy as np
import pytest

from tiny_transformer_scratch.autograd import Tensor, cross_entropy, no_grad
from tiny_transformer_scratch.gradcheck import gradcheck

rng = np.random.default_rng(42)


def t(*shape, positive=False):
    """Random float64 tensor; positive=True keeps values away from zero."""
    data = rng.uniform(0.5, 2.0, shape) if positive else rng.standard_normal(shape)
    return Tensor(data)


CASES = {
    "add": (lambda a, b: a + b, [(3, 4), (3, 4)], False),
    "add broadcast row": (lambda a, b: a + b, [(3, 4), (4,)], False),
    "add broadcast column": (lambda a, b: a + b, [(3, 4), (3, 1)], False),
    "sub": (lambda a, b: a - b, [(2, 5), (2, 5)], False),
    "rsub scalar": (lambda a: 3.0 - a, [(4,)], False),
    "mul broadcast": (lambda a, b: a * b, [(2, 3, 4), (1, 3, 1)], False),
    "div": (lambda a, b: a / b, [(3, 3), (3, 3)], True),
    "rdiv scalar": (lambda a: 2.0 / a, [(5,)], True),
    "neg": (lambda a: -a, [(3,)], False),
    "pow": (lambda a: a**3, [(3, 2)], False),
    "pow fractional": (lambda a: a**0.5, [(4,)], True),
    "matmul": (lambda a, b: a @ b, [(3, 4), (4, 2)], False),
    "matmul batched": (lambda a, b: a @ b, [(2, 3, 4), (2, 4, 5)], False),
    "matmul broadcast weights": (lambda a, b: a @ b, [(2, 3, 4), (4, 5)], False),
    "sum all": (lambda a: a.sum(), [(3, 4)], False),
    "sum axis": (lambda a: a.sum(axis=1), [(3, 4)], False),
    "sum keepdims": (lambda a: a.sum(axis=0, keepdims=True), [(3, 4)], False),
    "mean axis": (lambda a: a.mean(axis=-1), [(2, 5)], False),
    "reshape": (lambda a: a.reshape(6, 2) @ Tensor(np.ones((2, 3))), [(3, 4)], False),
    "transpose": (lambda a: a.transpose(0, 2, 1), [(2, 3, 4)], False),
    "index slice": (lambda a: a[1:, ::2], [(4, 5)], False),
    "index repeated rows": (lambda a: a[np.array([0, 2, 0, 0])], [(3, 4)], False),
    "exp": (lambda a: a.exp(), [(3, 3)], False),
    "log": (lambda a: a.log(), [(3, 3)], True),
    "tanh": (lambda a: a.tanh(), [(3, 3)], False),
    "relu": (lambda a: a.relu(), [(4, 4)], False),
    "gelu": (lambda a: a.gelu(), [(4, 4)], False),
    "softmax": (lambda a: a.softmax(axis=-1), [(3, 5)], False),
    "softmax axis 0": (lambda a: a.softmax(axis=0), [(3, 5)], False),
    "log_softmax": (lambda a: a.log_softmax(axis=-1), [(3, 5)], False),
    "reused node": (lambda a: a * a + a.exp() * a, [(3,)], False),
}


@pytest.mark.parametrize("name", CASES)
def test_gradient_matches_finite_differences(name):
    fn, shapes, positive = CASES[name]
    inputs = [t(*s, positive=positive) for s in shapes]
    if name == "relu":  # keep inputs away from the kink at 0, where the derivative is undefined
        inputs[0].data = np.where(np.abs(inputs[0].data) < 0.1, 0.5, inputs[0].data)
    assert gradcheck(fn, inputs) < 1e-6


def test_cross_entropy_gradient():
    targets = np.array([0, 3, 1, 3])
    assert gradcheck(lambda z: cross_entropy(z, targets), [t(4, 5)]) < 1e-6


def test_cross_entropy_value_matches_formula():
    logits = t(4, 5)
    targets = np.array([0, 3, 1, 3])
    x = logits.data
    expected = np.mean(np.log(np.exp(x).sum(axis=1)) - x[np.arange(4), targets])
    assert cross_entropy(logits, targets).item() == pytest.approx(expected)


def test_softmax_is_stable_for_large_logits():
    out = Tensor(np.array([[1000.0, 1000.0, -1000.0]])).softmax()
    assert np.allclose(out.data, [[0.5, 0.5, 0.0]])


def test_gradients_accumulate_across_backward_calls():
    a = Tensor(np.array([2.0]), requires_grad=True)
    (a * 3).sum().backward()
    (a * 3).sum().backward()
    assert a.grad.tolist() == [6.0]


def test_no_grad_records_nothing():
    a = Tensor(np.ones(3), requires_grad=True)
    with no_grad():
        b = (a * 2).sum()
    assert not b.requires_grad
    with pytest.raises(ValueError, match="needs a scalar output"):
        (a * 2).backward()  # non-scalar output needs an explicit gradient


def test_deep_graph_does_not_hit_recursion_limit():
    a = Tensor(np.array([1.0]), requires_grad=True)
    x = a
    for _ in range(5000):
        x = x * 1.0 + 0.0
    x.sum().backward()
    assert a.grad.tolist() == [1.0]


def test_tiny_classifier_learns():
    """End to end: gradient descent on a linear softmax classifier drives loss down."""
    gen = np.random.default_rng(0)
    x = gen.standard_normal((64, 2))
    y = (x[:, 0] + x[:, 1] > 0).astype(int)
    w = Tensor(gen.standard_normal((2, 2)) * 0.1, requires_grad=True)
    first = None
    for _ in range(100):
        loss = cross_entropy(Tensor(x) @ w, y)
        first = first if first is not None else loss.item()
        w.grad = None
        loss.backward()
        w.data -= 0.5 * w.grad
    assert loss.item() < first / 3
