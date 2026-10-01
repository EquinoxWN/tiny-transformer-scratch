# ADR 0002: NumPy-only engine, gradient checks in float64

- **Status:** Accepted

## Context

The engine must be both explainable and provably correct. A wrong gradient does not crash; it
silently trains a worse model, which is the hardest kind of bug to find later in M2.

## Decision

- The engine depends only on NumPy. There is no PyTorch or JAX at runtime.
- Every operation ships with a finite-difference gradient test. The check runs in float64 with a
  step of 1e-6, where central differences are accurate to about 1e-10, so a real bug cannot hide
  inside the tolerance.
- The engine keeps the dtype it is given, so M2 can train in float32 while the tests stay in
  float64.

## Consequences

- Every new operation needs a gradcheck case before a model can use it. The test table in
  `tests/test_autograd.py` makes that cheap.
- Worst error measured in M1 is 1.5e-9 (log_softmax), far below the 1e-6 threshold.
- CPU speed limits model size. That is accepted, because the goal is understanding, not scale.
