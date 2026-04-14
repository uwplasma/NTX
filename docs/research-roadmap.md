# Research Roadmap

NTX is ship-ready as a monoenergetic transport package. The next step is to
turn it into a research platform for open stellarator transport and
optimization problems.

This page summarizes the active development lanes, why they matter, and where
they map onto the current source tree.

## Research Goal

The near-term goal is:

1. retain the current fast monoenergetic solver and trusted imported workflows,
2. make derivatives robust enough for large optimization loops,
3. make profile and bootstrap-current workflows first-class,
4. and scale large database-generation campaigns across CPUs and GPUs.

The governing formulation is still the Legendre-space monoenergetic equation
described in Javier Escoto's thesis,
[arXiv:2510.27513](https://arxiv.org/abs/2510.27513). NTX already solves the
forward problem described there in:

- [`src/ntx/operators.py`](../src/ntx/operators.py)
- [`src/ntx/solver.py`](../src/ntx/solver.py)
- [`src/ntx/transport.py`](../src/ntx/transport.py)

The research-grade roadmap starts where the shipped `1.0` package currently
stops.

## Why These Lanes Matter

Several current themes in stellarator research shape the next steps for NTX:

- direct optimization of neoclassical objectives instead of post-processing
  transport tables,
- differentiable programming and adjoint methods for geometry and profile
  sensitivities,
- low-bootstrap-current and hidden-symmetry design problems,
- and scalable transport database generation for predictive workflows.

Relevant references:

- Javier Escoto, PhD thesis:
  [arXiv:2510.27513](https://arxiv.org/abs/2510.27513)
- Adjoint neoclassical optimization:
  [arXiv:1904.06430](https://arxiv.org/abs/1904.06430)
- Differentiable programming for plasma workflows:
  [arXiv:2410.11161](https://arxiv.org/abs/2410.11161)
- Zero-bootstrap-current piecewise omnigenity:
  [arXiv:2505.02546](https://arxiv.org/abs/2505.02546)
- Hidden-symmetry optimization:
  [arXiv:2502.09350](https://arxiv.org/abs/2502.09350)
- Combined omnigenity and piecewise-omnigenity optimization:
  [arXiv:2603.12139](https://arxiv.org/abs/2603.12139)

## Phase 1: Optimization-Grade Derivatives

Current state:

- the imported NTX solve is differentiable end to end,
- autodiff examples already exist for inverse problems and bootstrap-current
  optimization,
- but the dense solve still relies on generic reverse-mode differentiation.

That is sufficient for small examples, but not for large optimization loops
with many geometry parameters.

The next target is an implicit or adjoint derivative path for the dense solve:

```{math}
A(x) u(x) = b(x), \qquad
\frac{dJ}{dx}
=
\frac{\partial J}{\partial x}
- \lambda^\top
\left(
\frac{\partial A}{\partial x} u
- \frac{\partial b}{\partial x}
\right),
\qquad
A^\top \lambda = \frac{\partial J}{\partial u}.
```

This belongs primarily in:

- [`src/ntx/solver.py`](../src/ntx/solver.py)
- [`src/ntx/operators.py`](../src/ntx/operators.py)
- [`src/ntx/autodiff.py`](../src/ntx/autodiff.py)

First deliverables:

1. derivative audit against finite differences,
2. custom VJP or equivalent implicit derivative for the prepared solve,
3. tests comparing direct autodiff and implicit gradients on small systems,
4. runtime and memory comparisons for direct versus implicit differentiation.

The first step is already started in NTX with the derivative-audit workflow in
[`examples/derivative_audit.py`](../examples/derivative_audit.py), documented in
the [Autodiff](autodiff.md) and [Examples](examples.md) pages.

NTX now also exposes an explicit custom-VJP contract point in
[`src/ntx/solver.py`](../src/ntx/solver.py):

- `solve_prepared_coefficient_vector(...)`
- `solve_prepared_coefficient_vector_vjp(...)`

The current backward rule is still exact reverse-mode differentiation of the
raw prepared coefficient kernel. The point of this interface is to give NTX a
stable place to swap in a true implicit or adjoint derivative next.

## Phase 2: Profile-Grade Transport Workflows

NTX already exports NEOPAX-compatible monoenergetic arrays and HDF5 scans in:

- [`src/ntx/database.py`](../src/ntx/database.py)
- [`src/ntx/neopax.py`](../src/ntx/neopax.py)

The next step is to promote profile workflows to a first-class API:

- ambipolar `E_r(r)` root finding,
- bootstrap-current profile closure,
- differentiable profile sensitivity,
- and compressed database generation for repeated transport solves.

This is motivated by the way downstream profile tools consume NTX transport
data and rescale coefficients with radial-coordinate factors and collisionality
normalizations.

## Phase 3: Geometry Breadth For Open Design Problems

The code is already strong on VMEC/Boozer workflows. Research work will demand
more than trusted loading:

- hidden-symmetry studies,
- piecewise-omnigenous studies,
- low-bootstrap-current optimization across a radial family,
- and eventually geometry perturbation studies that stay in memory instead of
  bouncing through files.

That pushes on:

- [`src/ntx/geometry.py`](../src/ntx/geometry.py)
- [`src/ntx/vmec_jax_vmec.py`](../src/ntx/vmec_jax_vmec.py)
- [`src/ntx/vmec_jax_backend.py`](../src/ntx/vmec_jax_backend.py)

## Phase 4: Production Throughput

NTX already has:

- serial batched JAX scans,
- host/device parallel scans,
- and multiprocess one-worker-per-device scans.

The performance conclusion from the current benchmarks is:

- serial batched JAX is the right default for small and medium studies,
- multiprocess execution is the throughput lane for larger campaigns.

The next work is not just “more parallelism.” It is:

1. better prepared-geometry reuse for large database scans,
2. stable multi-device throughput on production grids,
3. clear crossover maps for CPU, GPU, and multi-process paths,
4. and, if needed, multi-host scan orchestration.

This work belongs mainly in:

- [`src/ntx/solver.py`](../src/ntx/solver.py)
- [`src/ntx/parallel.py`](../src/ntx/parallel.py)
- [`scripts/benchmark_scaling.py`](../scripts/benchmark_scaling.py)
- [`scripts/profile_parallel_runtime.py`](../scripts/profile_parallel_runtime.py)

## Phase 5: Physics Expansion

NTX is intentionally focused on the monoenergetic Lorentz-scattering problem.
Research-grade transport studies will eventually need:

- momentum-restoring closures,
- stronger ambipolar electric-field workflows,
- broader finite-collisionality validation,
- and possibly energy convolution layers for higher-level transport tasks.

This should happen only after the derivative and profile layers above are
stable.

## Adjacent-Code Lessons Incorporated Into The Plan

The roadmap is informed by nearby codes without turning NTX into a wrapper
around them:

- profile tools expect clean monoenergetic database interfaces and radial
  normalization hooks,
- practical multi-GPU throughput often works better as one worker per case or
  scan point than as one giant sharded solve,
- and adjoint or derivative diagnostics need direct validation against finite
  differences before they are trusted in optimization loops.

Those lessons are already reflected in the current NTX public API, parallel
execution notes, and the next derivative milestone.

## Immediate Milestone

The active implementation milestone is:

1. complete the derivative-audit workflow,
2. introduce an implicit-derivative pathway for the prepared dense solve,
3. validate it against direct autodiff and finite differences,
4. then use it in a stronger bootstrap-current optimization example.

This is the shortest path from a strong forward solver to a research tool that
can address open design and optimization problems.
