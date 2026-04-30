# Research Roadmap

NTX has a strong monoenergetic transport base and a published `0.2.0` package.
The next step is to turn the current open lanes into a research platform for
open stellarator transport and optimization problems.

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

The research-grade roadmap starts where the shipped `0.2.0` package currently
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
- Direct neoclassical ion-transport optimization:
  [arXiv:2406.04147](https://arxiv.org/abs/2406.04147)
- Near-axis quasi-isodynamic construction and verification:
  [JPP 2025](https://doi.org/10.1017/S0022377825000157)
- Zero-bootstrap-current piecewise omnigenity:
  [arXiv:2505.02546](https://arxiv.org/abs/2505.02546)
- Hidden-symmetry optimization:
  [arXiv:2502.09350](https://arxiv.org/abs/2502.09350)
- Combined omnigenity and piecewise-omnigenity optimization:
  [arXiv:2603.12139](https://arxiv.org/abs/2603.12139)

## Phase 1: Optimization-Grade Derivatives

Current state:

- the imported NTX solve is differentiable end to end,
- the prepared dense solve now has an implicit-adjoint VJP contract,
- autodiff examples already exist for inverse problems and bootstrap-current
  optimization,
- but broader geometry pullbacks, factorization reuse, and implicit-equilibrium
  promotion remain constrained by artifact-backed derivative gates.

That is sufficient for small examples and fixed-geometry profile studies, but
large optimization loops with many geometry parameters still need tighter
memory and derivative-path control.

The maintained implicit-adjoint dense-solve path follows:

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

Completed anchors:

1. derivative audit against finite differences,
2. custom VJP or equivalent implicit derivative for the prepared solve,
3. tests comparing direct autodiff and implicit gradients on small systems,
4. runtime and memory comparisons for direct versus implicit differentiation.

These are represented in NTX by:

- the derivative-audit workflow in
  [`examples/derivative_audit.py`](../examples/derivative_audit.py),
- and the prepared-derivative timing study in
  [`examples/derivative_path_benchmark.py`](../examples/derivative_path_benchmark.py),

both documented in the [Autodiff](autodiff.md) and [Examples](examples.md)
pages.

The first multi-parameter geometry-control stress benchmark is also in place:

- [`examples/geometry_control_derivative_benchmark.py`](../examples/geometry_control_derivative_benchmark.py)
  controls three independent Boozer harmonics on an owned analytic surface,
- its JSON artifact records direct autodiff, centered finite-difference
  Jacobians, and AD/FD mismatch metrics for `D11`, `D31`, and `D33`,
- and it is deliberately classified as a stress benchmark until the same audit
  is transferred to reusable VMEC/Boozer geometry-control families.

That transfer is now started on repository-owned file-backed inputs:

- [`examples/file_backed_geometry_control_derivative_benchmark.py`](../examples/file_backed_geometry_control_derivative_benchmark.py)
  repeats the same AD versus centered-finite-difference audit on sample Boozer
  and VMEC-backed surfaces loaded from files,
- so the remaining gap is no longer "analytic versus real geometry", but
  rather the broader reusable geometry-family basis and prepared
  implicit-adjoint geometry pullbacks.

The next imported boundary-control slice is now also benchmarked:

- [`examples/boundary_forward_mode_current_derivative_benchmark.py`](../examples/boundary_forward_mode_current_derivative_benchmark.py)
  uses repository-owned `vmec_jax` boundary controls, a boundary-projected
  VMEC state, `booz_xform_jax`, NTX, and NEOPAX to audit two scalar outputs
  against centered finite differences,
- the current validated contract is forward-mode on this low-dimensional
  boundary-projected geometry lane,
- and the supported self-consistent transfer is now the explicit-relaxed
  equilibrium sensitivity workflow rather than the projected-boundary map.

That implicit-equilibrium transfer is now closed as a non-shipping diagnostic on
the committed QA case:

- [`examples/implicit_equilibrium_forward_mode_derivative_benchmark.py`](../examples/implicit_equilibrium_forward_mode_derivative_benchmark.py)
  uses the implicit fixed-boundary `vmec_jax` residual solve with
  `residual_tangent_mode="auto"`,
- it records AD versus centered-finite-difference behavior for equilibrium
  volume, a Boozer scalar, and an NTX monoenergetic transport observable,
- the current result is asymmetric: equilibrium volume matches, but residual
  contraction is absent and the Boozer and NTX transport observables fail
  tangent parity,
- so the remaining work is now "restore this lane only after the backend
  residual solve contracts and Boozer/NTX tangent parity passes",
- and the current reverse-mode failure is concrete: the matching Boozer-scalar
  gradient is unavailable because JAX rejects reverse mode through the dynamic
  implicit solve.

That self-consistent forward-mode transfer is now in place on committed QA and
QH family cases:

- [`examples/explicit_relaxed_boundary_current_derivative_benchmark.py`](../examples/explicit_relaxed_boundary_current_derivative_benchmark.py)
  uses an explicitly relaxed fixed-boundary `vmec_jax` solve, `booz_xform_jax`,
  NTX, and NEOPAX on the low-resolution QA input and the lighter QH warm-start
  input,
- its JSON artifact records ordinary-versus-explicit primal-volume agreement in
  addition to the AD versus centered-finite-difference mismatch metrics on both
  cases,
- so the remaining open work is no longer "projected versus relaxed
  equilibrium", but rather additional geometry families, integrated-current
  objectives on the explicit-relaxed lane, and reverse-mode equilibrium paths.

NTX now also exposes an explicit custom-VJP contract point in
[`src/ntx/solver.py`](../src/ntx/solver.py):

- `solve_prepared_coefficient_vector(...)`
- `solve_prepared_coefficient_vector_vjp(...)`

The current backward rule now uses an implicit-adjoint block solve for the
prepared dense system. The next derivative step is to specialize that adjoint
further so it reuses even more of the prepared factorization and reduces memory
pressure on larger optimization scans.

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

The committed artifact-backed status is summarized in
[`examples/geometry_family_breadth_summary.py`](../examples/geometry_family_breadth_summary.py).
That script reads the analytic, file-backed, boundary-projected,
explicit-relaxed, and implicit-equilibrium derivative artifacts and produces a
manuscript-ready figure without rerunning expensive equilibrium solves. It is a
stress-summary lane: retired implicit Boozer/transport diagnostics remain
excluded from promoted claims.

The direct VMEC transport-convergence breadth check now lives in
[`examples/geometry_family_transport_convergence.py`](../examples/geometry_family_transport_convergence.py).
It discovers local public VMEC examples from `vmec_jax`, STELLOPT, and SIMSOPT,
then records production `D11/D31/D33` grid-ladder behavior across tokamak,
precise-QS, QI-style, W7-X EIM/EJM, and stellarator-family inputs. `D13` and
the normalized Onsager residual are retained in the JSON sidecar. That closes
the NTX geometry-family stress artifact; independent-code parity,
radial/electric-field/collisionality ladders, and owned W7-X KJM input coverage
remain promotion requirements.

The `Er_tilde` HDF5 export path now has one explicit geometry-backend
validation lane before any Boozer-file-backed coefficient claim is promoted:

- keep the VMEC surface backend as the default validation path for QA/QH and
  Redl/SFINCS-style benchmark generation,
- keep direct `boozmn` loading as an explicit audit mode until the coefficient
  difference from the VMEC-backed path is explained,
- diagnose the mismatch on the same VMEC `wout`, matching Boozer transform,
  identical `rho`, `nu_v`, and `Er_tilde` grids, and identical angular/pitch
  resolution,
- compare the imported geometry channels before solving transport:
  `B_{mn}`, `R_{mn}`, `B_{00}`, `iota`, Boozer `G/I`, `psi_p`, Jacobian sign,
  radial-coordinate factors, mode filtering, interpolation radius, and
  `Er_tilde -> Er/Es` conversion,
- then compare the assembled NTX sources/operators and final `D11`, `D31`, and
  `D33` coefficient ladders at increasing `N_theta`, `N_zeta`, and `N_xi`,
- accept the direct `boozmn` backend for promoted examples only after the
  coefficient gap is traced to a documented normalization, interpolation, sign,
  or mode-selection convention and the fix transfers to both precise-QS and
  W7-X-style cases without changing the VMEC-backed validation path.

## Phase 4: Production Throughput

NTX already has:

- serial batched JAX scans,
- host/device parallel scans,
- and multiprocess one-worker-per-device scans.

The performance conclusion from the current benchmarks is:

- serial batched JAX is the right default for small and medium studies,
- single-process device-parallel CPU scans now show production-grid crossover
  and fixed-workload strong-scaling wins once the scan is large enough,
- the tested two-GPU workstation exposes two CUDA devices but only one healthy
  NTX single-process parallel device, so the current GPU maps are
  characterization artifacts rather than multi-GPU speedup claims,
- multiprocess execution remains workload-specific and should not be promoted
  without a measured crossover on the target machine,
- prepared geometry reuse by itself is only near parity on the committed
  fixed-geometry profile, while the compiled prepared steady path is the
  current high-leverage optimization route.

The next work is not just “more parallelism.” It is:

1. broader prepared compiled-closure reuse for large database scans,
2. repeat the production-grid and strong-scaling maps on additional dedicated
   GPU nodes with reproducibly healthy devices,
3. add device-memory timelines and larger VMEC-family workload maps,
4. and, if needed, multi-host scan orchestration.

This work belongs mainly in:

- [`src/ntx/solver.py`](../src/ntx/solver.py)
- [`src/ntx/parallel.py`](../src/ntx/parallel.py)
- [`examples/prepared_geometry_reuse_profile.py`](../examples/prepared_geometry_reuse_profile.py)
- [`scripts/benchmark_scaling.py`](../scripts/benchmark_scaling.py)
- [`scripts/benchmark_strong_scaling.py`](../scripts/benchmark_strong_scaling.py)
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

The derivative-audit and prepared implicit-adjoint milestones are now in place.
The active implementation milestone is therefore the benchmark-matrix hardening
lane:

1. keep every promoted result mapped to a script, test, artifact, and manuscript
   figure through `scripts/build_benchmark_matrix.py`,
2. promote the new VMEC geometry-family convergence stress artifact only after
   independent-code parity and radial/electric-field/collisionality ladders,
3. transfer the three-control derivative audit to reusable VMEC/Boozer
   geometry-control families and compare direct autodiff, prepared adjoints, and
   centered finite differences,
4. lift the new boundary forward-mode lane from projected geometry to a
   self-consistent equilibrium sensitivity workflow and then re-audit the same
   NTX and NTX+NEOPAX outputs,
5. define reusable hidden-symmetry and omnigenous input families before adding
   new research-grade figures,
6. and keep the fixed-field current comparison scoped to the passing
   total-current stress gate until a transferable species-resolved closure model
   also passes the integrated W7-X gate.

This is the shortest path from a strong forward solver to a research tool with
reviewable validation claims instead of isolated example scripts.

## Next Development Pass

The next code pass should execute in this order:

1. keep the CI lane manifest, source map, and benchmark matrix locked as new
   tests and ownership splits are added;
2. expand owned geometry-family benchmark artifacts only from committed
   scripts/tests/docs;
3. extend the explicit-relaxed boundary-control derivative audit to additional
   owned QA/QH/QI cases;
4. restore the implicit-equilibrium Boozer and transport derivative lane only
   after residual contraction and centered-finite-difference parity pass;
5. profile prepared-geometry reuse and closure recompiles before evaluating
   Lineax or Equinox;
6. update the manuscript figure list only from artifacts generated by these
   maintained scripts.

This order avoids two failure modes: slow CI from benchmark creep, and strong
optimization claims built on derivative paths that have not passed a local
finite-difference gate.
