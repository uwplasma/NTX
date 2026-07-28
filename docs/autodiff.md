# Autodiff

NTX keeps the imported solve lane differentiable so transport coefficients can
be embedded in inverse problems, sensitivity analysis, and profile workflows.

## Bounding the reverse pass

Reverse-mode differentiation of the monoenergetic solve used to record the whole
Legendre elimination, so the memory of a gradient grew with `n_xi` even though
the forward solve's did not. It no longer does. The rows are handed to the solver
as an explicit parameter set (`ntx.operators.block_parameters`) and regenerated on
demand in both directions, so the reverse pass retains a window of rows rather
than a tape of the sweep.

The default retains every row. That is exact — it returns the same gradient the
tape did, to rounding — and it is already cheaper, so nothing needs changing to
benefit:

```python
grad = jax.grad(lambda nu: ntx.solve_monoenergetic(surface, grid, case(nu)).D11)(nu0)
```

A finite window retains `3 + adjoint_window` rows instead, and then the reverse
pass stops growing with the Legendre resolution:

```python
result = ntx.solve_monoenergetic(surface, grid, case, adjoint_window=33)
```

Where to put it is a property of the collision physics, not of the grid.
Pitch-angle scattering damps mode `l` like `nu*l(l+1)` while the streaming
coupling grows only like `l`, so the chain contracts faster the higher one
climbs, and the row where it starts contracting moves outward as collisions
weaken. `advise_adjoint_window` reads that crossover off the operator:

```python
advice = ntx.advise_adjoint_window(ctx, grid.n_xi, prepared.d_theta, prepared.d_zeta)
advice.window          # a starting point
advice.crossover_row   # where the transfer norms fall below one
advice.certified       # always False: this is an estimate, not a guarantee
```

Treat it as an initializer and widen it until the gradient stops moving. On the
example surface at `nu_hat = 1e-2`, `m = 81`:

| `n_xi` | taped | full window (exact) | advised window (33) |
|---|---|---|---|
| 32 | 10.7 MiB, 17.8 ms | 8.7 MiB, 12.6 ms | 8.3 MiB, 13.5 ms |
| 64 | 20.3 MiB, 51.8 ms | 16.8 MiB, 17.2 ms | 9.4 MiB, 18.4 ms |
| 128 | 39.6 MiB, 83.3 ms | 33.2 MiB, 38.1 ms | 9.4 MiB, 25.5 ms |
| 256 | 78.1 MiB, 125.9 ms | 66.1 MiB, 98.8 ms | 9.4 MiB, 60.9 ms |

The full-window gradient agrees with the taped one to between 1.3e-14 and
4.1e-14; the advised window costs 6.1e-07 and its memory is flat. Regenerate the
table with `benchmarks/bench_exact_window_adjoint.py`.

The forward solve is untouched in every case, and so are the transport
coefficients: a window bounds the derivative, not the answer.


## Inverse Problem Example

The script:

```bash
python examples/autodiff_inverse_problem.py
```

solves a small synthetic inverse problem on the analytic sample surface. One
Fourier amplitude is treated as an unknown parameter, synthetic `D11`
observations are generated from a target surface, and JAX gradients are used to
recover the amplitude.

The figure is written to:

```text
docs/_static/autodiff_inverse_problem.png
docs/_static/autodiff_inverse_problem.pdf
```

It shows:

- parameter convergence
- objective reduction
- recovered transport response against the target response

![Autodiff inverse problem](_static/autodiff_inverse_problem.png)

## Derivative Audit

The script:

```bash
python examples/derivative_audit.py
```

compares direct JAX gradients of the dense monoenergetic solve against centered
finite differences for two practically important controls:

- a Boozer harmonic amplitude at fixed electric field,
- and the radial electric field at fixed collisionality.

The example does not rely on one hidden helper. It walks through the explicit
prepared-solver workflow:

```python
from ntx import (
    GridSpec,
    MonoenergeticCase,
    example_surface,
    prepare_monoenergetic_system,
    solve_prepared_coefficient_vector,
    solve_prepared_coefficient_vector_vjp,
)
```

That is the contract point for the prepared implicit-adjoint derivative path:
the forward solve remains the same, while the backward rule stays isolated from
user-facing optimization scripts.

The low-level operator derivative used by this path is now test-gated directly:
`tests/test_operators.py` requires the hand-coded `dD_k/dnu_hat` and
`dD_k/depsi_hat` blocks to match JAX differentiation of the assembled
Legendre-space operator. This keeps the collisionality and radial-electric-field
normalizations tied to the implemented equations rather than to a downstream
finite-difference fit.

The figure is written to:

```text
docs/_static/derivative_audit.png
docs/_static/derivative_audit.pdf
```

It shows:

- gradient magnitude across collisionality for `D11` and `D33`,
- relative mismatch between autodiff and finite differences,
- electric-field sensitivities across `\hat E_r`,
- and the current numerical agreement used to validate the prepared
  implicit-adjoint path.

![Derivative audit](_static/derivative_audit.png)

## Prepared-Derivative Benchmark

For a parameter-dependent discrete drift-kinetic system

```{math}
A(p)f(p)=s(p),
```

the tangent equation is

```{math}
A f_p = s_p-A_p f.
```

For a scalar coefficient or objective `J(f,p)`, NTX instead reuses the primal
block factors to solve

```{math}
A^T\lambda=J_f,
\qquad
\frac{\mathrm dJ}{\mathrm dp}
=J_p+\lambda^T(s_p-A_p f).
```

The transpose action is algebraic: it is reconstructed independently from the
conditioned Legendre blocks, including the nullspace row, rather than inferred
from agreement with another differentiated program. This follows the standard
discrete-adjoint structure used for neoclassical optimization in Paul et al.
(2019).

Use the opt-in validity audit before promoting a scalar prepared derivative:

```python
from ntx import audit_prepared_coefficient_derivative

audit = audit_prepared_coefficient_derivative(
    prepared,
    case,
    coefficient="D11",
    parameter="er_hat",
)
if not bool(audit.valid):
    raise RuntimeError(audit.as_dict())
```

`valid` requires all of the following:

- finite primal, adjoint, and derivative values;
- full-system `||Af-s||/||s||` and `||A^T lambda-g||/||g||` below the requested
  residual tolerance;
- prepared-adjoint, forward-mode, and centered-finite-difference gradients
  within the requested tolerance of direct reverse mode.

The default centered-difference step is
`eps**(1/3) * max(abs(p), 1)`, which balances second-order truncation error
against floating-point cancellation. It is an independent diagnostic, not the
runtime derivative implementation.

The script:

```bash
python examples/derivative_path_benchmark.py
```

keeps the same prepared surface and coefficient derivatives, then compares:

- direct reverse-mode through `solve_prepared_coefficient_vector(...)`,
- selective recomputation through `jax.checkpoint(...)`,
- the factor-reusing custom VJP through
  `solve_prepared_coefficient_vector_vjp(...)`,
- forward mode,
- and centered finite differences.

The example is intentionally explicit. It shows how to:

- prepare a reusable system with `prepare_monoenergetic_system(...)`,
- define scalar coefficient objectives,
- wrap them with `jax.grad(...)` and `jax.vmap(...)`,
- JIT the resulting scan kernels,
- compare synchronized timing, XLA temporary memory, and agreement on the same
  `\hat E_r` scan.

The figure is written to:

```text
docs/_static/derivative_path_benchmark.png
docs/_static/derivative_path_benchmark.pdf
```

It shows:

- best-of-three wall times versus scan size,
- speedup and direct/prepared agreement,
- independent primal and transpose residuals across collisionality,
- and direct-reverse, prepared-adjoint, forward, and finite-difference
  agreement.

On the committed CPU artifact at scan size 32, the prepared adjoint reduces
XLA temporary memory from about `49.2 MiB` to `13.2 MiB` and is about `2.6x`
faster than direct reverse mode. Selective recomputation does not reduce memory
for this block solve, so it is measured but not enabled. These timings are
machine-specific evidence, not release gates.

The lowest-collisionality audit point is intentionally marked invalid: its
gradient methods agree, but its independently reconstructed primal residual is
above `1e-10`. This prevents accidental promotion of an unconverged derivative.
The separate non-degenerate `dD33/dnu_hat` resolution ladder passes all
primal/transpose and method-agreement gates and changes by `8.0%` and `5.4%`
over two successive `9 x 11 x 8 -> 11 x 13 x 10 -> 13 x 15 x 12`
refinements. The near-zero `dD11/dEr` observable is not used to claim resolution
convergence.

The JSON sidecar is now checked by the physics-gate registry. The promoted
release claim is derivative agreement, not benchmark-machine timing: the maximum
prepared-vs-direct relative mismatch must remain below `1e-4`; the reported
speedup is retained as performance evidence.

![Prepared derivative benchmark](_static/derivative_path_benchmark.png)

## Geometry-Control Derivative Benchmark

The script:

```bash
python examples/geometry_control_derivative_benchmark.py
```

extends the derivative checks from one scalar control to three independent
Boozer-harmonic amplitudes on the owned analytic surface. It compares direct
JAX geometry-control derivatives against centered finite differences for
`D11`, `D31`, and `D33` across collisionality.

The figure is written to:

```text
docs/_static/geometry_control_derivative_benchmark.png
docs/_static/geometry_control_derivative_benchmark.pdf
docs/_static/geometry_control_derivative_benchmark.json
```

This is an artifact-backed autodiff stress benchmark. It is not yet a
large-geometry-control validation claim; the non-promoted follow-up is to
transfer the same audit to reusable VMEC/Boozer geometry-control families and to
compare geometry pullbacks with a prepared implicit-adjoint path once that
pullback exists.
The JSON sidecar is checked by the physics-gate registry with a current
acceptance threshold of `2e-4` maximum relative direct-AD/finite-difference
mismatch on this owned analytic surface.

![Geometry-control derivative benchmark](_static/geometry_control_derivative_benchmark.png)

## File-Backed Geometry-Control Benchmark

The script:

```bash
python examples/file_backed_geometry_control_derivative_benchmark.py
```

takes the next step on the geometry-control autodiff lane. Instead of the owned
analytic surface, it loads two repository-owned file-backed cases:

- a Boozer-file sample surface,
- and a VMEC-backed sample surface.

For each case, NTX selects the dominant non-axisymmetric harmonics, perturbs
them through dimensionless scale factors, and compares direct JAX derivatives
against centered finite differences for `D11`, `D31`, and `D33`.

The figure is written to:

```text
docs/_static/file_backed_geometry_control_derivative_benchmark.png
docs/_static/file_backed_geometry_control_derivative_benchmark.pdf
docs/_static/file_backed_geometry_control_derivative_benchmark.json
```

This closes the first file-backed slice of the geometry-control derivative
path: the derivative audit now transfers from an owned analytic surface to
repository-owned file-backed magnetic geometry. It is still a stress benchmark
rather than a promoted design claim, since the non-promoted follow-up is a
reusable family of VMEC/Boozer controls plus prepared implicit-adjoint geometry
pullbacks.
The committed JSON sidecar is now a physics gate with a `5e-4` maximum relative
direct-AD/finite-difference mismatch threshold on the file-backed samples.

![File-backed geometry-control derivative benchmark](_static/file_backed_geometry_control_derivative_benchmark.png)

## Boundary Forward-Mode Benchmark

The script:

```bash
python examples/boundary_forward_mode_current_derivative_benchmark.py
```

checks the next imported differentiable lane built on the upstream
`vmex` and `booz_xform_jax` packages. It treats two low-order boundary
controls from the repository-owned sample input as independent variables,
builds the boundary-projected VMEC state, transforms it to Boozer
coordinates, and then differentiates two scalar outputs with respect to those
controls:

- an NTX monoenergetic transport response,
- and an NTX+NEOPAX integrated-current objective.

The figure is written to:

```text
docs/_static/boundary_forward_mode_current_derivative_benchmark.png
docs/_static/boundary_forward_mode_current_derivative_benchmark.pdf
docs/_static/boundary_forward_mode_current_derivative_benchmark.json
```

This is an artifact-backed stress benchmark for the boundary-to-output lane.
It is intentionally scoped to the boundary-projected geometry map, where
forward-mode autodiff matches centered finite differences on the committed
sample case. It does not yet claim a fully validated self-consistent
equilibrium sensitivity workflow for bootstrap current.
The committed JSON sidecar is checked with a `1e-5` maximum relative
forward-mode/finite-difference mismatch threshold.

![Boundary forward-mode current derivative benchmark](_static/boundary_forward_mode_current_derivative_benchmark.png)

## Implicit Equilibrium Forward-Mode Benchmark

The script:

```bash
python examples/implicit_equilibrium_forward_mode_derivative_benchmark.py
```

adds the next implicit-equilibrium diagnostic on the committed QA case. It uses
the same low-order boundary controls, but now routes them through the implicit
fixed-boundary `vmex` residual solve with
`residual_tangent_mode="auto"`. The benchmark then differentiates three scalar
outputs with respect to those controls:

- equilibrium volume,
- a Boozer-space scalar built from the implicit equilibrium,
- and an NTX monoenergetic transport response.

The figure is written to:

```text
docs/_static/implicit_equilibrium_forward_mode_derivative_benchmark.png
docs/_static/implicit_equilibrium_forward_mode_derivative_benchmark.pdf
docs/_static/implicit_equilibrium_forward_mode_derivative_benchmark.json
```

This closes the implicit-equilibrium lane as a non-shipping diagnostic, not as a
supported optimization path. The current JSON artifact shows a mixed result on
the committed QA case:

- the equilibrium-volume derivative matches centered finite differences,
- the Boozer scalar fails tangent parity on the implicit lane,
- the NTX transport observable fails more strongly on the same lane,
- the residual history does not contract under the committed iteration ladder,
- and the matching reverse-mode Boozer-scalar diagnostic is unavailable because
  the dynamic-loop implicit solve is not a valid promoted reverse-mode path.

The JSON sidecar is intentionally registered as a monitored diagnostic, not an
acceptance gate. The supported self-consistent equilibrium derivative route is
the explicit-relaxed fixed-boundary lane below. The implicit lane should only be
restored after the backend residual solve contracts and Boozer/NTX centered-FD
tangent parity passes.

![Implicit-equilibrium forward-mode derivative benchmark](_static/implicit_equilibrium_forward_mode_derivative_benchmark.png)

## Explicit-Relaxed Equilibrium Benchmark

The script:

```bash
python examples/explicit_relaxed_boundary_current_derivative_benchmark.py
```

closes the next imported lane on two repository-owned non-axisymmetric cases:
a low-resolution QA family input and a lighter QH warm-start input. It uses
the same low-order boundary controls, but instead of stopping at the
boundary-projected VMEC state it runs an explicitly relaxed fixed-boundary
`vmex` solve in a stable forward-mode regime and then differentiates three
scalar outputs on each case:

- a Boozer-space scalar built from the relaxed surface,
- an NTX monoenergetic transport response,
- and an `NTX+NEOPAX` integrated-current objective.

The figure is written to:

```text
docs/_static/explicit_relaxed_boundary_current_derivative_benchmark.png
docs/_static/explicit_relaxed_boundary_current_derivative_benchmark.pdf
docs/_static/explicit_relaxed_boundary_current_derivative_benchmark.json
```

This is the first committed self-consistent boundary-to-current forward-mode
benchmark family. The JSON artifact records that the ordinary and explicit
relaxed primal volumes agree on both committed cases, so the benchmark is not
just an internally consistent autodiff loop on a different equilibrium branch.
The non-promoted follow-up is now narrower:

- widen from the committed QA/QH cases to additional geometry families,
- add integrated-current objectives on the supported explicit-relaxed lane,
- and repair reverse mode on the relaxed-equilibrium lane.

The committed JSON sidecar is now checked by the physics-gate registry with a
`1e-4` maximum relative forward-mode/finite-difference mismatch threshold. The
artifact also reports the ordinary-vs-explicit-relaxed volume difference, which
is currently zero on the committed cases.

At the moment, the reverse-mode implicit diagnostic remains non-shipping for a
concrete reason: the matching QA Boozer-scalar probe is unavailable or guarded
to zero while centered finite differences are nonzero, so it is not a promoted
sensitivity workflow.

![Explicit-relaxed boundary current derivative benchmark](_static/explicit_relaxed_boundary_current_derivative_benchmark.png)

## Geometry-Family Breadth Summary

The script:

```bash
python examples/geometry_family_breadth_summary.py
```

does not rerun expensive equilibrium solves. It reads the committed derivative
artifacts and summarizes the current geometry-breadth status in one
publication-ready figure:

- analytic geometry-control derivatives,
- file-backed Boozer and VMEC geometry-control derivatives,
- boundary-projected current derivatives,
- explicit-relaxed QA/QH boundary-to-current derivatives,
- and the implicit-equilibrium diagnostic split into the validated volume
  objective and the retired non-shipping Boozer/NTX transport diagnostics.

The figure is written to:

```text
docs/_static/geometry_family_breadth_summary.png
docs/_static/geometry_family_breadth_summary.pdf
docs/_static/geometry_family_breadth_summary.json
```

This closes the artifact-backed geometry-breadth summary lane, not the full
geometry-family validation lane. The remaining promotion requirements are
explicit in the JSON sidecar: broader W7-X/QI/omnigenous inputs, direct
`D11/D31/D33` parity and convergence ladders, and implicit Boozer/transport
derivative parity.

![Geometry-family breadth summary](_static/geometry_family_breadth_summary.png)

## Geometry-Family Transport Convergence

The script:

```bash
python examples/geometry_family_transport_convergence.py --preset production
```

discovers reusable VMEC `wout` examples from local `vmex`, STELLOPT, and
SIMSOPT checkouts, loads each surface through the NTX VMEC path, and runs a
production `D11/D31/D33` grid ladder. The JSON also stores `D13` and the
normalized Onsager residual so coefficient convergence and reciprocity quality
are audited together. The figure and JSON sidecar are written to:

```text
docs/_static/geometry_family_transport_convergence.png
docs/_static/geometry_family_transport_convergence.pdf
docs/_static/geometry_family_transport_convergence.json
```

The current artifact is a convergence stress diagnostic across the available
public geometry families. It distinguishes cases that are below the tracked
stress tolerance from cases that need profile-ladder or independent-reference
promotion work.

![Geometry-family transport convergence](_static/geometry_family_transport_convergence.png)

## NEOPAX-Style Profile Example

The script:

```bash
python examples/neopax_autodiff_profiles.py
```

builds a small NTX scan, maps it into the NEOPAX monoenergetic data layout, and
then solves a low-dimensional electric-field profile inversion using autodiff.

The figure is written to:

```text
docs/_static/autodiff_neopax_profiles.png
docs/_static/autodiff_neopax_profiles.pdf
```

It shows:

- target and recovered radial electric-field profiles
- target and recovered `D33` profiles
- objective reduction
- the local sensitivity of `D33` to the profile parameters

The fast test suite now gates this interpolation layer directly:
`tests/test_autodiff.py` checks that `D33` sensitivities through the
electric-field profile basis agree with centered finite differences on a
controlled coefficient table. This keeps the profile inverse-design and
uncertainty examples tied to a checked differentiable map instead of relying
only on end-to-end objective reduction.

Large profile bases may set `jacobian_chunk_size="auto"` or a positive integer
in `example_neopax_profile_autodiff(...)` and
`example_neopax_profile_uncertainty(...)`. NTX then delegates bounded-memory
Jacobian assembly to SOLVAX and selects forward or reverse mode from the input
and output dimensions. The default remains native `jax.jacrev` because chunking
adds overhead and did not reduce memory for the small committed profile basis.
On a separate 16-control, 96-output high-mode QH performance diagnostic,
forward chunks of four reduced XLA temporary memory by about `3.6x` with a
roughly `20%` warm-runtime cost; chunks of one reduced memory by about `10x`
but approximately doubled runtime. This is execution guidance, not a promoted
physics artifact.

![Autodiff NEOPAX profiles](_static/autodiff_neopax_profiles.png)

## Profile Uncertainty Audit

The script:

```bash
python examples/autodiff_profile_uncertainty.py
```

uses the same differentiable NEOPAX-style profile fit, then compares two
uncertainty-propagation paths for the recovered `D33(\rho)` profile under a
small prescribed Gaussian uncertainty on the fitted radial electric-field basis
parameters:

- a linearized covariance propagation through the sensitivity matrix,
- and a small Monte Carlo ensemble in the fitted profile-parameter space.

The committed artifact uses a three-term odd-power radial basis by default and
also records a local Fisher/Gauss-Newton matrix plus a Hessian-vector-product
probe for the same combined `D11`/`D33` residual used by the fit. The HVP probe
is evaluated at the recovered profile parameters, where the residual term
vanishes, so it should agree with the Fisher/Gauss-Newton product. This
provides a local mathematical gate for profile-UQ derivatives without promoting
the synthetic profile family to a broad design claim.

The figure is written to:

```text
docs/_static/autodiff_profile_uncertainty.png
docs/_static/autodiff_profile_uncertainty.pdf
docs/_static/autodiff_profile_uncertainty.json
```

It shows:

- the fitted transport profile with propagated uncertainty bands,
- linearized versus Monte Carlo standard deviations,
- the fitted profile-parameter correlation matrix,
- the relative mismatch between the two uncertainty paths,
- and the Fisher/HVP consistency metrics stored in the JSON artifact.

This is the current artifact-backed uncertainty-propagation benchmark for the
autodiff lane. It is intentionally synthetic and is tracked as a monitored
stress benchmark rather than a parity gate, but it exercises the same
differentiable profile map used in inverse-design and profile-control studies.
Both the profile sensitivity and Fisher/Gauss-Newton residual Jacobians honor
the optional `jacobian_chunk_size` setting.

![Autodiff profile uncertainty](_static/autodiff_profile_uncertainty.png)

## Robust Bootstrap-Current Optimization

The script:

```bash
python examples/bootstrap_current_robust_optimization.py
```

adds a prescribed Gaussian uncertainty on the scalar geometry control used by
the bootstrap-current response optimization and compares:

- the deterministic objective landscape,
- the robust mean-minus-risk objective,
- the optimized nominal current profile,
- and the uncertainty band of that profile under the prescribed control
  perturbation.

The figure is written to:

```text
docs/_static/bootstrap_current_robust_optimization.png
docs/_static/bootstrap_current_robust_optimization.pdf
docs/_static/bootstrap_current_robust_optimization.json
```

This is a synthetic robust-design benchmark anchored to the same differentiable
current-response workflow as the main optimization example. It is currently a
tracked stress diagnostic, not a literature-grade validation claim. The JSON artifact
separates `robust_objective_relative_change`, which gates the optimization
workflow, from `weighted_current_ratio`, which is a signed current-profile
diagnostic and should not be interpreted as a standalone parity claim.

## Parallel Execution

Large scans do not need to stay on one device. NTX currently exposes two
parallel paths:

```python
from ntx import solve_monoenergetic_parallel_scan
from ntx import solve_monoenergetic_multiprocess_scan
```

`solve_monoenergetic_parallel_scan(...)` keeps execution inside one Python
process and is the lightest-weight option when all visible devices are healthy.
`solve_monoenergetic_multiprocess_scan(...)` runs one worker process per device
and is the robust option when the platform shows process-local solver behavior.

For local profiling:

```bash
python scripts/profile_parallel_runtime.py --output-json parallel-runtime.json
python scripts/profile_multiprocess_runtime.py --backend cpu --workers 2
```

For multi-CPU emulation on a workstation, start the script in a fresh process
with:

```bash
XLA_FLAGS=--xla_force_host_platform_device_count=4 python scripts/profile_parallel_runtime.py
```

On the office workstation, the single-process path exposes a cuSolver failure
mode on `cuda:1`, while the multiprocess pinned-device path is numerically
correct on both GPUs. For the repository smoke cases the multiprocess path is
still slower than the serial batched solve because worker startup dominates, so
it should be treated as a throughput lane for larger scans rather than a
default for small studies.
