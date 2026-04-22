# Autodiff

NTX keeps the imported solve lane differentiable so transport coefficients can
be embedded in inverse problems, sensitivity analysis, and profile workflows.

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

The script:

```bash
python examples/derivative_path_benchmark.py
```

keeps the same prepared surface and the same `D33` electric-field derivative,
then times two user-visible paths:

- direct reverse-mode through `solve_prepared_coefficient_vector(...)`,
- and the prepared custom-VJP path through
  `solve_prepared_coefficient_vector_vjp(...)`.

The example is intentionally explicit. It shows how to:

- prepare a reusable system with `prepare_monoenergetic_system(...)`,
- define scalar coefficient objectives,
- wrap them with `jax.grad(...)` and `jax.vmap(...)`,
- JIT the resulting scan kernels,
- and compare timing and agreement on the same `\hat E_r` scan.

The figure is written to:

```text
docs/_static/derivative_path_benchmark.png
docs/_static/derivative_path_benchmark.pdf
```

It shows:

- best-of-three wall times versus scan size,
- speedup of the prepared custom-VJP path,
- and the max relative mismatch between the two derivative paths.

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

![Autodiff NEOPAX profiles](_static/autodiff_neopax_profiles.png)

## Profile Uncertainty Audit

The script:

```bash
python examples/autodiff_profile_uncertainty.py
```

uses the same differentiable NEOPAX-style profile fit, then compares two
uncertainty-propagation paths for the recovered `D33(\rho)` profile under a
small prescribed Gaussian uncertainty on the fitted profile parameters:

- a linearized covariance propagation through the sensitivity matrix,
- and a small Monte Carlo ensemble in the fitted profile-parameter space.

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
- and the relative mismatch between the two uncertainty paths.

This is the current artifact-backed uncertainty-propagation benchmark for the
autodiff lane. It is intentionally synthetic and is tracked as a monitored
stress benchmark rather than a parity gate, but it exercises the same
differentiable profile map used in inverse-design and profile-control studies.

![Autodiff profile uncertainty](_static/autodiff_profile_uncertainty.png)

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
