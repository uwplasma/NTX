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
