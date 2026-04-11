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
```

It shows:

- target and recovered radial electric-field profiles
- target and recovered `D33` profiles
- objective reduction
- the local sensitivity of `D33` to the profile parameters

![Autodiff NEOPAX profiles](_static/autodiff_neopax_profiles.png)

## Parallel Execution

Large scans do not need to stay on one device. NTX now provides a
device-parallel scan helper:

```python
from ntx import solve_monoenergetic_parallel_scan
```

For local profiling:

```bash
python scripts/profile_parallel_runtime.py --output-json parallel-runtime.json
```

For multi-CPU emulation on a workstation, start the script in a fresh process
with:

```bash
XLA_FLAGS=--xla_force_host_platform_device_count=4 python scripts/profile_parallel_runtime.py
```

The same API is intended to scale to multi-GPU runs on machines with multiple
visible accelerators.
