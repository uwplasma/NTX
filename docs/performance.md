# Performance

NTX now includes explicit scaling benchmarks and figure-generation helpers for
serial batched scans and the multiprocess throughput lane. It also now includes
workflow profilers for the archive-backed fixed-field closure audit and the
corrected integrated W7-X workflow.

## Benchmark Scripts

Collect scaling data:

```bash
python scripts/benchmark_scaling.py --backend cpu --surface dkes --sizes 8,16,32,64
python scripts/benchmark_scaling.py --backend gpu --surface dkes --sizes 16,32,64 --workers 2
```

Generate publication-style figures:

```bash
python examples/performance_scaling.py \
  --cpu-json docs/_static/performance_scaling_cpu_smoke.json \
  --gpu-json docs/_static/performance_scaling_gpu_smoke.json \
  --figure-title "Smoke-grid serial vs multiprocess scaling" \
  --output-prefix docs/_static/performance_scaling_smoke
```

The example writes both PNG and PDF outputs.

Profile the corrected integrated W7-X workflow:

```bash
python scripts/profile_w7x_integrated_workflow.py \
  --output-json examples/outputs/profile_w7x_integrated_workflow/profile.json \
  --cprofile-out examples/outputs/profile_w7x_integrated_workflow/profile.pstats \
  --trace-dir examples/outputs/profile_w7x_integrated_workflow/trace
```

The script records:

- cached scan/database timings
- first-call and steady-state closure timings
- resident memory
- a Python `cProfile` dump
- a TensorFlow/JAX trace that can be opened in TensorBoard or Perfetto

## Smoke-Grid Scaling

Figure assets:

```text
docs/_static/performance_scaling_smoke.png
docs/_static/performance_scaling_smoke.pdf
```

![Smoke-grid scaling](_static/performance_scaling_smoke.png)

Interpretation:

- on the repository smoke grid `9 x 11 x 6`, serial batched JAX is the default
  choice on both CPU and GPU for small and medium scans
- the smallest GPU point is startup dominated and should not be interpreted as a
  real throughput crossover
- on the office workstation, the multiprocess lane is numerically correct but
  still slower than serial over the tested smoke-grid range

## Heavier-Grid Scaling

Figure assets:

```text
docs/_static/performance_scaling_heavy.png
docs/_static/performance_scaling_heavy.pdf
```

![Heavier-grid scaling](_static/performance_scaling_heavy.png)

Interpretation:

- on the heavier DKES grid `17 x 25 x 16`, the local 4-worker CPU multiprocess
  lane is close to the serial batched path by `32` cases and becomes faster by
  `64` cases
- on the same heavier grid, the office 2-GPU multiprocess lane remains slower
  than serial in the tested range under the current shared-office software and
  hardware stack
- the practical guidance from these measurements is:
  - use serial batched JAX for small and medium studies
  - use the multiprocess lane when the run is large enough that process startup
    is amortized, especially on CPU
  - treat office multi-GPU multiprocess execution as a robust isolation path
    first, and as a throughput path only after benchmarking the specific
    production workload

## Reproducibility

The figure JSON payloads committed in `docs/_static/` are:

- `performance_scaling_cpu_smoke.json`
- `performance_scaling_gpu_smoke.json`
- `performance_scaling_cpu_heavy.json`
- `performance_scaling_gpu_heavy.json`

They were collected on:

- local workstation CPU with `XLA_FLAGS=--xla_force_host_platform_device_count=4`
- office workstation GPU with `XLA_PYTHON_CLIENT_PREALLOCATE=false`

## Integrated W7-X Workflow

The corrected integrated W7-X raw branch is now the right profiling target
because the database normalization is closed there and the rebuilt workflow
matches the shipped reference current tightly.

Current local CPU profile, using the cached rebuilt W7-X scan:

- `reference_load_seconds`: `1.04e-2`
- `scan_prepare_seconds`: `2.94e-4`
- `rebuilt_scan_load_seconds`: `2.69e-3`
- `field_species_seconds`: `1.97`
- `database_seconds`: `2.55e-1`
- `no_momentum_first_seconds`: `8.64`
- `no_momentum_steady_seconds`: `2.63e-2`
- `momentum_correction_first_seconds`: `8.81`
- `momentum_correction_steady_seconds`: `1.58e-2`
- `current_reduction_seconds`: `3.29e-2`
- `max_rss_mb`: about `1847`

Interpretation:

- the corrected integrated workflow is compile-bound on first call, not
  arithmetic-bound
- the steady-state closure path is already fast on CPU once compiled
- the main performance priority is therefore to reduce recompiles and tracing,
  not to micro-optimize the final current reduction

The current `cProfile` dump is dominated by XLA compilation:

- about `15 s` in `backend_compile_and_load`
- about `20 s` total Python runtime

That points directly to the next speed lane:

- stabilize shapes and dtypes in the closure path
- hoist and reuse the compiled no-momentum and momentum-correction calls
- avoid retracing/vmap rebuilding across repeated workflow invocations
- then revisit deeper kernel/vectorization work only after those compile
  overheads are under control
