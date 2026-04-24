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
- on the refreshed local CPU run, the multiprocess and single-process
  device-parallel lanes are numerically correct but still slower than serial
  over the tested smoke-grid range
- the refreshed CPU smoke artifact reports process peak resident memory of
  about `1.76 GB`
- the refreshed office GPU smoke artifact reports process peak resident memory
  of about `1.29 GB`, with one of two GPUs passing the single-process
  device-parallel smoke filter

## Heavier-Grid Scaling

Figure assets:

```text
docs/_static/performance_scaling_heavy.png
docs/_static/performance_scaling_heavy.pdf
```

![Heavier-grid scaling](_static/performance_scaling_heavy.png)

Interpretation:

- on the heavier DKES grid `17 x 25 x 16`, the refreshed local CPU artifact
  shows the single-process device-parallel lane crossing serial by `32` cases,
  while the 4-worker CPU multiprocess lane remains slower through `64` cases
- on the same heavier grid, the office 2-GPU multiprocess lane remains slower
  than serial in the tested range under the current shared-office software and
  hardware stack
- the refreshed CPU heavy artifact reports process peak resident memory of
  about `2.70 GB`
- the refreshed office GPU heavy artifact reports process peak resident memory
  of about `1.42 GB`, again with one healthy single-process device
- the practical guidance from these measurements is:
  - use serial batched JAX for small and medium studies
  - use the single-process device-parallel lane on CPU only after checking that
    the target grid/scan size has crossed over
  - use the multiprocess lane only when a measured workload shows enough
    amortization of process startup on the target machine
  - treat office multi-GPU multiprocess execution as a robust isolation path
    first, and as a throughput path only after benchmarking the specific
    production workload

## Reproducibility

The figure JSON payloads committed in `docs/_static/` are:

- `performance_scaling_cpu_smoke.json`
- `performance_scaling_gpu_smoke.json`
- `performance_scaling_cpu_heavy.json`
- `performance_scaling_gpu_heavy.json`

Fresh runs of `scripts/benchmark_scaling.py` and
`scripts/profile_parallel_runtime.py` also record process peak resident memory
as `max_rss_mb`. That value is intentionally treated as a run-environment
metric rather than a parity target, but it keeps memory visible whenever timing
artifacts are regenerated. The committed CPU artifacts were refreshed locally;
the committed GPU artifacts were refreshed from a clean temporary checkout on
the office GPU workstation.

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

A simple persistent compilation-cache experiment is now also bounded out as a
first-order fix. Re-running the same workflow in a fresh process with
`--compilation-cache-dir` enabled leaves the first-call latencies essentially
unchanged:

- cold cached process:
  - `no_momentum_first_seconds`: `1.17e+1`
  - `momentum_correction_first_seconds`: `1.24e+1`
- warm cached process:
  - `no_momentum_first_seconds`: `1.17e+1`
  - `momentum_correction_first_seconds`: `1.23e+1`

So the current integrated workflow is not being held back by a missing on-disk
compilation cache alone. The speed lane should stay focused on shape
stability, static-argument control, and reusable compiled closure calls rather
than on cache toggles by themselves.

## Research-Grade Performance Plan

The next performance work should stay evidence-driven:

1. measure compile time, first-call time, steady-state time, peak resident
   memory, and device memory separately;
2. keep small PR tests and large profiling campaigns separate;
3. profile the exact workload before changing linear algebra, vectorization, or
   dependencies;
4. prefer stable shapes and prepared data structures over dynamic Python control
   inside `jit`;
5. promote multi-process or multi-device paths only when a measured production
   grid crosses over from serial batched JAX.

JAX-specific rules for NTX:

- use `jax.vmap` for independent collisionality, electric-field, species, or
  radial scan axes when all mapped leaves have compatible shapes;
- use `jax.lax.scan` for fixed-length iterative loops that would otherwise be
  unrolled inside `jit`;
- keep static arguments hashable, immutable, and low-cardinality so they do not
  create unnecessary recompiles;
- consider buffer donation only at public call boundaries where the caller will
  not reuse the donated arrays;
- use `jax.profiler.trace` or XProf/Perfetto for targeted traces, and JAX memory
  profiling for OOM or retained-buffer investigations;
- for GPU sharing, set explicit memory policy such as
  `XLA_PYTHON_CLIENT_PREALLOCATE=false` or `XLA_PYTHON_CLIENT_MEM_FRACTION`
  before launching concurrent runs.

Lineax and Equinox are useful but not automatic wins:

- Lineax should be evaluated first on repeated structured solve or
  Jacobian-linear-operator workloads where reuse or memory reduction can be
  measured against the current prepared dense solve.
- Equinox should be evaluated for typed PyTree modules and filtered transforms
  only if it simplifies static-versus-dynamic argument handling or custom
  derivative APIs without destabilizing the public NTX API.

Do not use broad XLA dump passes as the default profiling loop on normal
workstations. They are useful for focused compiler investigations, but the
current project bottlenecks are better attacked with smaller traces, shape
audits, and cached closure-only profiling.
