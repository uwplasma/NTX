# Performance

NTX now includes explicit scaling benchmarks and figure-generation helpers for
serial batched scans and the multiprocess throughput lane.

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
