# Validation

NTX ships with three complementary validation paths:

1. unit and regression tests in `tests/`
2. GPU smoke and regression checks through `tests/test_gpu_smoke.py` and
   `scripts/run_gpu_regression.py`
3. archived cross-code comparison scripts against DKES and SFINCS benchmark
   tables

## VMEC Validation

The VMEC path is covered by:

- loader and normalization tests in `tests/test_vmec.py`
- physics checks in `tests/test_vmec_physics.py`
- QI-specific checks in `tests/test_vmec_qi.py`
- regression snapshots in `tests/test_vmec_regression.py`
- scan coverage in `tests/test_vmec_scan.py`

The scan tests cover both:

- loop-versus-scan agreement
- `er_hat` versus explicit `epsi_hat` agreement on VMEC surfaces

## Archived Cross-Code Comparisons

Run:

```bash
python scripts/compare_archived_benchmarks.py --output-json archived-benchmarks.json
```

Or restrict the report to one benchmark family:

```bash
python scripts/compare_archived_benchmarks.py --case W7X-EIM
python scripts/compare_archived_benchmarks.py --case W7X-KJM
python scripts/compare_archived_benchmarks.py --case CIEMAT-QI
```

This script defaults to `JAX_PLATFORM_NAME=cpu` so the archived comparison does
not depend on accelerator FFT behavior.

This script evaluates vendored archived thesis benchmark tables for:

- W7-X EIM
- W7-X KJM
- CIEMAT-QI

and reports:

- the NTX coefficients at the chosen benchmark grid
- archived DKES and SFINCS coefficients
- archived monoenergetic reference coefficients when an exact grid-matched
  reference is vendored
- relative errors for `D11`, `D31`, and when available `D33`

Current interpretation:

- W7-X EIM: NTX matches the archived monoenergetic reference at the
  `23 x 55 x 80` grid used by the thesis convergence study, while still showing
  the expected spread against the archived DKES and SFINCS curves.
- W7-X KJM: NTX matches the archived monoenergetic reference at the
  `19 x 79 x 180` benchmark grid, again with visible cross-code spread against
  DKES and SFINCS.
- CIEMAT-QI: the archived DKES, SFINCS, and monoenergetic tables are vendored
  and parsed, but the exact `47 x 215 x 160` NTX solve is substantially heavier
  than the W7-X cases, so it is best run selectively with `--case CIEMAT-QI`
  rather than treated as a default smoke check.

## Runtime Profiling

Run:

```bash
python scripts/profile_runtime.py --output-json runtime-profile.json
```

This script also defaults to `JAX_PLATFORM_NAME=cpu`. Override the environment
when you want a GPU-specific profile.

This profiles the batched scan path against a Python loop for one DKES case and
one VMEC case, and writes:

- backend and device information
- scan compile-and-run time
- steady-state scan time
- loop time
- scan-versus-loop speedup

The current scan implementation uses a jitted batched kernel, which is the main
throughput path for parameter scans on both CPU and GPU.
