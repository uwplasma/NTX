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

This script evaluates vendored archived DKES and SFINCS tables for:

- W7-X EIM
- CIEMAT-QI

and reports:

- the NTX coefficients at the chosen benchmark grid
- the archived reference coefficients
- relative errors for `D11`, `D31`, and when available `D33`

The current archived comparison is a progress-tracking report, not yet a parity
gate. The output is intended to show exactly where the dense NTX implementation
still differs from the converged benchmark curves.

## Runtime Profiling

Run:

```bash
python scripts/profile_runtime.py --output-json runtime-profile.json
```

This profiles the batched scan path against a Python loop for one DKES case and
one VMEC case, and writes:

- backend and device information
- scan compile-and-run time
- steady-state scan time
- loop time
- scan-versus-loop speedup

The current scan implementation uses a jitted batched kernel, which is the main
throughput path for parameter scans on both CPU and GPU.
