# NTX Finalization Plan

## Goal

Ship NTX as a clean JAX-native implementation of the monoenergetic transport
formulation described in Javier Escoto's PhD thesis:
[arXiv:2510.27513](https://arxiv.org/abs/2510.27513).

## Final State

- [x] JAX-native monoenergetic solver
- [x] CLI entry point: `ntx input.toml`
- [x] DKES-style, magnetic-configuration, VMEC, and Boozer loaders
- [x] differentiable imported solve lane
- [x] direct NEOPAX scan and HDF5 mapping helpers
- [x] CPU and GPU smoke/regression scripts
- [x] package, build, and documentation scaffolding
- [x] removal of vendored benchmark families and non-NEOPAX external datasets
- [x] replacement of external repository fixtures with NTX-authored synthetic fixtures

## Current Validation Summary

- local test suite passes
- GPU smoke tests are available and skip cleanly on non-GPU machines
- office GPU hardware validation closed successfully with the NTX-owned smoke cases

## Remaining Maintenance Work

- keep synthetic loader fixtures minimal and readable
- keep NEOPAX mapping helpers aligned with the active NEOPAX interface
- continue profiling larger production grids when performance work is needed

## Next Development Lane

- [x] Add an autodiff validation example based on Escoto's formulation that solves
  a research-relevant inverse problem or sensitivity-analysis task and produces
  publication-ready figures.
- [x] Add an autodiff NEOPAX-profile example showing how NTX-generated
  monoenergetic data can be used in NEOPAX-style profile analysis with
  publication-ready figures.
- [x] Add explicit device-parallel execution for large scans across multiple CPU
  or GPU devices while preserving the differentiable imported lane.
- [x] Validate the new parallel execution path on:
  - local multi-CPU runs using forced host-device counts
  - office multi-GPU runs using the two visible accelerators
- [x] Benchmark serial versus device-parallel scan throughput and document when
  the added parallelism is actually beneficial.

## Active Work Log

- Started the post-1.0 development lane focused on autodiff demonstrations and
  explicit multi-device execution.
- The existing scan path already uses `jax.vmap` over collisionality/electric
  field. The next step is to add a device-sharded layer on top of the prepared
  solve instead of replacing the current serial JIT path.
- The first autodiff examples will stay on NTX-owned sample fixtures and
  analytic surfaces so the repository remains self-contained.
- Added the first device-parallel scan helper,
  `solve_monoenergetic_parallel_scan(...)`, using `jax.pmap` over the flattened
  scan.
- Added two autodiff helper workflows:
  - a one-parameter inverse problem on the analytic surface
  - a NEOPAX-style electric-field profile inversion on NTX-generated scan data
- Generated publication-style figures:
  - `docs/_static/autodiff_inverse_problem.png`
  - `docs/_static/autodiff_neopax_profiles.png`
- Local multi-CPU profile using
  `XLA_FLAGS=--xla_force_host_platform_device_count=4` on the sample scans:
  - DKES sample: parallel steady time `0.607 s` versus serial `0.685 s`
    (`1.13x` speedup)
  - VMEC sample: parallel steady time `0.601 s` versus serial `0.617 s`
    (`1.03x` speedup)
- Result so far: the new multi-device path is correct and already modestly
  beneficial on a forced 4-device CPU setup for the sample scans. The next real
  validation step is multi-GPU execution on office.
- office multi-GPU validation completed:
  - two GPUs are visible to JAX: `cuda:0`, `cuda:1`
  - only one GPU passes the NTX dense-solve smoke test under the current office
    software stack, so the parallel helper now excludes unhealthy devices
  - GPU sample timings with `healthy_parallel_device_count = 1`:
    - DKES sample: parallel steady `4.283 s` versus serial `3.682 s`
    - VMEC sample: parallel steady `3.236 s` versus serial `4.233 s`
- Current interpretation:
  - the parallel execution layer is now numerically guarded and safe to use
  - true multi-GPU scaling remains blocked on office by one unhealthy GPU for
    the NTX dense solve, not by incorrect NTX output on the healthy subset
- Added a separate multiprocess path,
  `solve_monoenergetic_multiprocess_scan(...)`, that assigns one worker process
  per device and pins GPU visibility with `CUDA_VISIBLE_DEVICES`.
- office root-cause result:
  - `cuda:1` is not intrinsically unhealthy
  - the failure is specific to the single-process multi-GPU cuSolver path
  - the multiprocess pinned-device lane is numerically correct on both office
    GPUs
- office multiprocess timings for the repository smoke scans:
  - DKES sample: multiprocess `44.015 s` versus serial `26.716 s`
  - VMEC sample: multiprocess `64.883 s` versus serial `5.589 s`
- Current interpretation after the multiprocess audit:
  - true two-GPU correctness is now closed
  - the repository smoke scans are too small to amortize process launch and
    per-worker compilation overhead
  - the multiprocess path is therefore a throughput option for larger scans,
    not the default path for small interactive studies
- Refined the autodiff figures and regenerated both PNG and PDF assets for:
  - `docs/_static/autodiff_inverse_problem.*`
  - `docs/_static/autodiff_neopax_profiles.*`
- Updated the NEOPAX-style autodiff inversion to fit both `D11` and `D33`
  profiles, which materially improved recovery of the target electric-field
  profile and made the example suitable for publication-facing documentation.
- Added explicit scaling benchmark and plotting workflows:
  - `scripts/benchmark_scaling.py`
  - `examples/performance_scaling.py`
- Collected smoke-grid DKES scaling data:
  - local CPU `9 x 11 x 6`: serial batched remains faster than multiprocess up
    through `256` cases
  - office GPU `9 x 11 x 6`: serial batched remains faster than multiprocess in
    the tested range after the smallest startup-dominated point
- Collected heavier-grid DKES scaling data:
  - local CPU `17 x 25 x 16`: multiprocess approaches parity by `32` cases and
    wins by `64` cases (`1.79x` speedup)
  - office GPU `17 x 25 x 16`: multiprocess remains slower than serial through
    `64` cases under the current shared-office stack
- Added publication-style performance figures and committed the JSON payloads
  used to generate them:
  - `docs/_static/performance_scaling_smoke.*`
  - `docs/_static/performance_scaling_heavy.*`
