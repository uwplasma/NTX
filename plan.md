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

- [ ] Add an autodiff validation example based on Escoto's formulation that solves
  a research-relevant inverse problem or sensitivity-analysis task and produces
  publication-ready figures.
- [ ] Add an autodiff NEOPAX-profile example showing how NTX-generated
  monoenergetic data can be used in NEOPAX-style profile analysis with
  publication-ready figures.
- [ ] Add explicit device-parallel execution for large scans across multiple CPU
  or GPU devices while preserving the differentiable imported lane.
- [ ] Validate the new parallel execution path on:
  - local multi-CPU runs using forced host-device counts
  - office multi-GPU runs using the two visible accelerators
- [ ] Benchmark serial versus device-parallel scan throughput and document when
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
