# NTX Research Plan

## Goal

Operate NTX as a research-grade JAX-native implementation of the
monoenergetic transport formulation described in Javier Escoto's PhD thesis:
[arXiv:2510.27513](https://arxiv.org/abs/2510.27513).

## Shipped Base State

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

## Research-Grade Open Lanes

- [ ] optimization-grade dense-solve derivatives:
  - add an implicit or adjoint derivative path for the prepared monoenergetic
    solve
  - validate it against direct autodiff and finite differences
  - benchmark gradient cost and memory against current reverse-mode JAX
- [ ] profile-grade transport workflows:
  - promote ambipolar `E_r(r)` and bootstrap-current self-consistency to
    first-class imported APIs
  - keep the NEOPAX coupling layer clean and differentiable
- [ ] broaden geometry studies:
  - support hidden-symmetry, omnigenous, and piecewise-omnigenous research
    campaigns with in-memory geometry perturbations
- [ ] production throughput:
  - improve prepared-geometry reuse and large scan throughput
  - benchmark CPU, GPU, and multiprocess crossover points on production grids
- [ ] physics expansion:
  - stage momentum-restoring and higher-level transport closures after the
    derivative and profile lanes are stable

## Research References

- Javier Escoto thesis:
  [arXiv:2510.27513](https://arxiv.org/abs/2510.27513)
- adjoint neoclassical optimization:
  [arXiv:1904.06430](https://arxiv.org/abs/1904.06430)
- differentiable programming for plasma workflows:
  [arXiv:2410.11161](https://arxiv.org/abs/2410.11161)
- zero-bootstrap-current piecewise omnigenity:
  [arXiv:2505.02546](https://arxiv.org/abs/2505.02546)
- hidden-symmetry optimization:
  [arXiv:2502.09350](https://arxiv.org/abs/2502.09350)
- combined omnigenity and piecewise omnigenity:
  [arXiv:2603.12139](https://arxiv.org/abs/2603.12139)

## Remaining Maintenance Work

- keep synthetic loader fixtures minimal and readable
- keep NEOPAX mapping helpers aligned with the active NEOPAX interface
- continue profiling larger production grids when performance work is needed
- keep the documentation synchronized with the shipped solver interfaces and
  publication-figure scripts

## Started Research Lane

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
- [x] Add a manuscript-ready validation summary figure and a one-command figure
  bundle generator so the publication assets can be regenerated directly from
  the repository.
- [x] Add a derivative-audit workflow that compares direct JAX gradients against
  centered finite differences before introducing an implicit/adjoint derivative
  path.
- [ ] Implement the first implicit-derivative prototype for the prepared dense
  solve and benchmark it against the current direct-autodiff path.

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
  - local CPU `17 x 25 x 16`: multiprocess is close to the serial batched path
    by `32` cases and wins by `64` cases (`1.79x` speedup)
  - office GPU `17 x 25 x 16`: multiprocess remains slower than serial through
    `64` cases under the current shared-office stack
- Added publication-style performance figures and committed the JSON payloads
  used to generate them:
  - `docs/_static/performance_scaling_smoke.*`
  - `docs/_static/performance_scaling_heavy.*`
- Added a manuscript-facing validation figure:
  - `examples/validation_summary.py`
  - `docs/_static/validation_summary.*`
- Added a one-command figure bundle generator:
  - `examples/make_publication_figures.py`
  - `docs/_static/publication_figure_manifest.json`
- Current publication inventory is now sufficient to start a methods-paper
  draft. The only optional missing figure is an application-specific science
  result, which is not required for the NTX methods manuscript itself.
- Added a science/application figure for differentiable bootstrap-current
  optimization:
  - `examples/bootstrap_current_optimization.py`
  - `docs/_static/bootstrap_current_optimization.*`
  - uses a VMEC-derived radial surface family
  - optimizes one dominant non-axisymmetric harmonic to increase a weighted
    bootstrap-current proxy
  - annotates representative serial and multiprocess scan timings
- Added an output-inspection plotting example:
  - `examples/plot_output_npz.py`
  - `docs/_static/output_file_summary.*`
  - turns a CLI `.npz` payload into a four-panel publication-style summary
- Updated the publication bundle generator so the science figure is part of the
  default figure inventory.
- Expanded the user-facing docs and README so they now start from the simplest
  `ntx input.toml` workflow, then cover Python solves, scans, parallelization,
  NEOPAX coupling, autodiff, output plotting, and manuscript figures.
- Added a polished NTX-only bootstrap-current-proxy figure from the imported
  VMEC/Boozer workflow:
  - `examples/bootstrap_current_from_vmec_or_boozmn.py`
  - `docs/_static/bootstrap_current_from_vmec_or_boozmn.*`
  - uses top-of-file configuration instead of parsed CLI arguments
  - writes a JSON summary of the NTX radial profiles without embedding
    machine-specific absolute paths
- Audited the W7-X benchmark path and found that the solver itself matches the
  benchmark pointwise when the direct VMEC lane and `25 x 25 x 64` resolution
  are used.
- Added a separate W7-X reference-audit example and test coverage so the public
  bootstrap-current example stays focused on the NTX workflow.
- Current local validation after that publication pass:
  - `113 passed, 2 skipped`
  - `ruff` clean
  - docs build clean
- Began the final documentation expansion to make NTX standalone and ship-ready
  as a solver package rather than only a code-and-figures repository.
- Added dedicated documentation pages for:
  - physics model and equations
  - geometry loading and normalization
  - numerics and algorithms
  - source-code mapping
  - testing and QA
  - literature and package references
- Reworked the landing page and README so they now start from the smallest
  install-and-run workflow before moving into Python solves, scans, autodiff,
  NEOPAX coupling, and throughput-oriented parallel execution.
- Audited the next research-grade gaps against nearby transport/profile tools
  and the current literature:
  - profile tools expect clean radial-normalization-aware database interfaces
  - practical multi-GPU throughput is strongest as one worker per case or scan
    point rather than one large sharded solve
  - adjoint or implicit derivatives should be introduced only after direct
    autodiff is checked carefully against finite differences
- Started the first concrete research-grade deliverable:
  - added a derivative-audit workflow for `D11` and `D33` sensitivities with
    respect to a Boozer harmonic amplitude and the radial electric field
  - this establishes the validation baseline for a future custom-VJP or
    implicit derivative path in the dense prepared solve
