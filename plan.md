# NTX JAX Neoclassical Transport Plan

## Summary

Build `NTX` as a new JAX-native neoclassical transport code in
the local NTX checkout, with a private GitHub repo at `github.com/uwplasma/NTX`.
The implementation is based on Escoto's Legendre-space monoenergetic DKE
formulation from arXiv:2510.27513. The existing local REFERENCE_EXECUTABLE checkout is used
only as an external numerical benchmark.

Important environment facts found during planning:

- Local thesis PDF exists in the local research workspace.
- Local benchmark checkout exists in the local research workspace,
  commit `428ca44`, with sample W7-X EIM outputs.
- `gh` is authenticated with `repo` and `workflow` scopes, so repo creation
  should work if the account has `uwplasma` permission.
- Local JAX is CPU only; GPU validation should be run through the user's
  `sh office` workflow when available.

## Key Changes

- Initialize a Python package with `pyproject.toml`, `src/ntx/`, `tests/`,
  `docs/`, `plan.md`, `.readthedocs.yaml`, and GitHub Actions workflows.
- Public API:
  - `BoozerSurface`: modes and flux-surface constants `Np`, `iota`, `psi_p`,
    `B_theta`, `B_zeta`, `B0`.
  - `GridSpec`: `n_theta`, `n_zeta`, `n_xi`, `dtype`, and x64/device options.
  - `MonoenergeticCase`: `nu_hat`, `Epsi_hat` or `Er_hat`.
  - `solve_monoenergetic(surface, grid, case) -> TransportResult`.
  - `TransportResult`: `D11`, `D31`, `D13`, `D33`, `D33_spitzer`, optional
    Legendre modes `f1[0:3]`, `f3[0:3]`, residual norms, and metadata.

## Numerical Implementation

- Solve `L_k f^(k-1) + D_k f^k + U_k f^(k+1) = s^k`.
- Sources: `s1` has modes 0 and 2; `s3` has mode 1.
- Enforce the nullspace condition by replacing the first row of the `k=0`
  diagonal block so `f^(0)(theta=0,zeta=0)=0`.
- Only back-substitute modes `k=0,1,2` for the transport coefficients, while
  allowing `n_xi` to be large in the forward Schur recursion.
- Use `jax_enable_x64=True` by default for physics runs, pure functional data,
  `jax.jit`, `jax.vmap`, `jax.lax.scan`, `jax.scipy.linalg.lu_factor`, and
  `jax.scipy.linalg.lu_solve`. Never form explicit matrix inverses.

## Validation

- Unit tests cover Legendre identities, Fourier derivatives, Boozer geometry,
  operator blocks, nullspace replacement, and small residuals.
- Physics tests cover uniform-field sanity, Onsager symmetry, diagonal
  positivity, Spitzer scaling, and convergence trends.
- Regression tests compare against external benchmark outputs with tolerances
  appropriate to the collisionality and resolution.
- CPU tests run in GitHub Actions. GPU tests are marked `gpu` and run via the
  local GPU workflow.

## Repo, Docs, And CI

- Create a local git repo and private GitHub repo at `uwplasma/NTX`.
- Use `pytest`, `ruff`, `mypy`, `numpy`, `scipy`, `jax`, `jaxlib`, and optional
  `netCDF4`/`xarray`.
- Add GitHub Actions for install, lint, unit tests, physics smoke tests, and docs.
- Add ReadTheDocs configuration and Sphinx docs with equations, API examples,
  benchmark methodology, and citations.

## Assumptions

- If `gh repo create uwplasma/NTX --private` fails due to org permissions, keep
  the local repo ready and request org admin creation rather than changing the
  target owner/name.
- First release scope is monoenergetic geometric coefficients `D11`, `D31`,
  `D13`, `D33` plus `D33_spitzer`; full momentum-restoring transport is deferred
  until after the monoenergetic solver is validated.

## Current Execution Plan

- [x] Establish the initial JAX solver, CLI, docs, CI, and DKES/VMEC support.
- [x] Add verbose `ntx input.toml` runs and rich `.npz` outputs.
- [x] Match the DKES path against the external local benchmark code.
- [x] Add the first VMEC fixture, examples, docs, and regression coverage.
- [x] Formalize VMEC transport normalization and add a principled `er_hat` path.
- [x] Add a second VMEC fixture and regression family.
- [x] Add GPU smoke/regression runs for one DKES case and one VMEC case.
- [ ] Finish the archived cross-code benchmark campaign, with exact CIEMAT-QI
  archived solves and a clear interpretation of the DKES/SFINCS spread.
- [x] Push the current scan and GPU performance gains through the full office workflow.
- [ ] Close the VMEC cross-code gap now that a direct REFERENCE_EXECUTABLE VMEC comparison path exists.
- [x] Add a compiled prepared-solver path for repeated fixed-geometry solves.
- [x] Keep runtime benchmarking honest by leaving eager mode as the default and
  making compiled mode explicit.
- [x] Reduce runtime-benchmark test cost by introducing a dedicated smoke
  benchmark case instead of reusing a production-sized case in tests.
- [x] Prove that the imported NTX core can differentiate through Boozer solves.
- [x] Add a pure-JAX in-memory monoenergetic database builder for NEOPAX-facing workflows.
- [x] Add an initial `vmec_jax -> booz_xform_jax -> NTX` imported geometry path.
- [x] Add an explicit NTX-to-NEOPAX monoenergetic database mapping layer.
- [x] Replace the legacy VMEC and Boozer file readers with `vmec_jax` and `booz_xform_jax`.
- [x] Close the remaining W7-X NEOPAX subset mismatch in `D11` and `D13` on the
  Eduardo-REFERENCE_EXECUTABLE-reference VMEC database path.
- [x] Add a direct `vmec_jax` VMEC-harmonic imported path for NEOPAX-facing scans.
- [x] Close the W7-X NEOPAX subset on the direct `vmec_jax` VMEC-harmonic path
  to better than `1e-2` relative error.
- [x] Add a callback-free imported NEOPAX scan builder and pure-array mapping layer.
- [x] Add a second-family imported NEOPAX example and QI HDF5 round-trip validation.
- [x] Scope the fully JAX `vmec_jax -> booz_xform_jax -> NTX` workflow as a
  separate Boozer-transform validation lane distinct from the W7-X NEOPAX
  parity path.
- [x] Scrub machine-specific absolute paths from source, tests, examples, and docs.
- [x] Add QA, QH, and QI omnigenous validation fixtures from the local study set.
- [x] Generate external REFERENCE_EXECUTABLE subset databases for QA, QH, and a parity-safe QI subset.
- [x] Close the remaining QI `rho = 0.25` mismatch on the Eduardo-REFERENCE_EXECUTABLE-reference VMEC path.
- [x] Add shipping-grade package and release workflows, with built-distribution validation.

## Work Log

### 2026-04-08

- Confirmed the existing VMEC path still used a placeholder `transport_psi_scale = 1.0`,
  which kept `epsi_hat` runs workable but made `er_hat` unsupported and left the VMEC
  transport normalization under-specified.
- Audited the local `sfincs_jax` VMEC radial conversions. The relevant derivative factors
  are in the local `sfincs_jax` checkout, where
  `dpsi_hat/dr_hat = 2 * psi_a_hat * sqrt(psi_n) / a_hat` and
  `dr_hat/dpsi_hat` is its reciprocal.
- Verified the local QI VMEC candidates in
  the local `sfincs_jax` additional examples are
  stellarator-symmetric (`lasym = 0`) and include `Aminor_p`, so they are suitable
  for NTX regression coverage.
- Implemented the VMEC normalization update so NTX now derives:
  - `r_n = sqrt(psi_n)`
  - `r_hat = Aminor_p * r_n`
  - `transport_psi_scale = dpsi_hat/dr_hat`
  - `dr_hat/dpsi_hat`
- Implemented VMEC `er_hat` support by resolving `epsi_hat = er_hat / transport_psi_scale`
  instead of hard-rejecting `er_hat` on VMEC surfaces.
- Extended the VMEC runtime metadata and `.npz` payload with `r_n`, `r_hat`,
  `dpsi_hat/dr_hat`, and `dr_hat/dpsi_hat`.
- Added focused tests for the new VMEC normalization and `er_hat` resolution path.
- What worked:
  - The new normalization path is consistent across solver execution, Rich terminal output,
    and `.npz` output metadata.
  - Focused tests passed after the patch.
- What did not:
  - The previous VMEC regression references are now stale because they were recorded with
    the old placeholder normalization. They need to be regenerated and updated before the
    full suite can be considered current again.

- Vendored a second VMEC fixture into NTX:
  `tests/fixtures/wout_QI_nfp2_stable_Er_006_000043_hires_scaled.nc`.
- Confirmed the QI fixture solves cleanly on the current VMEC path with:
  - `psi_n = 0.12247^2`
  - `nfp = 2`
  - `loaded_mode_count = 72`
  - `transport_psi_scale = 0.9673631438898428`
- Added a QI VMEC example input around the new `er_hat` path:
  `examples/qi_vmec_erhat.toml`.
- Added dedicated QI VMEC tests:
  - unit coverage in `tests/test_vmec.py`
  - physics coverage in `tests/test_vmec_qi.py`
  - regression coverage in `tests/test_vmec_regression.py`
  - CLI coverage in `tests/test_cli.py`
- What worked:
  - The QI fixture provides a genuinely different VMEC family from W7-X while remaining
    small enough for repository regression coverage.
  - The new `er_hat` path remains numerically consistent with explicit `epsi_hat` runs on
    both W7-X and the QI VMEC fixture.
- What did not:
  - Documentation and user-facing examples still need to be updated to describe the new
    QI fixture and the VMEC `er_hat` normalization explicitly.

- Updated the user-facing docs and examples to match the current VMEC and GPU paths:
  - `README.md`
  - `docs/index.md`
  - `docs/install.md`
  - `docs/input-file.md`
  - `docs/algorithm.md`
  - `docs/examples.md`
  - `docs/gpu.md`
  - `examples/w7x_vmec.toml`
  - `examples/qi_vmec_erhat.toml`
- Added GPU regression tooling:
  - `tests/test_gpu_smoke.py`
  - `tests/test_gpu_scripts.py`
  - `scripts/run_gpu_regression.py`
  - `scripts/sh_office_gpu_smoke.sh`
- Expanded `.npz` source-file metadata with:
  - `surface_source_name`
  - `surface_source_size_bytes`
  - `surface_source_mtime`
- Validation that worked:
  - `ruff check .`
  - `mypy src/ntx`
  - `pytest -q` -> `36 passed, 2 skipped`
  - `sphinx-build -b html docs docs/_build/html`
- What did not:
  - Actual GPU execution could not be validated in this local session because the available
    JAX environment is CPU-only. The GPU tests were added and skipped cleanly, and the
    `sh office` workflow is now scripted for the next hardware-backed run.

- Office GPU execution completed on 2026-04-08:
  - host: `office` (`pop-os`)
  - Python: `3.10.12`
  - JAX: `0.6.2`
  - jaxlib: `0.6.2`
  - backend: `gpu`
  - device: `cuda:0` (`NVIDIA RTX A4000`)
- What worked on the office machine:
  - `ruff check .`
  - `mypy src/ntx`
  - `pytest -q` -> `36 passed, 2 skipped`
  - `pytest -m gpu -q` -> `2 passed`
  - `python3 -m sphinx -b html docs docs/_build/html`
  - `python3 scripts/run_gpu_regression.py --output-json gpu-smoke-results.json`
- GPU regression results from `office`:
  - `dkes_w7x_smoke`: compile+run `1.444533 s`, steady `0.001732 s`,
    max relative error `9.439e-09`
  - `vmec_w7x_smoke`: compile+run `1.589561 s`, steady `0.002774 s`,
    max relative error `5.681e-13`
- Follow-up fixes required by the office run:
  - NTX originally rejected the office GPU Python because the package declared
    `requires-python >= 3.11`. This was fixed by making NTX explicitly Python 3.10
    compatible and adding a `tomli` fallback for the TOML loader.
  - The GPU regression script originally tried to `jit` a function that returned
    `TransportResult`, which is not a JAX pytree. This was fixed by jitting a helper
    that returns a plain coefficient vector and by enabling x64 explicitly inside the
    script before loading surfaces.
- What did not:
  - The office environment still emits a third-party `flatbuffers.compat` deprecation
    warning during pytest. This did not affect correctness and no NTX change was needed.

- Added a new imported JAX geometry lane:
  - `src/ntx/vmec_jax_backend.py`
  - `examples/vmec_jax_booz_xform_jax_ntx.py`
- Added a new Boozer-file helper for `boozmn` surfaces:
  - `src/ntx/booz.py`
  - `tests/test_boozmn.py`
- Added an explicit NTX-to-NEOPAX mapping layer:
  - `src/ntx/neopax.py`
  - `examples/neopax_with_ntx.py`
  - `tests/test_neopax_adapter.py`
- Local validation that worked for the new layer:
  - `pytest -q tests/test_boozmn.py tests/test_neopax_adapter.py` -> `4 passed`
  - `ruff check` on the new modules and tests
  - `mypy src/ntx`
- Quantified current W7-X subset behavior for the NEOPAX adapter path:
  - the NTX-to-NEOPAX constructor reproduces the existing NEOPAX/REFERENCE_EXECUTABLE HDF5
    mapping exactly when fed the reference HDF5 arrays
  - for an NTX-generated W7-X subset built from the VMEC file path with
    `vmec_nyquist_option = 2` and `vmec_mode_convention = "filtered_nyquist"`,
    the mapped `D33` stayed within 20% of the reference subset and the mapped
    `D11_log` stayed within 1 dex
- What worked:
  - the adapter itself is correct and explicit, without going through an
    intermediate HDF5 file
  - the `vmec_jax -> booz_xform_jax -> NTX` example runs locally and gives a
    viable imported path for end-to-end JAX workflows
- What did not:
  - full W7-X VMEC parity is not closed yet
  - a naive direct Boozer-surface solve from the archived `boozmn` file gave
    materially worse agreement than the existing VMEC file path, so the legacy
    VMEC loader cannot be removed until the new JAX-native geometry lane is
    benchmarked more carefully against the reference databases

### 2026-04-09

- Generated the missing omnigenous QI `boozmn` fixture locally through
  `booz_xform_jax` and added it to `tests/fixtures/`.
- Found and fixed two packed-radial-grid bugs:
  - `src/ntx/vmec_reference_executable.py` now reads `jlist` from packed `boozmn` files and
    interpolates the reference factors on the actual packed-surface grid.
  - `src/ntx/booz.py` now falls back to netCDF `jlist` metadata when the
    `booz_xform_jax` radial profile length does not match `bmnc_b`, which is
    required for the generated omnigenous QI `boozmn` file.
- Verified that the remaining omnigenous QI spread is not a geometry problem:
  - `load_vmec_surface_reference_executable_reference(...)` matches Eduardo REFERENCE_EXECUTABLE
    `Field.from_vmec_s(...)` to roundoff on the QI geometry arrays.
  - The solver-side transport coefficients still show a large mismatch at
    `rho = 0.25`.
- Generated external REFERENCE_EXECUTABLE subset databases under
  `tests/fixtures/benchmarks/omnigenity/` with
  `scripts/generate_reference_executable_omnigenity_references.py`:
  - `reference_executable_external_qa_subset.h5`
  - `reference_executable_external_qh_subset.h5`
  - `reference_executable_external_qi_subset.h5`
- Added `tests/test_reference_executable_reference_omnigenity.py`, which now checks:
  - the generated QI `boozmn` fixture loads successfully through NTX
  - QA external subset parity to about `2e-2`
  - QH external subset parity to about `3e-2`
  - QI external subset parity at `rho = 0.5` to roundoff
- What worked:
  - QA and QH now have real external REFERENCE_EXECUTABLE reference subsets vendored in the
    repository, not just internal transport-lane comparisons.
  - The QI family now also has an external subset reference, but only for the
    `rho = 0.5` point where NTX and REFERENCE_EXECUTABLE agree.
- What did not:
  - QI at `rho = 0.25` still differs strongly from REFERENCE_EXECUTABLE even after enforcing
  x64 and confirming geometry parity. This remains an open solver-side audit.

- Continued the QI audit on the relocated checkout at `local/NTX`.
- Compared the NTX operator against Eduardo REFERENCE_EXECUTABLE on the QI `rho = 0.25` point
  and found that the mismatch was not in the block-tridiagonal solve itself:
  NTX's Schur recursion matched an independent block-tridiagonal solve on the
  same NTX operator.
- Isolated the actual bug to the comparison-only VMEC reference loader:
  `load_vmec_surface_reference_executable_reference(...)` was using linear interpolation,
  while Eduardo REFERENCE_EXECUTABLE uses `interpax.interp1d(..., method='cubic')` on the
  VMEC half-grid and related radial profiles.
- Updated `src/ntx/vmec_reference_executable.py` so the comparison-only VMEC path now uses
  `interpax` cubic interpolation for:
  - the VMEC half-grid mode tables
  - `iotaf`
  - the Boozer-side `B00`, `R00`, `I`, `G`, and `iota` conversion profiles
- Re-ran the QI geometry check and closed the pointwise geometry gap to
  roundoff at both `rho = 0.25` and `rho = 0.5`.
- Re-ran the QI transport comparison and closed the `rho = 0.25` and
  `rho = 0.5` subset to roundoff against direct Eduardo REFERENCE_EXECUTABLE solves.
- Regenerated the vendored external QI subset database so it now includes both
  `rho = 0.25` and `rho = 0.5`.
- Added `interpax` as an NTX dependency because the comparison/reference lane
  and its tests now rely on the same cubic interpolation used by Eduardo REFERENCE_EXECUTABLE.
- What worked:
  - The QI mismatch was a reference-loader interpolation mismatch, not a dense
    solver bug.
  - QA, QH, and QI now all have real external REFERENCE_EXECUTABLE subset databases vendored
    in the repository.
- What did not:
  - Nothing new on this path after the interpolation fix; the remaining open
    items are elsewhere in the broader plan.

- Installed-stack validation on 2026-04-08:
  - `python -m pip install -e ../vmec_jax` succeeded
  - `python -m pip install -e ../tests/NEOPAX` succeeded
  - `python -m pip install -e .` succeeded
  - `python -m pip install -e ../booz_xform_jax` failed
    because `src/booz_xform_jax.egg-info` is owned by `root`, so setuptools
    could not update its timestamp during build
- Important environment result after the NEOPAX install:
  - NEOPAX downgraded the active stack to `jax==0.5.0`, `jaxlib==0.5.0`, and
    `numpy==2.2.2`
  - NTX still passed `pytest -q` in that installed environment:
    `68 passed, 2 skipped`
- What worked:
  - NTX remains compatible with the JAX version that the local NEOPAX package
    currently installs
  - the imported modules `ntx`, `vmec_jax`, `NEOPAX`, and `booz_xform_jax`
    all resolved successfully after the install pass
- What did not:
  - the local `booz_xform_jax` repo needs packaging/ownership cleanup if it is
    to be installed cleanly with `pip -e` instead of being imported from source

- Audited the package metadata and removed strict lower bounds on runtime and
  development dependencies in `pyproject.toml`.
- Expanded the GitHub Actions CPU workflow to a Python matrix covering `3.10`,
  `3.11`, and `3.12`, and separated docs into their own job.
- What worked:
  - NTX no longer encodes a preferred JAX or NumPy floor in its package metadata.
  - CI now reflects the supported Python range directly instead of assuming one
    interpreter version.
- What did not:
  - This change still depends on the solver and tests remaining portable across
    the matrix; the expanded CI needs to stay enabled so future regressions are
    caught by the workflow rather than by packaging metadata.

- Vendored small archived DKES and SFINCS benchmark tables plus the full
  W7-X EIM and CIEMAT-QI DKES surfaces into `tests/fixtures/benchmarks/` and
  `tests/fixtures/`.
- Added archived benchmark readers in `src/ntx/benchmarks.py` and a standalone
  comparison script at `scripts/compare_archived_benchmarks.py`.
- Validation that worked:
  - The archived reference tables parse reproducibly in tests.
  - The comparison script produces a structured JSON report for W7-X EIM and
    CIEMAT-QI.
- What did not:
  - NTX is not yet close to the archived converged DKES and SFINCS curves.
    The current comparison report shows large relative errors, especially for
    low-collisionality W7-X EIM and for the CIEMAT-QI DKES path. This is now
    quantified and reproducible, but not solved.

- Found and fixed a real VMEC bug in `solve_monoenergetic_scan`: VMEC `er_hat`
  scans incorrectly rejected valid surfaces because the scan path checked
  `psi_p` instead of the resolved VMEC transport scale.
- Added new VMEC scan coverage in `tests/test_vmec_scan.py` for:
  - loop-versus-scan agreement
  - `er_hat` versus explicit `epsi_hat` agreement
  - W7-X and QI scan regression values
- Added runtime profiling in `scripts/profile_runtime.py`.
- Local CPU profiling after jitting the batched scan kernel on 2026-04-08:
  - case: W7-X VMEC, `GridSpec(9, 11, 6)`, 8-point `nu_hat` scan at `er_hat = 1e-3`
  - scan compile+run: `1.3269 s`
  - scan steady-state: `0.2899 s`
  - Python loop: `2.1097 s`
  - steady-state scan speedup versus loop: about `7.3x`
- What worked:
  - The VMEC scan path now accepts `er_hat` directly.
  - The jitted scan kernel gives a clear throughput win for repeated solves.
- What did not:
  - The single-case dense solve path is still the dominant cost for archived
    high-resolution DKES comparisons, so the current performance gain helps scans
    more than it helps the hardest benchmark configurations.

- Audited the differentiability requirements against:
  - `tests/NEOPAX`
  - `tests/reference_executable_f0`
- Confirmed the key NEOPAX integration point is still the monoenergetic
  database abstraction in `NEOPAX/_database.py`, and that the commented
  lower-level path there expects a Python-callable monoenergetic solver similar
  to `reference_executable._core.monoenergetic_dke_solve_internal(...)`.
- Local autodiff probes showed:
  - gradients through Boozer Fourier coefficients already worked in NTX
  - gradients through `nu_hat` already worked
  - gradients through `er_hat` failed only because `solver.py` converted traced
    values with `float(...)`
  - `solve_monoenergetic_scan(...)` already differentiated through `er_hat`
  - `jax.jit` over a surface argument failed because the NTX surface/config
    dataclasses were not registered as pytrees
- Implemented the first differentiable-core pass:
  - registered `GridSpec`, `AngularGrid`, `BoozerSurface`, `VmecSurface`,
    `GeometryOnGrid`, `MonoenergeticCase`, `TransportResult`, and
    `PreparedMonoenergeticSystem` as JAX dataclass pytrees
  - removed tracer-breaking scalar coercions from the core `er_hat ->
    epsi_hat` resolution path
  - added `solve_monoenergetic_internal(...) -> (Dij, f, s)`
  - added `solve_prepared_internal(...) -> (Dij, f, s)`
- Validation that worked:
  - `grad` through `er_hat`
  - `grad` through `nu_hat`
  - `grad` through Boozer `b_cos`
  - `jit` over a Boozer surface argument in the imported solver path
  - local tests in `tests/test_solver.py`
- What did not:
  - the VMEC file loader remains non-differentiable by design because it still
    uses SciPy/NumPy file parsing, Python interpolation, and scalar coercions
  - this means the correct next step is not to force autodiff through the file
    loader, but to add a separate pure-JAX VMEC-array builder for in-memory use
    in higher-level JAX workflows

- Added an in-memory scan/database layer in `src/ntx/database.py`:
  - `MonoenergeticDatabaseArrays`
  - `build_monoenergetic_database_arrays(...)`
  - `stack_monoenergetic_database_arrays(...)`
- The builder produces tensor-product scan arrays over `(nu_hat, scan_field)`
  for one surface, entirely in memory, using the existing differentiable
  `solve_monoenergetic_scan(...)` path.
- Validation that worked:
  - database-shape and stacking tests in `tests/test_database.py`
  - gradient through the scan-field axis in the new database builder
- What did not:
  - this is not yet a direct NEOPAX adapter, because the NEOPAX-side
    normalization and database-field conventions still need an explicit mapping
    layer on top of these raw NTX scan arrays

- Revalidated the updated code on the office GPU machine after the packaging,
  scan, and benchmark-script changes.
- Office environment on 2026-04-08 after this round:
  - host: `office`

- Added `compile_prepared_solver()` to expose a jitted repeated-solve path for a
  fixed prepared geometry. This keeps the existing eager APIs intact while
  making the optimized path explicit for repeated solves.
- Measured the local macOS CPU runtime tradeoff on 2026-04-08 for the W7-X EIM
  `23 x 55 x 80` case:
  - eager mode: first `8.54 s`, steady `6.44 s`
  - compiled mode: first `6.82 s`, steady `5.71 s`
- What worked:
  - The compiled prepared-solver path matches the eager solver in tests.
  - The heavy local CPU case does get a modest runtime reduction from the
    compiled path.
- What did not:
  - The compiled path is not a universal win. On the `office` CPU host, a
    repeated compiled benchmark attempt for the same heavy case took long enough
    to be impractical for the benchmark workflow, so the benchmark harness now
    defaults back to eager mode and treats compiled mode as an explicit option.

- Added an explicit `--mode {eager,compiled}` switch to
  `scripts/benchmark_against_reference_executable.py`.
- Added a dedicated benchmark smoke case:
  - `w7x_eim_smoke`
  - grid `5 x 5 x 4`
  - `nu_hat = 1e-5`
  - `er_hat = 1e-3`
- Updated `tests/test_benchmark_runtime_script.py` to use the smoke benchmark
  case instead of the production-sized W7-X EIM case.
- What worked:
  - The benchmark script still covers the CLI/runtime JSON path.
  - The smoke benchmark cuts the local benchmark-script test down to about
    `18 s` for the combined solver and benchmark-script test file instead of
    spending most of that time in a production-sized runtime benchmark.
- What did not:
  - A full office rerun of the complete test suite still spends substantial
    wall time on heavy runtime-oriented tests. For the latest branch, the useful
    office validation set was the changed solver tests, GPU smoke tests, docs
    build, and the GPU regression script rather than another full remote
    end-to-end suite pass.

- Office targeted validation on 2026-04-08 after the smoke-benchmark change:
  - `python3 -m pytest -q tests/test_solver.py tests/test_benchmark_runtime_script.py`
    -> `8 passed, 1 warning in 41.81 s`
  - `python3 -m pytest -m gpu -q tests/test_gpu_smoke.py`
    -> `2 passed, 1 warning in 23.33 s`
  - `python3 -m sphinx -b html docs docs/_build/html`
    -> passed
  - `python3 scripts/run_gpu_regression.py --output-json gpu-smoke-results.json`
    -> passed
- Office GPU regression results in that targeted run:
  - `dkes_w7x_smoke`: compile+run `4.279937 s`, steady `0.004156 s`,
    max relative error `9.439e-09`
  - `vmec_w7x_smoke`: compile+run `6.576037 s`, steady `0.006947 s`,
    max relative error `3.268e-07`
- What worked:
  - The changed solver tests, benchmark-script test, GPU smoke tests, docs
    build, and GPU regression all passed on `office`.
- What did not:
  - Repeating the heavy same-host W7-X EIM CPU benchmark on the shared office
    host was not a good use of wall time once the earlier same-day eager
    measurements were already in hand, so that rerun was stopped and the prior
    validated eager benchmark numbers were kept as the reference.
  - Python: `3.10.12`
  
- Added a standalone runtime benchmark runner:
  - `scripts/benchmark_against_reference_executable.py`
  - test coverage in `tests/test_benchmark_runtime_script.py`
- What worked:
  - the benchmark payload now records `xla_preallocate`
  - CPU runs no longer report phantom GPU memory usage
  - timings explicitly block on device completion before serialization
- What did not:
  - the first benchmark attempt on `office` mixed CPU and GPU runs concurrently,
    which inflated the apparent wall time for both REFERENCE_EXECUTABLE and NTX. Those numbers
    were discarded and rerun sequentially.

- Built REFERENCE_EXECUTABLE successfully on `office` using the local miniforge `qh-gpu`
  toolchain:
  - `gfortran`: the office GPU environment compiler
  - NetCDF Fortran include: the office GPU environment include directory
  - NetCDF Fortran lib: the office GPU environment library directory
- What worked:
  - same-host NTX versus REFERENCE_EXECUTABLE runtime comparisons are now reproducible on
    `office`
- What did not:
  - the first build attempt ran before the REFERENCE_EXECUTABLE rsync had fully settled and
    produced a misleading `quadpack.f` make error. Re-running the build after
    the sync completed fixed it without any source change.

- Same-host W7-X EIM runtime comparison on `office`, 2026-04-08:
  - case: `23 x 55 x 80`, `nu_hat = 1e-5`, `er_hat = 0`
  - REFERENCE_EXECUTABLE CPU wall: `4.2915 s`
  - NTX CPU first run: `13.5266 s`
  - NTX CPU steady run: `8.9925 s`
  - NTX CPU RSS: `2151912 KiB`
  - REFERENCE_EXECUTABLE CPU RSS: `201944 KiB`
  - NTX/REFERENCE_EXECUTABLE steady CPU runtime ratio: `2.095x`
  - NTX/REFERENCE_EXECUTABLE CPU RSS ratio: `10.656x`
  - NTX GPU first run with `XLA_PYTHON_CLIENT_PREALLOCATE=false`: `10.6983 s`
  - NTX GPU steady run with `XLA_PYTHON_CLIENT_PREALLOCATE=false`: `4.3680 s`
  - NTX GPU sampled memory: `740 MiB`
  - NTX/REFERENCE_EXECUTABLE steady GPU runtime ratio: `0.873x`
- What worked:
  - NTX on GPU is now slightly faster than REFERENCE_EXECUTABLE CPU for this representative
    W7-X EIM benchmark case on the same machine.
  - the DKES coefficients remain matched to the external reference to roundoff.
- What did not:
  - NTX CPU remains materially slower and much heavier in host memory than
    REFERENCE_EXECUTABLE for the same case.

- Tested one additional CPU tuning on `office`:
  - `OMP_NUM_THREADS=1`
  - `XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"`
- Result:
  - first run: `18.98 s`
  - steady run: `15.18 s`
  - RSS: `2151408 KiB`
- Interpretation:
  - forcing single-threaded CPU execution is a regression for NTX on this case
    and does not materially reduce host memory use.

- Extended the standalone REFERENCE_EXECUTABLE comparison script to accept VMEC inputs by
  staging `VMEC.nc`, writing `reference_executable_input.surface`, and passing the resolved
  NTX `epsi_hat` to REFERENCE_EXECUTABLE.
- Added VMEC comparison coverage in `tests/test_reference_executable_script.py`.
- Updated the VMEC loader so:
  - `vmec_nyquist_option = 1` uses the primary VMEC mode set
  - `vmec_nyquist_option = 2` uses the full Nyquist mode set
- What worked:
  - the VMEC comparison path now runs cleanly instead of being blocked by input
    staging
  - tests now pin the distinction between the primary and Nyquist VMEC mode sets
- What did not:
  - direct W7-X VMEC comparison against REFERENCE_EXECUTABLE still shows a large mismatch, so
    the VMEC physics/normalization path is still not validated to the same level
    as DKES.
  - JAX: `0.6.2`

- Audited the archived benchmark parser against the vendored thesis tables and
  found a real tooling bug: the archived DKES and SFINCS tables were already in
  NTX coefficient units, so `compare_archived_benchmarks.py` was incorrectly
  multiplying them by extra powers of `psi_p`.
- Confirmed that the archived W7-X EIM benchmark labelled `0.200` actually uses
  the same DKES magnetic-configuration scalars as the example `s = 0.25`
  surface:
  - `psi_p = -0.5237`
  - `chi_p = 0.4512`
  - `iota = -0.861561962955891`
  - `B00 = 2.4311`
  - `B_zeta = -14.0876`
- Added a generic text magnetic-configuration loader to NTX so benchmark
  surfaces archived as Fourier tables can be solved directly without going
  through DKES or VMEC input formats.
- Vendored the W7-X KJM archived benchmark family into `tests/fixtures/`,
  including:
  - DKES results
  - SFINCS zero- and finite-electric-field scans
  - the archived magnetic configuration
  - the archived monoenergetic reference table
- Added archived monoenergetic reference tables for W7-X EIM and CIEMAT-QI and
  extended the benchmark tooling so NTX can compare against:
  - archived monoenergetic reference tables
  - archived DKES tables
  - archived SFINCS tables
- What worked:
  - W7-X EIM: at `23 x 55 x 80`, NTX matches the archived monoenergetic
    reference to floating-point tolerance.
  - W7-X KJM: at `19 x 79 x 180`, NTX matches the archived monoenergetic
    reference to floating-point tolerance.
  - The remaining spread versus archived DKES and SFINCS is now exposed as a
    real cross-code comparison rather than a parser artifact.
  - The full test suite remained green after the benchmark-tooling changes:
    `44 passed, 2 skipped`.
- What did not:
  - The exact archived CIEMAT-QI solve at `47 x 215 x 160` is substantially
    heavier than the W7-X cases. It is now wired into the archived benchmark
    script, but it is still too slow to treat as a default smoke check and
    should remain a selective validation run until the dense solver path is
    sped up further.

- Tried a targeted dense-solver speed cleanup in the low-mode back-substitution:
  - reused LU factors for the saved `k = 0, 1, 2` blocks
  - solved the `F1` and `F3` right-hand sides together where they share the
    same LU factor
- What worked:
  - The W7-X benchmark outputs stayed unchanged to roundoff after the patch,
    which confirms the algebraic refactor did not change the physics.
  - The low-mode solve path is now cheaper and simpler, with fewer repeated LU
    factorizations.
- What did not:
  - The exact CIEMAT-QI archived solve at `47 x 215 x 160`, `nu_hat = 1e-5`,
    `er_hat = 0` still did not return quickly enough on the local CPU after
    this cleanup to justify promoting it into the default validation smoke path.
    More substantial dense-kernel work or a GPU-backed validation path is still
    needed there.
  - jaxlib: `0.6.2`
  - backend: `gpu`
  - device: `cuda:0`
- What worked on office after syncing the updated tree over SSH:
  - `python3 -m pip install -e ".[dev,docs,io]"`
  - `python3 -m ruff check .`
  - `python3 -m mypy src/ntx`
  - `python3 -m pytest -q` -> `42 passed, 2 skipped`
  - `python3 -m pytest -m gpu -q` -> `2 passed`
  - `python3 -m sphinx -b html docs docs/_build/html`
  - `python3 scripts/run_gpu_regression.py --output-json gpu-smoke-results.json`
  - `python3 scripts/profile_runtime.py --output-json runtime-profile.json`
- Office GPU regression results after this round:
  - `dkes_w7x_smoke`: compile+run `1.592315 s`, steady `0.001621 s`,
    max relative error `9.439e-09`
  - `vmec_w7x_smoke`: compile+run `1.638515 s`, steady `0.002720 s`,
    max relative error `5.681e-13`
- Office CPU runtime profile from `scripts/profile_runtime.py`:
  - `dkes_w7x_scan`: steady scan `2.0031 s` versus loop `7.6240 s`
    (`3.81x` speedup)
  - `vmec_w7x_scan`: steady scan `2.3734 s` versus loop `6.1010 s`
    (`2.57x` speedup)
- What did not:
  - The office shell does not have the user-site script directory on `PATH`, so
    `ruff`, `mypy`, and `pytest` had to be invoked as `python3 -m ...`.
  - The archived benchmark comparison script was too heavy to finish in a
    reasonable office run at its current comparison grids. The script now
    defaults to `JAX_PLATFORM_NAME=cpu`, but the archived benchmark report is
    still treated as an offline progress-tracking tool rather than a routine
    office smoke check.
- 2026-04-08: closed the main VMEC parity gap.
  - What worked:
    - The remaining W7-X VMEC mismatch was traced to two separate issues in the
      NTX VMEC path:
      - VMEC mode selection needed an explicit convention split. NTX now
        supports:
        - `vmec_mode_convention = "reduced"`: reduced `(xm, xn)` table with the
          coefficient arrays truncated by position.
        - `vmec_mode_convention = "filtered_nyquist"`: filtered Nyquist subset
          with `|m| < mpol` and `|n| <= ntor` in field-period units.
      - VMEC coefficient normalization had been conflated with the `er_hat ->
        epsi_hat` conversion scale. NTX now uses:
        - `transport_psi_scale = dpsi_hat/dr_hat` to resolve `er_hat`.
        - `coefficient_psi_scale = 1` for Escoto-style VMEC monoenergetic
          outputs.
    - VMEC radial interpolation now follows the centered quadratic Lagrange
      stencil used in REFERENCE_EXECUTABLE instead of simple linear interpolation.
    - After those fixes, the direct live W7-X VMEC comparison against the local
      REFERENCE_EXECUTABLE executable closed to roundoff at the example resolution
      (`9 x 11 x 8`, `nu_hat = 1e-3`):
      - `epsi_hat = 0`:
        - `D11`: `-7.03e-12`
        - `D31`: `+1.96e-11`
        - `D13`: `+7.05e-12`
        - `D33`: `-1.53e-11`
      - `er_hat = 1e-3`:
        - `D11`: `-2.53e-15`
        - `D31`: `-7.19e-13`
        - `D13`: `+6.26e-14`
        - `D33`: `-3.95e-12`
    - The local validation suite was updated to the corrected VMEC baselines and
      passed:
      - `ruff check .`
      - `mypy src/ntx`
      - `pytest -q` -> `49 passed, 2 skipped`
  - What did not:
    - Some earlier VMEC regression and physics tests had been encoding the old
      incorrect normalization. Those baselines were invalid and had to be
      replaced rather than preserved.
    - Coarse W7-X VMEC grids do not satisfy a small-Onsager-residual expectation.
      That is consistent with the live REFERENCE_EXECUTABLE comparison and should not be used
      as a physics gate at low angular / Legendre resolution.
- 2026-04-08: added independent `sfincs_jax` VMEC geometry validation and a
  prepared-system performance path.
  - What worked:
    - Added `src/ntx/sfincs_geometry.py` plus
      `scripts/compare_sfincs_geometry.py` to compare NTX VMEC geometry against
      a local `sfincs_jax` checkout.
    - Established the required convention conversion for the comparison:
      - reverse the sampled zeta direction
      - flip the Jacobian sign reconstructed from `sfincs_jax`
    - With those conversions, the snapped W7-X filtered-Nyquist geometry arrays
      (`B`, derivatives, covariant and contravariant components, Jacobian)
      matched `sfincs_jax` to roundoff.
    - Added cached repeated-solve support through
      `prepare_monoenergetic_system(...)` and `solve_prepared(...)`.
    - On a local W7-X sample DKES solve at `9 x 11 x 8`, the cached repeated
      solve reduced steady wall time from about `0.275 s` to about `0.206 s`
      (`1.34x` speedup).
  - What did not:
    - The `sfincs_jax` VMEC geometry path is not in the default CI environment,
      so the new cross-code geometry test is intentionally skipped unless the
      local checkout exists.
- 2026-04-08: fixed the broken CI/CD path introduced by the new Boozer loader.
  - What worked:
    - Root-caused the failed GitHub Actions run to two independent issues:
      - the test workflow installed only `.[dev]`, but the public NTX import
        path now exposes the Boozer loader, and the original implementation
        imported `netCDF4` at module import time.
      - Python 3.10 `mypy` rejected a couple of NumPy stub inferences in
        `src/ntx/vmec.py` and `src/ntx/booz.py`.
    - Fixed the workflow to install `.[dev,io]` for the matrix test jobs, which
      matches the real test surface area now covered in CI.
    - Made `src/ntx/booz.py` import `netCDF4` lazily inside
      `load_boozmn_surface(...)`, so `import ntx` no longer fails when the IO
      extras are absent.
    - Resolved the Python 3.10 typing failures by pinning the relevant NumPy
      array/scalar expectations explicitly.
    - Local validation after the fix passed:
      - `ruff check .`
      - `mypy src/ntx`
      - `pytest -q` -> `68 passed, 2 skipped`
      - `python -m sphinx -b html docs docs/_build/html`
  - What did not:
    - Relying on unconditional imports from optional IO code made the public API
      brittle and masked the real packaging boundary. The workflow fix alone
      would have hidden that design problem instead of fixing it.
- 2026-04-08: hardened the local-only JAX integration tests and re-audited the
  NEOPAX-facing parity path.
  - What worked:
    - Root-caused the remaining GitHub Actions failures after the IO fix to
      local-only integration tests that imported `NEOPAX` unconditionally at
      collection time.
    - Updated the local integration suites to skip cleanly when the expected
      editable checkouts are not present:
      - `tests/test_neopax_adapter.py`
      - `tests/test_boozmn.py`
      - `tests/test_jax_neopax_examples.py`
    - Local validation after the skip-guard update still passed:
      - `ruff check .`
      - `mypy src/ntx`
      - `pytest -q` -> `68 passed, 2 skipped`
    - Re-ran a local W7-X Boozer-to-NEOPAX subset comparison against
      `tests/NEOPAX/tests/inputs/Dij_NEOPAX_FULL_S_NEW_W7X.h5`.
      In NEOPAX storage conventions, `D33` stays within about `11%` to `19%`
      on the sampled subset, while `D11` and `D13` remain materially offset.
      That confirms the current gap is physical / normalization-related, not a
      broken adapter constructor.
  - What did not:
    - The fully JAX W7-X `vmec_jax -> booz_xform_jax -> NTX` parity audit is
      not closed yet. The available W7-X VMEC input file found locally has
      `cfg.ns = 51`, while the NEOPAX W7-X `wout` has `ns = 201`. Feeding that
      mismatched pair into `vmec_jax.booz_input.booz_xform_inputs_from_state`
      fails with an internal broadcast error in the lambda reconstruction path.
    - Until the exact matching W7-X VMEC input (or an equivalent way to rebuild
      the required static metadata directly from the `wout`) is identified, the
      W7-X JAX-native parity test remains blocked on geometry setup rather than
      on the NTX solver itself.
- 2026-04-08: added a WOUT-backed `vmec_jax` helper to keep the JAX geometry
  lane usable on the local W7-X data.
  - What worked:
    - Verified experimentally that rebuilding the VMEC static configuration with
      `ns = wout.ns` is enough to make
      `vmec_jax.booz_xform_inputs_from_state(...)` accept the local W7-X
      reference `wout`.
    - Added `surface_from_vmec_jax_wout(...)` in
      `src/ntx/vmec_jax_backend.py`. It:
      - loads the VMEC input with `vmec_jax.load_config(...)`
      - loads the `wout` with `vmec_jax.api.read_wout(...)`
      - rebuilds `cfg` with `ns/mpol/ntor` from the `wout` when needed
      - constructs the Boozer surface entirely through `vmec_jax` and
        `booz_xform_jax`
    - Updated the NEOPAX example to use the new helper, and the example ran
      locally with the expected output shapes.
    - Added `tests/test_vmec_jax_backend.py` for the new helper.
  - What did not:
    - Using the WOUT-backed helper does not by itself close the W7-X NEOPAX
      parity gap. On the sampled subset the resulting database still shows
      roughly the same `D11` / `D13` offset as the previous mixed-loader path,
      while `D33` remains the closest channel.
- 2026-04-09: fixed the Boozer `boozmn` geometry convention mismatch against
  the JAX REFERENCE_EXECUTABLE field loader and re-ran the local validation gates.
  - What worked:
    - Root-caused the remaining Boozer-side geometry offset to two issues in
      `src/ntx/booz.py`:
      - NTX had been snapping to the nearest stored surface instead of
        interpolating the scalar Boozer profiles in `s`.
      - NTX was not applying the handedness/sign convention used by
        `reference_executable.Field.from_booz_xform(...)` for `iota`, `buco`, and `bvco`.
    - Updated `load_boozmn_surface(...)` so it now:
      - interpolates `bmnc`, `iota`, `buco`, and `bvco` in `s`
      - handles the mixed full-grid / half-grid radial storage used in the
        local W7-X `boozmn` file
      - applies the same right-handed sign convention as the JAX REFERENCE_EXECUTABLE loader
    - Added a direct geometry-regression check in `tests/test_boozmn.py` at
      both `rho = 0.12247` and `rho = 0.5`.
    - After the fix, NTX and JAX REFERENCE_EXECUTABLE now agree on the shared W7-X Boozer
      geometry to the expected numerical tolerance for:
      - `B`
      - `dB/dtheta`
      - `dB/dzeta`
      - `B_theta`
      - `B_zeta`
      - `B x grad(psi) . grad(B) / B^3`
      - `iota`
    - Re-ran the full local validation suite after the loader fix:
      - `ruff check .`
      - `mypy src/ntx`
      - `pytest -q` -> `70 passed, 2 skipped`
      - `python -m sphinx -b html docs docs/_build/html`
  - What did not:
    - The W7-X NEOPAX subset mismatch is not closed by the geometry fix alone.
      `D33` remains comparatively close, while `D11` and `D13` are still
      materially offset on the sampled subset.
    - A direct JAX REFERENCE_EXECUTABLE monoenergetic solve on the archived W7-X `boozmn`
      file still returns `NaN` because that file carries `phip_b = 0`, so the
      raw REFERENCE_EXECUTABLE Boozer solve is not yet a usable transport reference on this
      dataset even though the geometry loader conventions now match.
- 2026-04-09: moved the file-backed VMEC and Boozer loaders fully onto the JAX
  geometry stack and refreshed the W7-X VMEC regression fixture.
  - What worked:
    - Replaced the direct VMEC `wout` parser in `src/ntx/vmec.py` with
      `vmec_jax.api.read_wout(...)`.
    - Replaced the direct Boozer `boozmn` parser in `src/ntx/booz.py` with
      `booz_xform_jax.Booz_xform.read_boozmn(...)`.
    - Confirmed that the old vendored W7-X `wout` fixture was not compatible
      with `vmec_jax` because it lacked the `chipf` data expected by the modern
      reader, then replaced it with the compatible W7-X `wout` from the local
      NEOPAX inputs.
    - Re-ran the focused regression gates on the new loader path:
      - `pytest -q tests/test_vmec.py tests/test_boozmn.py tests/test_vmec_regression.py tests/test_vmec_physics.py tests/test_vmec_qi.py tests/test_sfincs_vmec_geometry.py`
        -> `19 passed`
    - Verified that the `sfincs_jax` geometry comparison still closes on the
      VMEC path after the `vmec_jax` swap, so the new loader preserves the
      geometry conventions already used in the research workflow.
    - Added a dedicated `vmec_jax -> booz_xform_jax -> NTX` convergence check
      in `tests/test_vmec_jax_backend.py`.
    - Updated GitHub Actions so the CPU test matrix installs `vmec_jax` and
      `booz_xform_jax` from GitHub before running the NTX suite.
  - What did not:
    - The file-backed reduced-mode VMEC path and the explicit
      `vmec_jax -> booz_xform_jax -> NTX` Boozer-transform path are not
      transport-equivalent on the sampled W7-X case. The JAX transform path
      uses far fewer Boozer modes on the current `mboz/nboz` settings and still
      shows large `D11`, `D13`, and sign-level `D31` differences relative to
      the VMEC harmonic path.
    - That means the solver-side parity question is not yet closed for the
      imported JAX transform lane. The immediate remaining work is to raise the
      Boozer transform resolution and normalization audit there rather than to
      re-open the file readers.
- 2026-04-09: closed the handedness bug in the imported
  `vmec_jax -> booz_xform_jax -> NTX` Boozer-transform lane.
  - What worked:
    - Root-caused the remaining imported-JAX parity gap to the Boozer sign
      convention in `surface_from_vmec_jax_state(...)`.
    - The transformed surface carried the opposite signs for `iota` and
      `B_zeta` relative to the file-backed `boozmn` loader. Applying the same
      right-handed convention used in `src/ntx/booz.py` collapses the transport
      mismatch on the local W7-X Boozer reference.
    - Added a dedicated regression in `tests/test_vmec_jax_backend.py` that
      compares the imported JAX transform lane directly against
      `load_boozmn_surface(...)` on the W7-X `boozmn` reference at
      `13 x 17 x 16`, requiring all four transport channels to agree within
      `2%`.
    - Re-ran the focused parity suite:
      - `pytest -q tests/test_vmec_jax_backend.py tests/test_boozmn.py tests/test_neopax_adapter.py`
        -> `9 passed`
    - Re-evaluated the local NTX-to-NEOPAX W7-X subset after the sign fix:
      - max `|D11_log - reference|` on the sampled subset: about `0.979`
      - max relative `D33` difference on the sampled subset: about `10.7%`
      - max absolute `D13` difference on the sampled subset: about `1.42`
  - What did not:
    - This fix closes the Boozer handedness problem, but it does not eliminate
      the remaining W7-X subset spread against the NEOPAX reference tables.
      The open gap is now in the database/normalization side of that workflow,
      not in the imported JAX Boozer transform itself.
- 2026-04-09: changed the W7-X NEOPAX parity gate to Eduardo Neto's
  `vmec_neopax` REFERENCE_EXECUTABLE workflow and closed the reference subset mismatch.
  - What worked:
    - Cloned the local `reference_executable_edu` checkout on branch
      `vmec_neopax` at commit `27d4bc2` and audited:
      - `Examples/DKES_like_database/Test_Monoenergetic_database_VMEC_s_coordinate_W7X.py`
      - `reference_executable/_field.py::Field.from_vmec_s(...)`
    - Root-caused the important reference conventions:
      - use `xm_nyq`, `xn_nyq`
      - interpolate VMEC Fourier coefficients on the half grid
      - negate `iota`
      - negate the toroidal mode sign when mapping into NTX
      - keep `jacobian_cos = +gmnc`
      - use `B0 = max(abs(b_mnc))`
      - keep `transport_psi_scale = 1` for this comparison lane
      - map REFERENCE_EXECUTABLE `nl` to NTX `n_xi = nl - 1`
    - Added `src/ntx/vmec_reference_executable.py` with:
      - `load_vmec_surface_reference_executable_reference(...)`
      - `reference_executable_vmec_factors(...)`
    - Extended `src/ntx/neopax.py` with:
      - `build_reference_executable_reference_vmec_scan(...)`
      - `write_neopax_scan_hdf5(...)`
    - Added the NTX replacement script:
      `examples/DKES_like_database/Test_Monoenergetic_database_VMEC_s_coordinate_W7X.py`
    - Added `tests/test_reference_executable_reference_vmec.py`, covering:
      - VMEC geometry parity against Eduardo's `Field.from_vmec_s(...)`
      - a hard single-point transport parity check against
        `monoenergetic_dke_solve_internal(...)`
      - subset database parity against
        `tests/NEOPAX/tests/inputs/Dij_NEOPAX_FULL_S_NEW_W7X.h5`
      - example-script HDF5 generation
    - The W7-X subset now matches the existing NEOPAX reference HDF5 to better
      than `1e-2` relative error for `D11`, `D13`, `D31`, and `D33`.
  - What did not:
    - `python -m pip install -e ../tests/reference_executable_edu`
      is still not viable because Eduardo's fork is not packaged as an editable
      project. The local source checkout works through `PYTHONPATH`, which is
      how the parity tests currently run.
    - This closes the reference-comparison lane, not the fully JAX
      `vmec_jax -> booz_xform_jax -> NTX` W7-X NEOPAX lane. That lane still
      needs a separate normalization/convergence audit.
- 2026-04-09: added a direct `vmec_jax` VMEC-harmonic imported lane for the
  W7-X NEOPAX parity workflow and moved the parity gate onto that path.
  - What worked:
    - Audited the remaining imported-JAX mismatch and showed that the fully JAX
      `vmec_jax -> booz_xform_jax -> NTX` Boozer-transform lane is not the
      right parity gate for the local W7-X NEOPAX subset on `D11` and `D13`,
      even after the handedness fix.
    - Confirmed the direct VMEC harmonic interpretation is the stable parity
      path by building the surface from `vmec_jax.api.read_wout(...)` and
      matching the validated reference VMEC solve to roundoff at fixed
      `s`, `nu_hat`, and `epsi_hat`.
    - Added `src/ntx/vmec_jax_vmec.py` with:
      - `surface_from_vmec_jax_vmec_wout(...)`
      - `surface_from_vmec_jax_vmec_wout_file(...)`
    - Updated the imported NEOPAX-facing example and adapter tests to use this
      direct VMEC harmonic path:
      - `examples/neopax_with_ntx.py`
      - `tests/test_neopax_adapter.py`
      - `tests/test_vmec_jax_vmec.py`
    - The imported W7-X subset built through `vmec_jax` now matches the local
      NEOPAX reference subset to better than `1e-2` relative error for `D11`,
      `D13`, `D31`, and `D33`.
    - Focused validation passed:
      - `python -m ruff check src/ntx/vmec_jax_vmec.py src/ntx/__init__.py examples/neopax_with_ntx.py tests/test_neopax_adapter.py tests/test_vmec_jax_vmec.py`
      - `python -m pytest -q tests/test_vmec_jax_vmec.py tests/test_neopax_adapter.py`
        -> `4 passed`
  - What did not:
    - This does not make the `vmec_jax -> booz_xform_jax -> NTX` Boozer lane a
      W7-X NEOPAX parity path. That lane remains useful for direct Boozer and
      end-to-end transform workflows, but it needs its own convergence and role
      definition instead of being treated as interchangeable with the direct
      VMEC-harmonic NEOPAX path.
- 2026-04-09: added a callback-free imported NEOPAX scan path and a QI imported
  example/round-trip workflow.
  - What worked:
    - Added `build_ntx_neopax_scan_from_surfaces(...)` to accept an explicit
      tuple of NTX surfaces in memory instead of forcing imported workflows
      through a Python `surface_loader(float(rho))` callback.
    - Added `scan_to_neopax_arrays(...)` and the `NeopaxMonoenergeticArrays`
      container so the NEOPAX normalization step can stay in pure JAX arrays
      before constructing the external `NEOPAX.Monoenergetic` object.
    - Registered `NeopaxScan` and `NeopaxMonoenergeticArrays` as pytrees.
    - Added focused imported-path tests:
      - `tests/test_neopax_arrays.py`
      - `tests/test_neopax_qi.py`
    - Added `examples/qi_neopax_with_ntx.py` as a second-family imported VMEC
      example that:
      - builds a small QI scan from explicit in-memory surfaces
      - maps it through `scan_to_neopax_arrays(...)`
      - writes a NEOPAX-style HDF5 file
    - Focused validation passed:
      - `python -m ruff check src/ntx/neopax.py src/ntx/__init__.py tests/test_neopax_arrays.py tests/test_neopax_qi.py`
      - `python -m mypy src/ntx/neopax.py`
      - `python -m pytest -q tests/test_neopax_arrays.py tests/test_neopax_qi.py`
        -> `4 passed`
      - `python -m pytest -q tests/test_neopax_examples.py` -> `1 passed`
  - What did not:
    - This is an imported-API and repository-fixture closure, not an external
      QI parity closure. There is still no archived or NEOPAX reference QI
      database in the tree to use as an absolute cross-code target.
- 2026-04-09: fixed the CI failure from the new QI imported tests and scoped
  the Boozer-transform lane explicitly.
  - What worked:
    - Audited the failed GitHub Actions runs `24203941953` and `24203949237`
      and traced the breakage to macOS-specific absolute paths introduced in:
      - `tests/test_neopax_examples.py`
      - `tests/test_neopax_qi.py`
      - `examples/qi_neopax_with_ntx.py`
    - Replaced those hard-coded paths with repository-relative paths derived
      from `Path(__file__).resolve()`, making the new QI imported example and
      test portable across CI runners.
    - Strengthened the local Boozer-transform regression in
      `tests/test_vmec_jax_backend.py` so the
      `vmec_jax -> booz_xform_jax -> NTX` lane is checked against the file-backed
      `boozmn` transport reference at two operating points instead of one.
    - Updated the docs and plan so this lane is now explicitly treated as a
      separate validated Boozer-transform workflow, not as a hidden W7-X NEOPAX
      parity promise.
    - Validation that worked:
      - `python -m pytest -q tests/test_neopax_examples.py tests/test_neopax_qi.py`
        -> `2 passed`
      - `python -m pytest -q tests/test_vmec_jax_backend.py` -> `4 passed`
      - `python -m ruff check .`
      - `python -m mypy src/ntx`
      - `python -m pytest -q` -> `84 passed, 2 skipped`
      - `python -m sphinx -b html docs docs/_build/html`
  - What did not:
    - The GitHub-hosted runners still do not provide an external QI reference
      database, so the QI imported path remains a repository-backed round-trip
      validation rather than a cross-code parity gate.
- 2026-04-09: removed machine-specific absolute paths from the repo and added
  QA/QH/QI omnigenous validation families.
  - What worked:
    - Added `src/ntx/_checkout_paths.py` so tests, examples, and optional local
      integrations discover sibling checkouts through environment variables or
      workspace-relative defaults rather than hard-coded machine paths.
    - Updated the optional `vmec_jax`, `booz_xform_jax`, `NEOPAX`,
      `sfincs_jax`, and REFERENCE_EXECUTABLE integration points to use the new helper.
    - Added repository fixtures from the local `omnigenity_optimization` study
      set:
      - `tests/fixtures/wout_nfp3_QA_fixed_resolution_final.nc`
      - `tests/fixtures/boozmn_nfp3_QA_fixed_resolution_final.nc`
      - `tests/fixtures/wout_nfp3_QH_fixed_resolution_final.nc`
      - `tests/fixtures/boozmn_nfp3_QH_fixed_resolution_final.nc`
      - `tests/fixtures/wout_nfp3_QI_fixed_resolution_final.nc`
      - matching VMEC input files for QA, QH, and QI
    - Added `tests/test_omnigenity_cases.py`, covering:
      - QA/QH transform-vs-Boozer transport checks at two operating points
      - QI VMEC-harmonic parity against the comparison-reference loader
      - QI imported NEOPAX-array scan coverage
    - Focused validation passed:
      - `python -m pytest -q tests/test_omnigenity_cases.py tests/test_vmec_jax_backend.py tests/test_jax_neopax_examples.py tests/test_reference_executable_reference_vmec.py`
        -> `17 passed`
  - What did not:
    - QA and QH extend the transform-vs-Boozer validation lane, but they still
      do not have external NEOPAX-style reference databases in the repository,
      so they are not database-parity gates yet.
- 2026-04-09: relocated the working checkout to `local/NTX`, closed the QI
  `rho = 0.25` external-reference mismatch, and aligned the direct `vmec_jax`
  parity lane with the same interpolation convention.
  - What worked:
    - Moved the repository from `local/.NTX` to `local/NTX` and continued all
      validation from the new checkout path.
    - Verified the last pre-push GitHub Actions run from the relocated checkout
      was green before proceeding with new changes.
    - Traced the remaining QI `rho = 0.25` mismatch to the comparison-only VMEC
      reference loader, not to NTX's dense block-tridiagonal solve.
    - Updated `src/ntx/vmec_reference_executable.py` to use `interpax` cubic interpolation for
      VMEC half-grid mode tables and related radial profiles, matching Eduardo
      REFERENCE_EXECUTABLE.
    - Updated `src/ntx/vmec_jax_vmec.py` to use the same cubic interpolation for
      the direct `vmec_jax` imported parity lane.
    - Regenerated the vendored QA/QH/QI external REFERENCE_EXECUTABLE subset databases from
      the new checkout.
    - Added standalone example path bootstrapping so example scripts can be run
      directly from the repository without requiring an editable NTX install.
    - Validation passed from `local/NTX`:
      - `python -m ruff check .`
      - `python -m mypy src/ntx`
      - `python -m pytest -q` -> `94 passed, 2 skipped`
      - `python -m sphinx -b html docs docs/_build/html`
  - What did not:
    - The GitHub-hosted workflow still emits the upstream Node 20 deprecation
      warning for `actions/checkout@v4` and `actions/setup-python@v5`; this is
      not an NTX code failure, but the workflow should be bumped when the
      action authors publish their next stable versions.

- 2026-04-10: added the first shipping-oriented package and release lane.
  - What worked:
    - Added shipping metadata to `pyproject.toml`, including classifiers,
      keywords, package URLs, and build/test-publish development tools.
    - Added `src/ntx/__main__.py` so the installed package supports both:
      - `ntx ...`
      - `python -m ntx ...`
    - Added release-facing repository files:
      - `LICENSE`
      - `CHANGELOG.md`
      - `docs/release.md`
    - Added CI workflows:
      - `tests.yml` updated to `actions/checkout@v5` and `actions/setup-python@v6`
      - `package.yml` for wheel/sdist build, `twine check`, and clean-install
        smoke tests across Python `3.10` to `3.12`
      - `release.yml` for tag-driven GitHub release artifact publishing
    - Added shipping tests in `tests/test_packaging.py`.
    - Local shipping validation passed:
      - `python -m ruff check .`
      - `python -m mypy src/ntx`
      - `python -m pytest -q` -> `96 passed, 2 skipped`
      - `python -m sphinx -b html docs docs/_build/html`
      - `python -m build`
      - `python -m twine check dist/*`
      - clean wheel install smoke test
      - clean sdist install smoke test
  - What did not:
    - Nothing substantive in the shipping lane after the patch. The next work
      is broader release hardening, not a broken packaging path.
