# NTX JAX Neoclassical Transport Plan

## Summary

Build `NTX` as a new JAX-native neoclassical transport code in
`/Users/rogeriojorge/local/.NTX`, with a private GitHub repo at
`github.com/uwplasma/NTX`. The implementation is based on Escoto's Legendre-space
monoenergetic DKE formulation from arXiv:2510.27513. The existing local
`/Users/rogeriojorge/local/tests/REFERENCE_EXECUTABLE` checkout is used only as an external
numerical benchmark.

Important environment facts found during planning:

- Local thesis PDF exists at `/Users/rogeriojorge/local/tests/Escoto_Thesis.pdf`.
- Local benchmark checkout exists at `/Users/rogeriojorge/local/tests/REFERENCE_EXECUTABLE`,
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

## Work Log

### 2026-04-08

- Confirmed the existing VMEC path still used a placeholder `transport_psi_scale = 1.0`,
  which kept `epsi_hat` runs workable but made `er_hat` unsupported and left the VMEC
  transport normalization under-specified.
- Audited the local `sfincs_jax` VMEC radial conversions. The relevant derivative factors
  are in `/Users/rogeriojorge/local/tests/sfincs_jax/sfincs_jax/io.py`, where
  `dpsi_hat/dr_hat = 2 * psi_a_hat * sqrt(psi_n) / a_hat` and
  `dr_hat/dpsi_hat` is its reciprocal.
- Verified the local QI VMEC candidates in
  `/Users/rogeriojorge/local/tests/sfincs_jax/examples/additional_examples/` are
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
  `/Users/rogeriojorge/local/.NTX/tests/fixtures/wout_QI_nfp2_stable_Er_006_000043_hires_scaled.nc`.
- Confirmed the QI fixture solves cleanly on the current VMEC path with:
  - `psi_n = 0.12247^2`
  - `nfp = 2`
  - `loaded_mode_count = 72`
  - `transport_psi_scale = 0.9673631438898428`
- Added a QI VMEC example input around the new `er_hat` path:
  `/Users/rogeriojorge/local/.NTX/examples/qi_vmec_erhat.toml`.
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
  - `/Users/rogeriojorge/local/.NTX/scripts/benchmark_against_reference_executable.py`
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
  - `gfortran`: `/home/rjorge/miniforge3/envs/qh-gpu/bin/gfortran`
  - NetCDF Fortran include: `/home/rjorge/miniforge3/envs/qh-gpu/include`
  - NetCDF Fortran lib: `/home/rjorge/miniforge3/envs/qh-gpu/lib`
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
