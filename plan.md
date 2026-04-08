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
- [ ] Push the current scan and GPU performance gains through the full office workflow.

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
  - Python: `3.10.12`
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
