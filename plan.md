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
