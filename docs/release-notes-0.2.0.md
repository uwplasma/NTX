# NTX 0.2.0 Release Notes

NTX `0.2.0` is the first PyPI release after the existing GitHub-only `v0.1.0`
tag.

## Shipping Scope

This release ships the repository-owned monoenergetic NTX package, CLI,
file-backed inputs, NEOPAX array/database bridge, validation artifacts,
autodiff examples, and documentation. Optional geometry-coupled workflows using
`vmex` and `booz_xform_jax` remain documented external installs until those
upstream packages are available from standard package indexes under stable
constraints.

## Validation Claims

Promoted release claims:

- monoenergetic validation summary with Legendre convergence and Onsager gates
- fixed-field Redl/SFINCS interior agreement on the precise-QS benchmark family
- W7-X integrated imported-workflow transfer on the rebuilt raw branch
- prepared derivative-path agreement against direct reverse-mode

Monitored stress diagnostics:

- fixed-field `NTX+NEOPAX` reduced-closure current comparison
- VMEC geometry-family `D11/D31/D33` convergence breadth scan
- geometry-control and boundary-current derivative stress artifacts
- implicit-equilibrium derivative diagnostic, closed as non-shipping
- prepared-geometry and compiled-solver reuse performance profile

## Release Checks

Verified locally on 2026-04-24:

- `python -m ruff check .`
- `python -m mypy src/ntx`
- `python -m pytest -q`: `320 passed, 5 skipped`
- `python -m sphinx -b html docs docs/_build/html`
- `python -m build`
- `python -m twine check dist/*`
- clean-venv wheel smoke for `ntx --help`, `python -m ntx --help`, and
  importing `GridSpec`

The release commit `c98ec8c` had green GitHub `tests` and `package` workflows
on `main`. The `v0.2.0` tag workflows were green for `tests`, `package`, and
`release`, and the release workflow published the wheel and sdist to PyPI
through Trusted Publishing.

## Tagging

The published tag is `v0.2.0`.
