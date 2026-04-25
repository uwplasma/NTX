# NTX 0.2.1 Release Notes

NTX `0.2.1` is a validation and documentation hardening release after the first
PyPI publication.

## Shipping Scope

This release keeps the public package surface compatible with `0.2.0` while
refreshing the research artifacts used by the README, docs, benchmark matrix,
and manuscript notes. Optional geometry-coupled workflows still require local
`vmec_jax`, `booz_xform_jax`, and imported workflow checkouts.

## Validation Claims

Promoted release claims:

- monoenergetic validation summary with Legendre convergence and Onsager gates
- fixed-field Redl/SFINCS interior agreement on the precise-QS QA/QH family
- fixed-field `NTX+NEOPAX` reduced-closure total-current stress gate below
  `1e-1`, with no fitted bridge constants
- W7-X integrated imported-workflow transfer on the rebuilt raw branch
- prepared derivative-path agreement against direct reverse-mode

Monitored stress diagnostics:

- species-resolved fixed-field closure parity remains future work
- VMEC geometry-family `D11/D31/D33` convergence breadth scan
- explicit-relaxed geometry/current derivative stress artifacts
- implicit-equilibrium derivative diagnostic remains non-shipping
- CPU/GPU production and fixed-workload strong-scaling performance maps

## User-Facing Changes

- The README fixed-field bootstrap-current figure now shows only overlaid
  SFINCS, Redl, and `NTX+NEOPAX` current profiles for QA/QH. Relative-error
  diagnostics remain in the JSON artifact and physics-gate tests.
- `examples/bootstrap_current_fixed_field_validation.py` remains the
  archive-backed QA/QH example for generating that figure and validating both
  Redl and `NTX+NEOPAX` below the `1e-1` interior gate.
- `examples/bootstrap_current_with_neopax.py` writes the selected `D33` branch
  to JSON and uses the corrected bootstrap-current assembly without adding the
  no-momentum and already-corrected momentum branches together.

## Release Checks

Verified locally on 2026-04-25 before tagging:

- `python -m ruff check .`
- `python -m mypy src/ntx`
- `python -m pytest -q`: `331 passed, 5 skipped`
- `python -m sphinx -b html docs docs/_build/html`
- `python -m build`
- `python -m twine check dist/*`
- clean-venv wheel smoke test for `ntx --help`, `python -m ntx --help`, and
  importing `GridSpec`

## Tagging

The published tag is `v0.2.1`.
