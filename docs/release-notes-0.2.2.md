# NTX 0.2.2 Release Notes

NTX `0.2.2` is a CI and release-hardening patch on top of the `0.2.1`
validation release.

## Shipping Scope

This release preserves the public package API and the promoted `0.2.1`
validation scope. It refreshes the robust bootstrap-current optimization stress
artifact and fixes the corresponding CI smoke gate.

## User-Facing Changes

- `examples/bootstrap_current_robust_optimization.py` now writes
  `robust_objective_relative_change` separately from the signed
  `weighted_current_ratio`.
- The retained `robust_gain` JSON key is now explicitly defined as the
  optimized weighted-current proxy normalized by the absolute baseline
  weighted-current proxy.
- The robust optimization smoke test now verifies artifact/schema and finite
  uncertainty metrics, while objective improvement remains covered by the
  dedicated autodiff unit test.

## Release Checks

Verified locally on 2026-04-25 before tagging:

- `python -m ruff check .`
- `python -m mypy src/ntx`
- `python -m pytest -q`: `331 passed, 5 skipped`
- `python scripts/test_lane_manifest.py core_robust_bootstrap_workflow > /tmp/ntx-tests.txt && xargs pytest -q -m "not gpu" --cov=src/ntx --cov-report= < /tmp/ntx-tests.txt`
- `python scripts/check_physics_gates.py`
- `python -m sphinx -b html docs docs/_build/html`
- `python -m build --outdir /tmp/ntx-dist-0.2.2`
- `python -m twine check /tmp/ntx-dist-0.2.2/*`
- clean-venv wheel smoke test for `ntx --help`, `python -m ntx --help`, and
  importing `GridSpec`

## Tagging

The published tag is `v0.2.2`.
