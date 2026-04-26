# Changelog

## 0.2.2

CI and release hardening after the `0.2.1` validation release.

Highlights:

- robust bootstrap-current optimization artifacts now separate robust-objective
  improvement from the signed weighted-current ratio
- the robust example smoke test now checks artifact/schema and finite
  uncertainty metrics instead of requiring improvement from a two-step toy run
- refreshed the robust-design figure artifact and JSON metadata for the docs

## 0.2.1

Validation, documentation, and performance-artifact hardening after the first
PyPI release.

Highlights:

- fixed-field QA/QH bootstrap-current validation figure is now an overlay-only
  SFINCS, Redl, and `NTX+NEOPAX` current comparison, with the `<1e-1`
  interior-window gates retained in JSON and tests
- `examples/bootstrap_current_with_neopax.py` keeps the corrected
  no-momentum/default current assembly and records the selected `D33` branch in
  the generated summary
- fixed-field closure diagnostics, benchmark matrix, manuscript artifacts, and
  physics-gate docs were refreshed from committed artifacts
- solver, profile, and bootstrap-autodiff internals were split into smaller
  ownership modules without changing public APIs
- production and fixed-workload strong-scaling performance artifacts were added
  for CPU/GPU and multiprocess/device-parallel lanes
- README and example documentation were tightened around the promoted claims,
  monitored stress gates, and remaining research lanes

## 0.2.0

Release-candidate update for the first PyPI publication.

Highlights:

- artifact-backed W7-X integrated transfer and fixed-field Redl/SFINCS release
  gates
- fixed-field `NTX+NEOPAX` retained as a reduced-closure stress diagnostic,
  not a parity claim
- geometry-family `D11/D31/D33` convergence stress figure across local public
  VMEC example families
- prepared-geometry and compiled-solver reuse performance profile for repeated
  fixed-geometry scans
- explicit-relaxed `vmec_jax -> booz_xform_jax -> NTX` derivative stress lane
  retained as the promoted differentiable equilibrium path
- implicit-equilibrium derivative path closed as a non-shipping diagnostic
- refreshed benchmark matrix, physics-gate registry, manuscript artifact
  bundle, and publication figure manifest
- split-lane CI, package workflow, docs build, wheel/sdist smoke checks, and
  tag-gated release workflow

## 0.1.0

Initial NTX release for research use.

Highlights:

- Escoto-style monoenergetic Legendre-space solver in JAX
- DKES and VMEC surface support
- direct `ntx input.toml` execution path with Rich terminal output
- `.npz` output bundles with coefficients, metadata, diagnostics, and modes
- imported differentiable solve and scan APIs
- NEOPAX mapping helpers and reference-database workflows
- artifact-backed monoenergetic, fixed-field Redl/SFINCS, W7-X transfer,
  closure-stress, geometry-control, boundary-derivative, and performance
  validation reports
- maintained benchmark matrix, physics-gate registry, manuscript artifact
  bundle, and publication figure manifest
- geometry-family breadth summary that keeps unresolved implicit and broader
  W7-X/QI/omnigenous lanes scoped as future work
- CPU CI, docs CI, and packaging/release validation workflows
