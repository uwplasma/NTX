# Changelog

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
