# Changelog

Release notes for tagged versions live in `docs/release-notes-*.md`.

## Unreleased

- Each public module now owns its implementation: `solver`, `neopax`, `autodiff`, `geometry` and `inputfiles` replace their private twins, and the VMEC wout readers plus the VMEX state and Boozer workflows live in one `ntx.vmec` (`ntx.vmex_vmec`, `ntx.vmex_backend` and `ntx._vmex` are gone; top-level `ntx` names are unchanged). `example_neopax_profile_autodiff` no longer takes a `maybe_import_neopax` callback.
- The benchmark matrix, physics gates and finite-beta audits moved out of the package to `tools/ntx_validation` (not installed); `ntx.validation`, `ntx.physics_gates` and the gate names on `ntx` are removed.
- Python 3.11 is the minimum (VMEX needs it); CI runs 3.13 with the latest JAX and checks 3.11 and 3.12.
- SOLVAX floor raised to 0.28.2; the factorization narrows its return type explicitly.
- VMEX and booz_xform_jax are optional extras (`ntx[vmex]`, `ntx[booz]`) installed from PyPI in CI.
- Committed artifacts and examples no longer carry machine-specific paths: producers write repo-relative paths or `checkout:relative` labels via `ntx._checkout_paths.portable_path`, and a test rejects home and temp paths in tracked files.
- JAX's persistent compilation cache is on by default (zero compile-time threshold); `NTX_DISABLE_COMPILATION_CACHE=1` turns it off.
