# Changelog

Release notes for tagged versions live in `docs/release-notes-*.md`.

## Unreleased

- The implicit adjoint contracts its Legendre rows in a `lax.scan` instead of an unrolled Python loop: compile time no longer grows with `n_xi`. W7-X 19x41x48 gradient on an A4000: cold 39-41 s to 24-30 s, warm unchanged (0.93 s).
- Python 3.11 is the minimum (VMEX needs it); CI runs 3.13 with the latest JAX and checks 3.11 and 3.12.
- SOLVAX floor raised to 0.28.2; the factorization narrows its return type explicitly.
- VMEX and booz_xform_jax are optional extras (`ntx[vmex]`, `ntx[booz]`) installed from PyPI in CI.
- Committed artifacts and examples no longer carry machine-specific paths: producers write repo-relative paths or `checkout:relative` labels via `ntx._checkout_paths.portable_path`, and a test rejects home and temp paths in tracked files.
- JAX's persistent compilation cache is on by default (zero compile-time threshold); `NTX_DISABLE_COMPILATION_CACHE=1` turns it off.
