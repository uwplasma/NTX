# Changelog

Release notes for tagged versions live in `docs/release-notes-*.md`.

## Unreleased

- Validation: `tests/fixtures/beidler2011_w7x_monoenergetic.csv` holds D11*, D31* and D33* of the W7-X standard configuration at rho = 0.5 digitized from Figs 17, 21 and 26 of Beidler et al. (2011) (errors in its header: 3% on D11*, 0.003 on D31*, 0.002 on D33*). `tests/test_beidler_w7x_reference.py` solves the 52-harmonic surface `tests/fixtures/w7x_standard_rho05.ddkes2.data` against it: D11* within +1.5 to +7.0%, D31* within 0.007, D33* within 1.6% at v_E = 0, and D11* at v_E = 3e-3 inside the scatter of the DKES and Monte Carlo symbols.
- Python 3.11 is the minimum (VMEX needs it); CI runs 3.13 with the latest JAX and checks 3.11 and 3.12.
- SOLVAX floor raised to 0.28.2; the factorization narrows its return type explicitly.
- VMEX and booz_xform_jax are optional extras (`ntx[vmex]`, `ntx[booz]`) installed from PyPI in CI.
- Committed artifacts and examples no longer carry machine-specific paths: producers write repo-relative paths or `checkout:relative` labels via `ntx._checkout_paths.portable_path`, and a test rejects home and temp paths in tracked files.
- JAX's persistent compilation cache is on by default (zero compile-time threshold); `NTX_DISABLE_COMPILATION_CACHE=1` turns it off.
