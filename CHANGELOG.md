# Changelog

Release notes for tagged versions live in `docs/release-notes-*.md`.

## Unreleased

- `load_geiger_boozer_surface` reads the Geiger/IPP "CC Boozer-coordinate data file" (the SFINCS/DKX `.bc` format, ten-number header with trailing text), interpolating linearly in `s` between the bracketing surfaces.
- Fix: Boozer harmonics beyond the grid Nyquist (`|m| > n_theta / 2` or `|n| > n_zeta / 2`) are dropped (`ntx.geometry.truncate_to_grid`) instead of aliased onto lower harmonics, as SFINCS and DKX do. A file with many weak high harmonics (the hydra_Np04 file has `m, n` up to 36) gave `D11` that changed by tens of percent between `17 x 41` and `25 x 61` at `rho = 0.875` and at `efield > 0` was 2 to 4 times DKX and DKES; with the truncation it is grid independent and within 1% of DKX.
- Validation: NTX against the IPP DKES results of the HELIAS reactor hydra_Np04 at seven radii (`scripts/validate_dkes_hydra_np04.py`, `docs/validation.md`). The DKES data and the Boozer file are private (`NTX_DKES_HYDRA_DIR`, `NTX_HYDRA_BOOZER`): the committed record `benchmarks/results/dkes_hydra_np04.json` and the tests keep agreement statistics only and skip without the data.
- Python 3.11 is the minimum (VMEX needs it); CI runs 3.13 with the latest JAX and checks 3.11 and 3.12.
- SOLVAX floor raised to 0.28.2; the factorization narrows its return type explicitly.
- VMEX and booz_xform_jax are optional extras (`ntx[vmex]`, `ntx[booz]`) installed from PyPI in CI.
- Committed artifacts and examples no longer carry machine-specific paths: producers write repo-relative paths or `checkout:relative` labels via `ntx._checkout_paths.portable_path`, and a test rejects home and temp paths in tracked files.
- JAX's persistent compilation cache is on by default (zero compile-time threshold); `NTX_DISABLE_COMPILATION_CACHE=1` turns it off.
