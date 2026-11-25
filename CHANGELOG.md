# Changelog

Release notes for tagged versions live in `docs/release-notes-*.md`.

## Unreleased

- `ntx._interp` evaluates every cubic Hermite through SOLVAX (`cubic_spline`, `cubic_spline_evaluate`) and takes its Akima and PCHIP slopes from SOLVAX; NTX keeps only its mean-of-secants and parabolic slope rules. SOLVAX floor raised to 0.37.0. Values agree with the previous code within 5e-14 relative inside the grid (4e-13 extrapolated in 2-D).
- `examples/cross_code_conversion.py` writes out the factors between NTX, SFINCS and the DKES-scaled tables of YANCC and MONKES (collisionality, radial `dpsi/dr`, `E_r`, Beidler `D*`, SFINCS hat units) and prints a table of NTX against the committed SFINCS RHSMode 3 decks with relative errors against a 2% gate; a live solve on the sample surface checks the inline factors against the ones NTX stores. Measured against the committed ladder, every coefficient agrees within 0.72% at rho = 0.3 and 0.5.
- Python 3.11 is the minimum (VMEX needs it); CI runs 3.13 with the latest JAX and checks 3.11 and 3.12.
- SOLVAX floor raised to 0.28.2; the factorization narrows its return type explicitly.
- VMEX and booz_xform_jax are optional extras (`ntx[vmex]`, `ntx[booz]`) installed from PyPI in CI.
- Committed artifacts and examples no longer carry machine-specific paths: producers write repo-relative paths or `checkout:relative` labels via `ntx._checkout_paths.portable_path`, and a test rejects home and temp paths in tracked files.
- JAX's persistent compilation cache is on by default (zero compile-time threshold); `NTX_DISABLE_COMPILATION_CACHE=1` turns it off.
