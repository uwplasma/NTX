# NTX 0.2.3 Release Notes

NTX `0.2.3` is an output, plotting, and runtime-path patch on top of the
`0.2.2` release.

## Shipping Scope

This release keeps the public solver APIs stable while making file-backed runs
more useful for research workflows. TOML and CLI runs now write NetCDF by
default, can select NetCDF/NPZ/HDF5 from the output filename, and can generate a
summary plot panel directly from the saved payload.

## User-Facing Changes

- `ntx input.toml --plot` writes the run output and a PDF summary panel next to
  it.
- TOML input files can use `[output] path = "case.nc"`; `.nc`, `.npz`, `.h5`,
  and `.hdf5` suffixes select the writer.
- Python callers can use `run_from_input_file(..., output_path=..., plot=True)`
  or `save_run_output(...)` for the same suffix-selected output behavior.
- NetCDF support is now installed with the base package through `netCDF4`.
- File-backed runs reuse one prepared geometry/operator system for reporting,
  solving, and output writing.
- The docs now introduce the local monoenergetic drift-kinetic equation before
  the projected Legendre block system.

## Release Checks

Verified locally on 2026-04-26 before tagging:

- `python -m ruff check .`
- `python -m mypy src/ntx`
- `python -m pytest -q`: `336 passed, 5 skipped`
- `python -m sphinx -b html docs docs/_build/html`
- `python -m build --outdir /tmp/ntx-dist-0.2.3`
- `python -m twine check /tmp/ntx-dist-0.2.3/*`
- remote `office` GPU smoke regression and GPU test subset

## Tagging

The published tag is `v0.2.3`.
