# Examples

## Built-In Surface

```bash
ntx examples/example_surface.toml
```

This is the smallest end-to-end run and does not require any external files.

## DKES-Style Surface

```bash
ntx examples/sample_dkes.toml
```

This uses the repository sample `ddkes2.data` fixture and writes a `.npz`
result under `examples/outputs/`.

## VMEC Surface

```bash
ntx examples/sample_vmec.toml
```

This uses the repository sample `wout` fixture and exercises the VMEC
normalization path.

## Python API

```python
from ntx import GridSpec, MonoenergeticCase, load_vmec_surface, solve_monoenergetic

surface = load_vmec_surface("wout.nc", psi_n=0.25)
grid = GridSpec(n_theta=9, n_zeta=11, n_xi=12)
case = MonoenergeticCase(nu_hat=1e-3, er_hat=1e-3)
result = solve_monoenergetic(surface, grid, case)
```

## NEOPAX Mapping

```bash
python examples/neopax_with_ntx.py
```

This example:

- loads a small NEOPAX-style HDF5 table
- builds an NTX VMEC scan from the sample `wout`
- maps the result into NEOPAX-style arrays

Use it as the minimal reference for NTX-to-NEOPAX coupling.

## Autodiff Inverse Problem

```bash
python examples/autodiff_inverse_problem.py
```

This writes `docs/_static/autodiff_inverse_problem.png` and shows recovery of a
surface harmonic from synthetic transport data using JAX gradients. A matching
PDF is also written for manuscript workflows.

## Autodiff NEOPAX Profiles

```bash
python examples/neopax_autodiff_profiles.py
```

This writes `docs/_static/autodiff_neopax_profiles.png` and shows a
low-dimensional electric-field profile inversion on NEOPAX-style arrays. A
matching PDF is also written for manuscript workflows.
