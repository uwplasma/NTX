# Examples

This page lists the main ways to run NTX, from the smallest CLI solve to the
publication-figure scripts.

## 1. Simplest CLI Run

```bash
ntx examples/example_surface.toml
```

This is the smallest end-to-end solve. It requires no external files and is the
best first command to confirm that NTX is installed correctly.

## 2. DKES-Style CLI Run

```bash
ntx examples/sample_dkes.toml
```

This writes a compressed `.npz` result under `examples/outputs/`.

## 3. VMEC CLI Run

```bash
ntx examples/sample_vmec.toml
```

This exercises the VMEC normalization path on the bundled sample `wout` file.

## 4. Open And Plot An Output File

```bash
python examples/plot_output_npz.py examples/outputs/sample_dkes.npz
```

This reads an NTX `.npz` payload and writes:

- `docs/_static/output_file_summary.png`
- `docs/_static/output_file_summary.pdf`

The output figure contains:

- magnetic-field strength on the angular grid
- the radial-drift source on the same grid
- the solved transport coefficients
- a run-summary panel with key diagnostics

## 5. Python Single-Case Solve

```python
from ntx import GridSpec, MonoenergeticCase, load_vmec_surface, solve_monoenergetic

surface = load_vmec_surface("wout.nc", psi_n=0.25)
grid = GridSpec(n_theta=9, n_zeta=11, n_xi=12)
case = MonoenergeticCase(nu_hat=1e-3, er_hat=1e-3)
result = solve_monoenergetic(surface, grid, case)
```

## 6. NEOPAX Mapping

```bash
python examples/neopax_with_ntx.py
```

This example:

- loads a VMEC equilibrium
- builds an NTX monoenergetic scan
- maps that scan into NEOPAX-style arrays

Use it as the minimal reference for NTX-to-NEOPAX coupling.

## 7. Bootstrap Current From VMEC Or Boozmn

```bash
python examples/bootstrap_current_from_vmec_or_boozmn.py
```

This example is the most direct answer to the common workflow:

- start from a VMEC `wout` file and use `vmec_jax`
- or, if a Boozer `boozmn` file already exists, use `booz_xform_jax` output directly
- build an NTX monoenergetic database
- map that database into NEOPAX and compute a bootstrap-current profile
- compare the result against a local reference database and, when available, a
  local SFINCS-JAX `transportMatrix` output

Use `--surface-source vmec` to force the VMEC-harmonic lane or
`--surface-source boozmn` to force the direct Boozer lane when both files are
available.

## 8. Autodiff Inverse Problem

```bash
python examples/autodiff_inverse_problem.py
```

This writes `docs/_static/autodiff_inverse_problem.{png,pdf}` and demonstrates
recovery of a Boozer harmonic from synthetic transport data using JAX
gradients.

## 9. Autodiff NEOPAX Profiles

```bash
python examples/neopax_autodiff_profiles.py
```

This writes `docs/_static/autodiff_neopax_profiles.{png,pdf}` and demonstrates
a low-dimensional electric-field profile inversion on NEOPAX-style
monoenergetic arrays.

## 10. Science Case: Bootstrap-Current Optimization

```bash
python examples/bootstrap_current_optimization.py
```

This writes `docs/_static/bootstrap_current_optimization.{png,pdf}` and shows a
differentiable geometry-control problem:

- a VMEC-derived radial surface family
- one dominant non-axisymmetric harmonic used as the control variable
- autodiff optimization of a weighted bootstrap-current proxy
- explicit serial-versus-multiprocess timing annotations

This is the main application/science-case figure for a methods paper centered
on bootstrap-current analysis and optimization.

## 11. Performance Scaling

```bash
python examples/performance_scaling.py --cpu-json ... --gpu-json ...
```

This writes publication-style CPU/GPU scaling figures from benchmark JSON
payloads.

## 12. Validation Summary

```bash
python examples/validation_summary.py
```

This writes `docs/_static/validation_summary.{png,pdf}`. It is the recommended
core validation figure for a methods paper because it combines transport
trends, Onsager closure, and Legendre convergence.

## 13. Full Publication Bundle

```bash
python examples/make_publication_figures.py
```

This regenerates the manuscript-ready figure bundle and writes a manifest to
`docs/_static/publication_figure_manifest.json`.
