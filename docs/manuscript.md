# Manuscript Figures

NTX now includes a manuscript-ready figure bundle built directly from repository
examples.

## Recommended Figure Set

1. `validation_summary.{png,pdf}`
   - transport-curve behavior on the sample DKES-style and VMEC surfaces
   - Onsager closure
   - Legendre convergence
2. `autodiff_inverse_problem.{png,pdf}`
   - inverse recovery of a surface harmonic from synthetic transport data
3. `autodiff_neopax_profiles.{png,pdf}`
   - autodiff-based profile inversion on NEOPAX-style arrays
4. `bootstrap_current_optimization.{png,pdf}`
   - science/application figure for differentiable bootstrap-current
     optimization
5. `performance_scaling_smoke.{png,pdf}`
   - CPU/GPU scaling on the repository smoke grid
6. `performance_scaling_heavy.{png,pdf}`
   - heavier-grid scaling where throughput effects are visible

Together these figures cover:

- formulation and numerical behavior
- validation and convergence
- differentiable inverse and profile problems
- a science-facing bootstrap-current optimization workflow
- practical performance guidance

## One-Command Figure Bundle

```bash
python examples/make_publication_figures.py
```

This writes the full figure set into `docs/_static/` and also creates:

```text
docs/_static/publication_figure_manifest.json
```

Generate a subset:

```bash
python examples/make_publication_figures.py --figures validation,science
```

## Science Figure

```bash
python examples/bootstrap_current_optimization.py
```

The science/application figure is written to:

```text
docs/_static/bootstrap_current_optimization.png
docs/_static/bootstrap_current_optimization.pdf
```

It uses:

- a VMEC-derived radial surface family
- a dominant non-axisymmetric harmonic as the control parameter
- a weighted bootstrap-current proxy based on the current-response coefficients
- JAX autodiff to optimize that control directly

This is the recommended figure for a paper focused on differentiable bootstrap
current analysis and optimization with NTX.
