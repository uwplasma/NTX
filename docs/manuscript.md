# Manuscript Figures

NTX now includes a complete figure bundle for a methods-focused manuscript built
directly from repository examples.

## Recommended Figure Set

1. `validation_summary.{png,pdf}`
   - transport-curve behavior on the sample DKES-style and VMEC surfaces
   - Onsager residual over the collisionality scan
   - Legendre-resolution convergence of the low-order coefficients
2. `autodiff_inverse_problem.{png,pdf}`
   - inverse recovery of a Boozer harmonic from synthetic transport data
3. `autodiff_neopax_profiles.{png,pdf}`
   - autodiff-based profile inversion on NEOPAX-style arrays
4. `performance_scaling_smoke.{png,pdf}`
   - CPU/GPU scaling on the repository smoke grid
5. `performance_scaling_heavy.{png,pdf}`
   - CPU/GPU scaling on a heavier grid where throughput effects are visible

Together these figures are sufficient to start drafting a methods manuscript:

- formulation and numerical behavior
- validation and convergence
- differentiable inverse-problem workflows
- NEOPAX-facing profile analysis
- practical performance guidance

The only figure that remains optional rather than mandatory is an
application-specific science case. That depends on the paper target, not on the
NTX methods core.

## One-Command Figure Bundle

```bash
python examples/make_publication_figures.py
```

This writes the full figure set into `docs/_static/` and also creates:

```text
docs/_static/publication_figure_manifest.json
```

Use `--figures` to generate a subset:

```bash
python examples/make_publication_figures.py --figures validation,performance_heavy
```

## Validation Figure

```bash
python examples/validation_summary.py
```

The validation figure is written to:

```text
docs/_static/validation_summary.png
docs/_static/validation_summary.pdf
```

It is intended to be the first figure in a methods paper because it combines
coefficient trends, Onsager closure, and spectral convergence in one panel set.
