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
4. `derivative_path_benchmark.{png,pdf}`
   - prepared-derivative timing and agreement against direct reverse-mode
5. `bootstrap_current_optimization.{png,pdf}`
   - science/application figure for differentiable bootstrap-current
     optimization
6. `bootstrap_current_from_vmec_or_boozmn.{png,pdf}`
   - NTX-only bootstrap-current-proxy profile from VMEC/Boozer input
7. `bootstrap_current_reference_audit_w7x.{png,pdf}`
   - W7-X imported-workflow bootstrap-current convergence audit
8. `performance_scaling_smoke.{png,pdf}`
   - CPU/GPU scaling on the repository smoke grid
9. `performance_scaling_heavy.{png,pdf}`
   - heavier-grid scaling where throughput effects are visible
10. `ambipolar_profile.{png,pdf}`
   - profile-grade ambipolar electric-field solve and bootstrap-current proxy
11. `ambipolar_profile_family.{png,pdf}`
   - control-parameter family of ambipolar closures and scalar bootstrap-current objective
12. `profile_control_optimization.{png,pdf}`
   - differentiable optimization of a scalar profile control on top of the ambipolar closure
13. `profile_basis_optimization.{png,pdf}`
   - low-dimensional radial-basis optimization of the same profile closure
14. `profile_transport_loop.{png,pdf}`
   - explicit self-consistent transport-relaxation iteration on the same profile closure
15. `primitive_profile_transport.{png,pdf}`
   - primitive density/temperature transport iteration mapped back to ambipolar-field and bootstrap-current evolution

Together these figures cover:

- formulation and numerical behavior
- validation and convergence
- differentiable inverse and profile problems
- derivative cost for prepared optimization workflows
- a science-facing bootstrap-current optimization workflow
- a pure NTX radial-profile figure
- a profile-grade ambipolar and bootstrap-current-proxy workflow
- a control-parameter family view of the same profile-grade closure
- a direct optimization view of the profile-grade closure
- a low-dimensional multi-parameter version of that optimization
- a self-consistent transport-relaxation view of the same closure
- a primitive-profile transport view with positive density and temperature updates
- a W7-X imported-workflow convergence figure
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

## Prepared-Derivative Efficiency Figure

```bash
python examples/derivative_path_benchmark.py
```

This writes:

```text
docs/_static/derivative_path_benchmark.png
docs/_static/derivative_path_benchmark.pdf
```

Use this figure when the paper needs an explicit statement of how NTX moves from
plain reverse-mode to a prepared differentiable workflow that is better suited
to repeated optimization scans.

## NTX Bootstrap-Current Proxy Figure

```bash
python examples/bootstrap_current_from_vmec_or_boozmn.py
```

This writes:

```text
docs/_static/bootstrap_current_from_vmec_or_boozmn.png
docs/_static/bootstrap_current_from_vmec_or_boozmn.pdf
docs/_static/bootstrap_current_from_vmec_or_boozmn.json
```

It is the recommended figure when the paper needs a compact NTX-only radial
profile panel without bringing in the external database workflow. The panel
stays close to directly interpretable quantities: geometry, profile inputs,
parallel-flow drive, and the resulting interior bootstrap-current proxy built
from analytic profile gradients.

![NTX bootstrap-current proxy profile](_static/bootstrap_current_from_vmec_or_boozmn.png)

## Ambipolar Profile Figure

```bash
python examples/ambipolar_profile.py
```

This writes:

```text
docs/_static/ambipolar_profile.png
docs/_static/ambipolar_profile.pdf
```

Use this figure when the paper needs a profile-grade closure panel built
entirely from NTX scan data, including the ambipolar residual landscape over
the scanned `E_r` axis and the resulting bootstrap-current proxy.

![Ambipolar profile](_static/ambipolar_profile.png)

## Ambipolar Profile Family Figure

```bash
python examples/ambipolar_profile_family.py
```

This writes:

```text
docs/_static/ambipolar_profile_family.png
docs/_static/ambipolar_profile_family.pdf
```

Use this figure when the paper needs an optimization-facing profile figure that
shows how a scalar control changes the residual landscape and the
bootstrap-current proxy profiles, while also exposing a one-dimensional
objective landscape.

![Ambipolar profile family](_static/ambipolar_profile_family.png)

## Profile-Control Optimization Figure

```bash
python examples/profile_control_optimization.py
```

This writes:

```text
docs/_static/profile_control_optimization.png
docs/_static/profile_control_optimization.pdf
```

Use this figure when the paper needs a direct optimization panel on top of the
profile closure itself, rather than the separate geometry-control science
figure.

![Profile control optimization](_static/profile_control_optimization.png)

## Profile-Basis Optimization Figure

```bash
python examples/profile_basis_optimization.py
```

This writes:

```text
docs/_static/profile_basis_optimization.png
docs/_static/profile_basis_optimization.pdf
```

Use this figure when the paper needs a profile-control optimization panel beyond
one scalar amplitude while still keeping the optimization space compact and
interpretable.

![Profile basis optimization](_static/profile_basis_optimization.png)

## Profile Transport Loop Figure

```bash
python examples/profile_transport_loop.py
```

This writes:

```text
docs/_static/profile_transport_loop.png
docs/_static/profile_transport_loop.pdf
```

Use this figure when the paper needs a self-consistent profile-transport panel
instead of a pure control-optimization panel. It shows how the ambipolar
residual, bootstrap-current proxy, and thermodynamic-force profiles evolve
under an accepted-step transport-relaxation iteration.

![Profile transport loop](_static/profile_transport_loop.png)

## Primitive Profile Transport Figure

```bash
python examples/primitive_profile_transport.py
```

This writes:

```text
docs/_static/primitive_profile_transport.png
docs/_static/primitive_profile_transport.pdf
```

Use this figure when the paper needs to move beyond direct `A1/A3` proxy
updates and show a primitive profile workflow in which density and temperature
remain positive, respond to explicit source-target closure terms, and feed back
into the ambipolar closure through reconstructed thermodynamic forces. The
panel is now framed around initial-versus-final closure profiles and the
derived monoenergetic forces rather than a noisy iteration trace.

![Primitive profile transport](_static/primitive_profile_transport.png)

## W7-X Bootstrap-Current Convergence Figure

```bash
python examples/bootstrap_current_reference_audit_w7x.py
```

This writes:

```text
docs/_static/bootstrap_current_reference_audit_w7x.png
docs/_static/bootstrap_current_reference_audit_w7x.pdf
docs/_static/bootstrap_current_reference_audit_w7x.json
```

Use this figure when the paper needs an explicit W7-X imported-workflow
bootstrap-current convergence panel alongside the NTX-only methods figures.

![W7-X bootstrap-current convergence](_static/bootstrap_current_reference_audit_w7x.png)
