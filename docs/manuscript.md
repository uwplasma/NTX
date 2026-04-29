# Manuscript Figures

NTX now includes a manuscript-ready figure bundle built directly from repository
examples.

## Curated Figure Set

### Main Text

1. `validation_summary.{png,pdf,json}`
2. `closure_validation_report.{png,pdf,json,txt}`
3. `bootstrap_current_reference_audit_w7x.{png,pdf}`
4. `derivative_path_benchmark.{png,pdf,json}`
5. `bootstrap_current_optimization.{png,pdf,json}`
6. `performance_scaling_production.{png,pdf,json}`
7. `primitive_profile_transport.{png,pdf}`

### Supplement

1. `autodiff_inverse_problem.{png,pdf}`
2. `autodiff_neopax_profiles.{png,pdf}`
3. `autodiff_profile_uncertainty.{png,pdf,json}`
4. `geometry_control_derivative_benchmark.{png,pdf,json}`
5. `file_backed_geometry_control_derivative_benchmark.{png,pdf,json}`
6. `boundary_forward_mode_current_derivative_benchmark.{png,pdf,json}`
7. `implicit_equilibrium_forward_mode_derivative_benchmark.{png,pdf,json}`
8. `explicit_relaxed_boundary_current_derivative_benchmark.{png,pdf,json}`
9. `geometry_family_breadth_summary.{png,pdf,json}`
10. `geometry_family_transport_convergence.{png,pdf,json}`
11. `owned_geometry_neopax_dataset.{png,pdf,json}`
12. `owned_finite_beta_sfincs_jax_inputs.{png,pdf,json}`
13. `owned_finite_beta_sfincs_jax_resolution_audit.{png,pdf,json}`
14. `owned_finite_beta_sfincs_jax_production_ladder_audit.{png,pdf,json}`
15. `owned_finite_beta_bootstrap_comparison.{png,pdf,json}`
16. `owned_finite_beta_closure_localization.{png,pdf,json}`
17. `owned_finite_beta_profile_current_observable_audit.{png,pdf,json}`
18. `owned_finite_beta_current_conditioning_audit.{png,pdf,json}`
19. `bootstrap_current_from_vmec_or_boozmn.{png,pdf,json}`
20. `bootstrap_current_robust_optimization.{png,pdf,json}`
21. `performance_scaling_smoke.{png,pdf,json}`
22. `performance_scaling_heavy.{png,pdf,json}`
23. `performance_strong_scaling_production.{png,pdf,json}`
24. `prepared_geometry_reuse_profile.{png,pdf,json}`
25. `ambipolar_profile.{png,pdf}`
26. `ambipolar_profile_family.{png,pdf}`
27. `profile_force_reconstruction_audit.{png,pdf,json}`
28. `profile_control_optimization.{png,pdf}`
29. `profile_basis_optimization.{png,pdf,json}`
30. `profile_transport_loop.{png,pdf}`

## Full Figure Inventory

1. `validation_summary.{png,pdf,json}`
   - transport-curve behavior on the sample DKES-style and VMEC surfaces
   - Onsager closure
   - Legendre convergence
   - machine-readable benchmark metrics for the literature-anchored methods lane
2. `closure_validation_report.{png,pdf,json,txt}`
   - fixed-field precise-QS Redl gate and scoped NTX+NEOPAX total-current
     closure stress gate in the same manuscript-facing validation report
3. `autodiff_inverse_problem.{png,pdf}`
   - inverse recovery of a surface harmonic from synthetic transport data
4. `autodiff_neopax_profiles.{png,pdf}`
   - autodiff-based profile inversion on NEOPAX-style arrays
5. `autodiff_profile_uncertainty.{png,pdf,json}`
   - three-term radial-basis uncertainty propagation on the same differentiable
     profile fit, including Monte Carlo, linearized covariance, and
     Fisher/Hessian-vector consistency diagnostics
6. `geometry_control_derivative_benchmark.{png,pdf,json}`
   - three-harmonic geometry-control derivative audit against centered finite
     differences; tracked as an autodiff stress benchmark
7. `file_backed_geometry_control_derivative_benchmark.{png,pdf,json}`
   - file-backed Boozer and VMEC geometry-control derivative audit against
     centered finite differences; stronger than the owned-surface stress test
     but still below a reusable geometry-family claim
8. `boundary_forward_mode_current_derivative_benchmark.{png,pdf,json}`
   - low-dimensional boundary controls propagated through boundary-projected
     `vmec_jax -> booz_xform_jax -> NTX` and an `NTX+NEOPAX` integrated-current
     objective under forward mode
9. `implicit_equilibrium_forward_mode_derivative_benchmark.{png,pdf,json}`
   - low-dimensional boundary controls propagated through the implicit
     fixed-boundary `vmec_jax` residual solve, `booz_xform_jax`, and an NTX
     monoenergetic transport proxy under forward mode, with the reverse-mode
     Boozer failure recorded in the JSON artifact
10. `explicit_relaxed_boundary_current_derivative_benchmark.{png,pdf,json}`
   - low-dimensional boundary controls propagated through an explicitly relaxed
     fixed-boundary `vmec_jax -> booz_xform_jax -> NTX` path and an
     `NTX+NEOPAX` integrated-current objective, with ordinary-versus-explicit
     primal-volume agreement recorded on committed QA and QH family cases
11. `geometry_family_breadth_summary.{png,pdf,json}`
   - artifact-backed breadth summary across analytic, file-backed,
     boundary-projected, explicit-relaxed, and implicit-equilibrium derivative
     paths; this is a stress summary and not a broad geometry-family
     validation claim
12. `geometry_family_transport_convergence.{png,pdf,json}`
   - public VMEC example-family `D11/D31/D33` convergence stress scan across
     tokamak, precise-QS, QI-style, W7-X, and stellarator-family inputs when
     the local checkouts are available; this is not an independent-code parity
     claim
13. `owned_geometry_neopax_dataset.{png,pdf,json}`
   - finite-beta owned `input/wout -> NTX -> NEOPAX-style` provenance figure,
     with the physical VMEC edge-flux scale passed into the Boozer-coordinate
     path, direct VMEC-harmonic interpolation-path stress diagnostics, and
     explicit geometry-backend blockers in the JSON sidecar
14. `owned_finite_beta_sfincs_jax_inputs.{png,pdf,json}`
   - six-point same-grid SFINCS-JAX finite-beta coefficient ladder with
     completed HDF5 ingestion, the SFINCS-reported `nuPrime -> nu_n` bridge,
     and a coefficient-level NTX `L13/L31/L33` comparison before
     profile-current parity promotion
15. `owned_finite_beta_sfincs_jax_resolution_audit.{png,pdf,json}`
   - production stress-radius rerun of the same finite-beta coefficient point
     at `35 x 43 x 48`, plus a tighter VMEC harmonic-cutoff probe; the
     coefficient floor remains near `2.05e-2`, about `15.8x` above the
     cancellation-conditioned current target
16. `owned_finite_beta_sfincs_jax_production_ladder_audit.{png,pdf,json}`
   - production six-point finite-beta QA same-grid SFINCS-JAX/NTX ladder across
     radius and collisionality; all coefficient differences stay below
     `2.07e-2`, with the current-conditioned precision gap localized to the
     inner stress point
17. `owned_finite_beta_bootstrap_comparison.{png,pdf,json}`
   - same finite-beta QA pressure/current `wout`, Boozer transform, analytic
     profiles, production radial/collisionality ladder, physical `nu/v`
     support, `D33_spitzer` branch, and current normalization used for Redl
     and `NTX+NEOPAX`; retained as a reduced-closure stress audit because the
     outer-radius current is near the `1e-1` target while the inner-radius gap
     remains open
18. `owned_finite_beta_closure_localization.{png,pdf,json}`
   - sidecar figure and JSON that compare the same-grid coefficient ladder with
     the finite-beta profile-current stress artifact; at the inner gap the
     coefficient-level error is about `2.1e-2`, while the current-profile error
     remains about `3.1e-1`
19. `owned_finite_beta_profile_current_observable_audit.{png,pdf,json}`
   - stress-radius decomposition of the profile-current observable into
     no-momentum current, applied momentum correction, correction needed to match
     Redl, species-current cancellation scale, local profile/geometry drivers,
     and Pmax trend
20. `owned_finite_beta_current_conditioning_audit.{png,pdf,json}`
   - cancellation-conditioned coefficient-precision requirement for the
     finite-beta net-current observable; this explains why the smoke
     coefficient ladder is not yet sufficient for a `1e-1` bootstrap-current
     parity claim
21. `derivative_path_benchmark.{png,pdf}`
   - prepared-derivative timing and agreement against direct reverse-mode
22. `bootstrap_current_optimization.{png,pdf}`
   - science/application figure for differentiable bootstrap-current
     optimization
23. `bootstrap_current_robust_optimization.{png,pdf,json}`
   - deterministic versus robust optimization under a prescribed control
     uncertainty; tracked as an open robust-design lane
24. `bootstrap_current_from_vmec_or_boozmn.{png,pdf}`
   - NTX-only bootstrap-current-proxy profile from VMEC/Boozer input
25. `bootstrap_current_reference_audit_w7x.{png,pdf}`
   - W7-X imported-workflow bootstrap-current convergence audit
26. `performance_scaling_smoke.{png,pdf,json}`
   - CPU/GPU scaling on the repository smoke grid
27. `performance_scaling_heavy.{png,pdf,json}`
   - heavier-grid scaling where throughput effects are visible
28. `performance_scaling_production.{png,pdf,json}`
   - production-grid CPU/GPU scaling with serial, device-parallel,
     multiprocess, memory, and coefficient-agreement metadata
29. `performance_strong_scaling_production.{png,pdf,json}`
   - fixed-workload CPU/GPU strong scaling with worker/device sweeps, memory,
     and coefficient-agreement metadata
30. `prepared_geometry_reuse_profile.{png,pdf,json}`
   - fixed-geometry repeated-solve profile showing the direct, prepared, and
     compiled prepared solver paths with coefficient agreement recorded in the
     JSON artifact
31. `ambipolar_profile.{png,pdf}`
   - profile-grade ambipolar electric-field solve and bootstrap-current proxy
32. `ambipolar_profile_family.{png,pdf}`
   - control-parameter family of ambipolar closures and scalar bootstrap-current objective
33. `profile_force_reconstruction_audit.{png,pdf,json}`
   - archived precise-QS QA/QH primitive-to-force reconstruction audit
34. `profile_control_optimization.{png,pdf}`
   - differentiable optimization of a scalar profile control on top of the ambipolar closure
35. `profile_basis_optimization.{png,pdf,json}`
   - low-dimensional radial-basis optimization of the same profile closure
36. `profile_transport_loop.{png,pdf}`
   - explicit self-consistent transport-relaxation iteration on the same profile closure
37. `primitive_profile_transport.{png,pdf}`
   - primitive density/temperature transport iteration mapped back to ambipolar-field and bootstrap-current evolution

Together these figures cover:

- formulation and numerical behavior
- validation and convergence
- fixed-field Redl validation and reduced-closure total-current stress
  reporting
- differentiable inverse and profile problems
- differentiable uncertainty propagation on the same profile map
- multi-parameter geometry-control derivative auditing
- file-backed Boozer and VMEC geometry-control derivative auditing
- boundary-to-output forward-mode auditing on projected `vmec_jax` geometry
- implicit-equilibrium derivative diagnostics that isolate where parity is lost:
  equilibrium volume matches, but Boozer geometry and NTX transport are closed
  as non-shipping diagnostics
- equilibrium-relaxed boundary-to-current forward-mode auditing on committed QA/QH family cases
- artifact-backed geometry-breadth status across the committed derivative
  families, with unresolved implicit objectives kept out of promoted claims
- same-grid finite-beta Redl and `NTX+NEOPAX` bootstrap-current stress
  diagnostics with the physical Boozer flux scale, production radial/
  collisionality ladder, adaptive `nu/v` support, and Sonine-order convergence
  sidecar recorded while the inner-radius gap remains open work
- production same-grid finite-beta SFINCS-JAX coefficient ladders that close
  radius/collisionality resolution as the leading explanation for that gap
- a deterministic robust-design stress benchmark for differentiable current optimization
- derivative cost for prepared optimization workflows
- a science-facing bootstrap-current optimization workflow
- a pure NTX radial-profile figure
- a profile-grade ambipolar and bootstrap-current-proxy workflow
- a control-parameter family view of the same profile-grade closure
- a literature-anchored primitive-to-force reconstruction audit on the precise-QS profile family
- a direct optimization view of the profile-grade closure
- a low-dimensional multi-parameter version of that optimization
- a self-consistent transport-relaxation view of the same closure
- a primitive-profile transport view with positive density and temperature updates
- a W7-X imported-workflow convergence figure
- practical performance guidance
- prepared-geometry and compiled-solver reuse guidance for optimization workloads

## Manuscript Tables And Reproducibility

```bash
python scripts/build_manuscript_artifacts.py
```

This writes:

```text
docs/_static/manuscript_artifacts.json
docs/_static/manuscript_tables.md
docs/_static/manuscript_claims.md
```

These artifacts collect the current NTX commit, software environment, the
validated W7-X convergence numbers, derivative benchmark summaries,
production-grid CPU/GPU performance and strong-scaling tables,
geometry-control derivative stress metrics,
finite-beta bootstrap-current stress and closure-localization metrics,
bootstrap-current optimization
summaries, and the exact commands needed to
regenerate the figures and validation subset used in the manuscript.

## One-Command Figure Bundle

```bash
python examples/make_publication_figures.py
```

This writes the full figure set into `docs/_static/` and also creates:

```text
docs/_static/publication_figure_manifest.json
```

Generate the frozen main-text set:

```bash
python examples/make_publication_figures.py --figures main_text
```

Generate the supplement set:

```bash
python examples/make_publication_figures.py --figures supplement
```

## Science Figure

```bash
python examples/bootstrap_current_optimization.py
```

The science/application figure is written to:

```text
docs/_static/bootstrap_current_optimization.png
docs/_static/bootstrap_current_optimization.pdf
docs/_static/bootstrap_current_optimization.json
```

It uses:

- a VMEC-derived radial surface family
- a dominant non-axisymmetric harmonic as the control parameter
- a weighted bootstrap-current proxy based on the current-response coefficients
- JAX autodiff to optimize that control directly

The committed JSON artifact is also a monitored benchmark-matrix and
physics-gate entry: the optimized weighted-current proxy must remain at least
as large as the baseline before the manuscript cites the gain. Broader
stellarator-design claims still require reusable geometry-family controls and
their derivative audits.

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
docs/_static/derivative_path_benchmark.json
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
docs/_static/profile_basis_optimization.json
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
