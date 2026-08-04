# Source-Code Map

This page maps the main physics formulas and workflows onto the NTX source
tree.

## Core Solver Modules

| Topic | Main file | Key functions / classes |
| --- | --- | --- |
| Surface dataclasses and geometry evaluation | `src/ntx/geometry.py`, `src/ntx/_geometry.py` | `BoozerSurface`, `VmecSurface`, `GeometryOnGrid`, `geometry_on_grid(...)` |
| Angular grids and Fourier differentiation | `src/ntx/grids.py` | `GridSpec`, `periodic_grid(...)`, `fourier_derivative_matrix(...)` |
| Legendre-space operator coefficients | `src/ntx/operators.py` | `OperatorContext`, `coefficients_for_k(...)`, `operator_blocks(...)`, `source_modes(...)` |
| Dense block solve and scans | `src/ntx/solver.py`, `src/ntx/_solver.py`, `src/ntx/_solver_derivative_audit.py`, `src/ntx/_solver_scan.py` | `MonoenergeticCase`, `TransportResult`, shared operator-context construction, solve orchestration, SOLVAX adapters, prepared solve wrappers, custom-VJP adjoint algebra, adjoint-window selection (heuristic and certified), single-device scan orchestration, fixed-bucket compilation, device health/sharding, and independently gated derivative audits |
| Transport post-processing | `src/ntx/transport.py` | `coefficients_from_modes(...)`, `onsager_error(...)` |
| CLI/TOML workflow | `src/ntx/inputfiles.py`, `src/ntx/_inputfiles.py`, `src/ntx/plotting.py`, `src/ntx/cli.py` | `load_run_config(...)`, `run_from_input_file(...)`, `save_run_output(...)`, `plot_run_output(...)` |
| VMEC loading | `src/ntx/vmec.py` | `load_vmec_surface(...)` |
| In-memory `vmex -> booz_xform_jax` boundary workflows | `src/ntx/vmex_backend.py`, `src/ntx/_vmex.py`, `src/ntx/_neopax.py` | `build_vmex_boundary_context(...)`, `initial_guess_vmex_boundary_state(...)`, `solve_vmex_boundary_state(...)`, VMEX state to Boozer-surface builders, VMEC scalar/profile transfer, imported Boozer and NEOPAX field builders |
| Boozer file loading | `src/ntx/booz.py` | Boozer harmonic file loaders |
| NEOPAX coupling | `src/ntx/neopax.py`, `src/ntx/_neopax.py`, `src/ntx/_neopax_scan.py` | `build_ntx_neopax_scan(...)`, scan field-channel normalization, scan coefficient and bridge assembly, `scan_to_neopax_arrays(...)`, `write_neopax_scan_hdf5(...)`, differentiable imported-field helpers |
| Profile-grade imported workflows | `src/ntx/profiles.py`, `src/ntx/_profiles.py`, `src/ntx/_profiles_control.py`, `src/ntx/_profiles_transport.py` | radial profile dataclasses, scan-channel interpolation, primitive-force reconstruction, ambipolar `E_r(r)` solve, scalar controls, basis controls, normalized transport terms, closures, and transport loops |
| Throughput-oriented multi-device execution | `src/ntx/parallel.py` | `solve_monoenergetic_multiprocess_scan(...)` |
| Autodiff examples and optimization helpers | `src/ntx/autodiff.py`, `src/ntx/_autodiff.py`, `src/ntx/_autodiff_bootstrap.py` | inverse, sensitivity, uncertainty, and deterministic/robust bootstrap-current optimization helpers |
| Validation registries | `src/ntx/validation/benchmark_matrix.py`, `src/ntx/validation/_benchmark_matrix.py`, `src/ntx/validation/_benchmark_matrix_geometry.py`, `src/ntx/validation/physics_gates.py`, `src/ntx/validation/_physics_gate.py`, `src/ntx/validation/_physics_gate_artifacts.py`, `src/ntx/validation/_finite_beta_closure_target.py`, `src/ntx/validation/_finite_beta_source_channels.py` | benchmark-matrix evaluator, benchmark claim types, lane-owned benchmark metadata, finite-beta geometry-breadth metadata, analytical physics-gate definitions, artifact-backed gate definitions, finite-beta artifact-gate registry definitions, registry facade, shared artifact-gate evaluators, finite-beta artifact-gate ownership, finite-beta closure-target artifact parsing, and finite-beta source-channel artifact summaries |

The compatibility modules remain the primary public import locations. New
implementation ownership can sit behind those facades when the split improves
testing or source navigation. The newer namespace packages provide stable
grouped imports:

- `ntx.core` for solver, scan, and transport helpers.
- `ntx.workflows` for autodiff, profile, and imported database helpers.
- `ntx.validation` for benchmark and validation registries.

## Refactoring Target

The source tree should keep moving toward smaller ownership boundaries, but only
when a move improves testing, documentation, or API clarity.

Target ownership:

| Package | Responsibility |
| --- | --- |
| `ntx.core` | grids, operator assembly, dense solve, transport coefficients, prepared solve/VJP |
| `ntx.geometry` | analytic, Boozer, VMEC, and imported JAX geometry evaluation |
| `ntx.io` | TOML, NPZ, HDF5, NetCDF, and benchmark artifact loading/writing |
| `ntx.workflows` | profile, NEOPAX, autodiff, optimization, and publication workflows |
| `ntx.validation` | benchmark matrix, physics gates, literature claim metadata, artifact summaries |

Compatibility facades should remain stable for users. Internal modules can move
behind those facades when the new location has direct unit tests and the docs
map has been updated in the same change.
`tests/test_source_map.py` keeps the split internal modules listed here, so each
future ownership split must update this map in the same change.

## Equation-To-Code Mapping

## Fourier Representation Of `B`

```{math}
B(\theta,\zeta)
=
\sum_{m,n} B_{mn}\cos(m\theta + n N_\mathrm{fp}\zeta)
```

Implemented in:

- `evaluate_fourier_series(...)`
- `evaluate_boozer_modes(...)`

in [`src/ntx/geometry.py`](../src/ntx/geometry.py).

## Boozer Jacobian And Field Components

```{math}
\mathcal J = |B_\zeta + \iota B_\theta|/B^2
```

Implemented in:

- `_boozer_geometry_on_grid(...)` in
  [`src/ntx/geometry.py`](../src/ntx/geometry.py)
- `_apply_boozer_sign_convention(...)` and
  `_apply_boozer_sign_convention_profiles(...)` in
  [`src/ntx/_vmex.py`](../src/ntx/_vmex.py), which
  enforce the same right-handed sign convention for in-memory JAX Boozer data
  that the file-backed loader applies before constructing `BoozerSurface`

The fast physics-gate suite checks the right-handed Boozer identity
`\mathcal J B^2 = B_\zeta + \iota B_\theta`, plus the corresponding
contravariant identities `B^\theta \mathcal J = \iota` and
`B^\zeta \mathcal J = 1`, before any transport coefficient is interpreted.

## Radial-Drift Spatial Factor

```{math}
\hat v_m
=
\frac{B_\theta \partial_\zeta B - B_\zeta \partial_\theta B}
{\mathcal J B^3}
```

Implemented in:

- `_boozer_geometry_on_grid(...)`
- `_vmec_geometry_on_grid(...)`

in [`src/ntx/geometry.py`](../src/ntx/geometry.py).

## Legendre-Space Block System

```{math}
L_k f^{(k-1)} + D_k f^{(k)} + U_k f^{(k+1)} = s^{(k)}
```

Implemented in:

- `coefficients_for_k(...)`
- `operator_blocks(...)`

in [`src/ntx/operators.py`](../src/ntx/operators.py), and solved by
`_solve_modes(...)` in
[`src/ntx/_solver.py`](../src/ntx/_solver.py).

## Nullspace Fix

```{math}
f^{(0)}(\theta_0,\zeta_0)=0
```

Implemented in:

- `apply_nullspace_condition(...)` in
  [`src/ntx/operators.py`](../src/ntx/operators.py)

## Flux-Surface Averages And Transport Coefficients

Implemented in:

- `coefficients_from_modes(...)` in
  [`src/ntx/transport.py`](../src/ntx/transport.py)

This is the only module that turns solved Legendre modes into the reported
`D11`, `D31`, `D13`, `D33`, and `D33_spitzer`.

## Electric-Field Normalization

Implemented in:

- `MonoenergeticCase.resolved_epsi_hat(...)` in
  [`src/ntx/solver.py`](../src/ntx/solver.py)
- `load_vmec_surface(...)` in [`src/ntx/vmec.py`](../src/ntx/vmec.py)

## Scan And Differentiable Workflows

Implemented in:

- `solve_monoenergetic_scan(...)` in
  [`src/ntx/solver.py`](../src/ntx/solver.py)
- `build_ntx_neopax_scan(...)` in
  [`src/ntx/_neopax_scan.py`](../src/ntx/_neopax_scan.py)
- scan field-channel normalization in
  [`src/ntx/_neopax_scan.py`](../src/ntx/_neopax_scan.py)
- scan coefficient and bridge-block assembly in
  [`src/ntx/_neopax_scan.py`](../src/ntx/_neopax_scan.py)
- helpers in [`src/ntx/autodiff.py`](../src/ntx/autodiff.py)

## Profile Forces And Ambipolarity

The profile workflow reconstructs the thermodynamic-force channels

```{math}
A_3 = d\ln T/dr,
\qquad
A_1 = d\ln n/dr - \frac{3}{2} d\ln T/dr + C_E Z E_r
```

from primitive density, temperature, charge, and radial-electric-field inputs in
[`src/ntx/_profiles.py`](../src/ntx/_profiles.py).
Scan-channel interpolation and species particle/current responses live in
[`src/ntx/_profiles.py`](../src/ntx/_profiles.py). The
ambipolar profile residual uses the charge-weighted particle-flux condition

```{math}
\sum_s Z_s \Gamma_s(E_r) = 0
```

before the radial-electric-field solve in
[`src/ntx/_profiles.py`](../src/ntx/_profiles.py).
Normalized transport mismatch terms, update clipping, and primitive
density/temperature mismatch algebra live in
[`src/ntx/_profiles_transport.py`](../src/ntx/_profiles_transport.py);
the positivity-preserving explicit update itself lives in
[`src/ntx/_profiles_transport.py`](../src/ntx/_profiles_transport.py).

## Publication Figures

The publication-ready example scripts live in [`examples/`](../examples):

- `validation_summary.py`
- `bootstrap_current_optimization.py`
- `bootstrap_current_from_vmec_or_boozmn.py`
- `bootstrap_current_with_neopax.py`
- `bootstrap_current_fixed_field_validation.py`
- `bootstrap_current_reference_audit_w7x.py`
- `owned_geometry_neopax_dataset.py`
- `owned_finite_beta_sfincs_jax_inputs.py`
- `owned_finite_beta_sfincs_jax_resolution_audit.py`
- `owned_finite_beta_sfincs_jax_production_ladder_audit.py`
- `owned_finite_beta_sfincs_jax_profile_current_audit.py`
- `owned_finite_beta_sfincs_jax_profile_current_resolution_audit.py`
- `owned_finite_beta_bootstrap_comparison.py`
- `owned_finite_beta_closure_localization.py`
- `owned_finite_beta_profile_current_observable_audit.py`
- `owned_finite_beta_current_conditioning_audit.py`
- `owned_finite_beta_closure_quadrature_audit.py`
- `owned_finite_beta_source_channel_audit.py`
- `owned_finite_beta_source_response_profile_audit.py`
- `owned_finite_beta_closure_target_audit.py`
- `performance_scaling.py`
- `performance_strong_scaling.py`
- `prepared_geometry_reuse_profile.py`
- `autodiff_inverse_problem.py`
- `neopax_autodiff_profiles.py`
- `derivative_audit.py`
- `derivative_path_benchmark.py`
- `geometry_control_derivative_benchmark.py`
- `file_backed_geometry_control_derivative_benchmark.py`
- `explicit_relaxed_boundary_current_derivative_benchmark.py`
- `geometry_family_breadth_summary.py`
- `geometry_family_transport_convergence.py`
- `ambipolar_profile.py`
- `ambipolar_profile_family.py`
- `profile_control_optimization.py`
- `profile_basis_optimization.py`
- `profile_transport_loop.py`
- `primitive_profile_transport.py`
- `plot_output_file.py`

The fixed-field validation script delegates reviewable artifact math to
`examples/_fixed_field_validation_metrics.py` and
`examples/_fixed_field_validation_plotting.py`, and
`examples/_fixed_field_validation_summary.py`; the closure-specific diagnostic
assembly lives in `examples/_fixed_field_validation_closure.py`. The expensive
QA/QH simulation path remains in
`examples/bootstrap_current_fixed_field_validation.py`.

The figure bundle generator is:

- `make_publication_figures.py`

The benchmark-matrix artifact generator is:

- `scripts/build_benchmark_matrix.py`

Performance timing artifacts are generated by:

- `scripts/benchmark_scaling.py`
- `scripts/benchmark_strong_scaling.py`
