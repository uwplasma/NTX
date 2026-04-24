# Source-Code Map

This page maps the main physics formulas and workflows onto the NTX source
tree.

## Core Solver Modules

| Topic | Main file | Key functions / classes |
| --- | --- | --- |
| Surface dataclasses and geometry evaluation | `src/ntx/geometry.py`, `src/ntx/_geometry_types.py`, `src/ntx/_geometry_eval.py` | `BoozerSurface`, `VmecSurface`, `GeometryOnGrid`, `geometry_on_grid(...)` |
| Angular grids and Fourier differentiation | `src/ntx/grids.py` | `GridSpec`, `periodic_grid(...)`, `fourier_derivative_matrix(...)` |
| Legendre-space operator coefficients | `src/ntx/operators.py` | `OperatorContext`, `coefficients_for_k(...)`, `operator_blocks(...)`, `source_modes(...)` |
| Dense block solve and scans | `src/ntx/solver.py`, `src/ntx/_solver_types.py`, `src/ntx/_solver_core.py`, `src/ntx/_solver_factorization.py`, `src/ntx/_solver_scan.py` | `MonoenergeticCase`, `TransportResult`, `solve_monoenergetic(...)`, `_solve_modes(...)`, `solve_prepared_coefficient_vector_vjp(...)` |
| Transport post-processing | `src/ntx/transport.py` | `coefficients_from_modes(...)`, `onsager_error(...)` |
| CLI/TOML workflow | `src/ntx/inputfiles.py`, `src/ntx/_inputfiles_model.py`, `src/ntx/_inputfiles_reporting.py`, `src/ntx/_inputfiles_run.py`, `src/ntx/cli.py` | `load_run_config(...)`, `run_from_input_file(...)`, `save_run_npz(...)` |
| VMEC loading | `src/ntx/vmec.py` | `load_vmec_surface(...)` |
| In-memory `vmec_jax -> booz_xform_jax` boundary workflows | `src/ntx/vmec_jax_backend.py`, `src/ntx/_vmec_jax_boozer.py`, `src/ntx/_neopax_vmec_jax_field.py`, `src/ntx/_neopax_field.py` | `build_vmec_jax_boundary_context(...)`, `initial_guess_vmec_jax_boundary_state(...)`, `solve_vmec_jax_boundary_state(...)`, imported Boozer and NEOPAX field builders |
| Boozer file loading | `src/ntx/booz.py` | Boozer harmonic file loaders |
| NEOPAX coupling | `src/ntx/neopax.py`, `src/ntx/_neopax_types.py`, `src/ntx/_neopax_io.py`, `src/ntx/_neopax_bridge.py`, `src/ntx/_neopax_scan.py`, `src/ntx/_neopax_field.py`, `src/ntx/_neopax_fluxes.py`, `src/ntx/_neopax_field_utils.py`, `src/ntx/_neopax_vmec_jax_field.py` | `build_ntx_neopax_scan(...)`, `scan_to_neopax_arrays(...)`, `write_neopax_scan_hdf5(...)`, differentiable imported-field helpers |
| Profile-grade imported workflows | `src/ntx/profiles.py`, `src/ntx/_profiles_types.py`, `src/ntx/_profiles_radial.py`, `src/ntx/_profiles_channels.py`, `src/ntx/_profiles_primitives.py`, `src/ntx/_profiles_eval.py`, `src/ntx/_profiles_controls.py`, `src/ntx/_profiles_transport_terms.py`, `src/ntx/_profiles_transport_closure.py`, `src/ntx/_profiles_transport.py` | radial profile helpers, scan-channel interpolation, primitive-force reconstruction, ambipolar `E_r(r)` solve, controls, normalized transport terms, closures, and transport loops |
| Throughput-oriented multi-device execution | `src/ntx/parallel.py` | `solve_monoenergetic_multiprocess_scan(...)` |
| Autodiff examples and optimization helpers | `src/ntx/autodiff.py`, `src/ntx/_autodiff_types.py`, `src/ntx/_autodiff_helpers.py`, `src/ntx/_autodiff_workflows.py`, `src/ntx/_autodiff_inverse.py`, `src/ntx/_autodiff_derivatives.py`, `src/ntx/_autodiff_profile.py`, `src/ntx/_autodiff_bootstrap.py` | inverse, sensitivity, uncertainty, and bootstrap-current optimization helpers |
| Validation registries | `src/ntx/validation/benchmark_matrix.py`, `src/ntx/validation/_benchmark_matrix_types.py`, `src/ntx/validation/_benchmark_matrix_entries.py`, `src/ntx/validation/physics_gates.py`, `src/ntx/validation/_physics_gate_types.py`, `src/ntx/validation/_physics_gate_registry.py`, `src/ntx/validation/_physics_gate_artifacts.py` | benchmark-matrix evaluator, benchmark claim types, maintained claim metadata, physics-gate definitions, and artifact-gate evaluation |

The compatibility modules remain the primary implementation locations. The
newer namespace packages provide stable grouped imports:

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
  [`src/ntx/_vmec_jax_boozer.py`](../src/ntx/_vmec_jax_boozer.py), which
  enforce the same right-handed sign convention for in-memory JAX Boozer data
  that the file-backed loader applies before constructing `BoozerSurface`

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
[`src/ntx/_solver_factorization.py`](../src/ntx/_solver_factorization.py).

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
- helpers in [`src/ntx/autodiff.py`](../src/ntx/autodiff.py)

## Profile Forces And Ambipolarity

The profile workflow reconstructs the force proxies

```{math}
A_3 = d\ln T/dr,
\qquad
A_1 = d\ln n/dr - \frac{3}{2} d\ln T/dr + C_E Z E_r
```

from primitive density, temperature, charge, and radial-electric-field inputs in
[`src/ntx/_profiles_primitives.py`](../src/ntx/_profiles_primitives.py).
Scan-channel interpolation and species particle/current responses live in
[`src/ntx/_profiles_channels.py`](../src/ntx/_profiles_channels.py). The
ambipolar profile residual uses the charge-weighted particle-flux condition

```{math}
\sum_s Z_s \Gamma_s(E_r) = 0
```

before the radial-electric-field solve in
[`src/ntx/_profiles_eval.py`](../src/ntx/_profiles_eval.py).
Normalized transport mismatch terms, update clipping, and primitive
density/temperature mismatch algebra live in
[`src/ntx/_profiles_transport_terms.py`](../src/ntx/_profiles_transport_terms.py);
the positivity-preserving explicit update itself lives in
[`src/ntx/_profiles_transport_closure.py`](../src/ntx/_profiles_transport_closure.py).

## Publication Figures

The publication-ready example scripts live in [`examples/`](../examples):

- `validation_summary.py`
- `bootstrap_current_optimization.py`
- `bootstrap_current_from_vmec_or_boozmn.py`
- `bootstrap_current_with_neopax.py`
- `bootstrap_current_fixed_field_validation.py`
- `bootstrap_current_reference_audit_w7x.py`
- `performance_scaling.py`
- `autodiff_inverse_problem.py`
- `neopax_autodiff_profiles.py`
- `derivative_audit.py`
- `derivative_path_benchmark.py`
- `geometry_control_derivative_benchmark.py`
- `file_backed_geometry_control_derivative_benchmark.py`
- `explicit_relaxed_boundary_current_derivative_benchmark.py`
- `ambipolar_profile.py`
- `ambipolar_profile_family.py`
- `profile_control_optimization.py`
- `profile_basis_optimization.py`
- `profile_transport_loop.py`
- `primitive_profile_transport.py`
- `plot_output_npz.py`

The figure bundle generator is:

- `make_publication_figures.py`

The benchmark-matrix artifact generator is:

- `scripts/build_benchmark_matrix.py`
