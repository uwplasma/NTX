# Source-Code Map

This page maps the main physics formulas and workflows onto the NTX source
tree.

## Core Solver Modules

| Topic | Main file | Key functions / classes |
| --- | --- | --- |
| Surface dataclasses and geometry evaluation | `src/ntx/geometry.py`, `src/ntx/_geometry_types.py`, `src/ntx/_geometry_eval.py` | `BoozerSurface`, `VmecSurface`, `GeometryOnGrid`, `geometry_on_grid(...)` |
| Angular grids and Fourier differentiation | `src/ntx/grids.py` | `GridSpec`, `periodic_grid(...)`, `fourier_derivative_matrix(...)` |
| Legendre-space operator coefficients | `src/ntx/operators.py` | `OperatorContext`, `coefficients_for_k(...)`, `operator_blocks(...)`, `source_modes(...)` |
| Dense block solve and scans | `src/ntx/solver.py` | `MonoenergeticCase`, `TransportResult`, `solve_monoenergetic(...)`, `_solve_modes(...)`, `solve_prepared_coefficient_vector_vjp(...)` |
| Transport post-processing | `src/ntx/transport.py` | `coefficients_from_modes(...)`, `onsager_error(...)` |
| CLI/TOML workflow | `src/ntx/inputfiles.py`, `src/ntx/_inputfiles_model.py`, `src/ntx/_inputfiles_reporting.py`, `src/ntx/cli.py` | `load_run_config(...)`, `run_from_input_file(...)`, `save_run_npz(...)` |
| VMEC loading | `src/ntx/vmec.py` | `load_vmec_surface(...)` |
| Boozer file loading | `src/ntx/booz.py` | Boozer harmonic file loaders |
| NEOPAX coupling | `src/ntx/neopax.py`, `src/ntx/_neopax_types.py`, `src/ntx/_neopax_io.py`, `src/ntx/_neopax_bridge.py` | `build_ntx_neopax_scan(...)`, `scan_to_neopax_arrays(...)`, `write_neopax_scan_hdf5(...)` |
| Profile-grade imported workflows | `src/ntx/profiles.py` | species-profile closures, ambipolar `E_r(r)` solve, bootstrap-current proxy |
| Throughput-oriented multi-device execution | `src/ntx/parallel.py` | `solve_monoenergetic_multiprocess_scan(...)` |
| Autodiff examples and optimization helpers | `src/ntx/autodiff.py` | inverse, sensitivity, and bootstrap-current optimization helpers |

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
`_solve_modes(...)` in [`src/ntx/solver.py`](../src/ntx/solver.py).

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
  [`src/ntx/neopax.py`](../src/ntx/neopax.py)
- helpers in [`src/ntx/autodiff.py`](../src/ntx/autodiff.py)

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
- `ambipolar_profile.py`
- `ambipolar_profile_family.py`
- `profile_control_optimization.py`
- `profile_basis_optimization.py`
- `profile_transport_loop.py`
- `primitive_profile_transport.py`
- `plot_output_npz.py`

The figure bundle generator is:

- `make_publication_figures.py`
