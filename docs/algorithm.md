# Algorithm

NTX solves the truncated Legendre-space monoenergetic drift-kinetic equation

```text
L_k f^(k-1) + D_k f^(k) + U_k f^(k+1) = s^(k)
```

for the low-order modes needed to evaluate:

- `D11`
- `D31`
- `D13`
- `D33`
- `D33_spitzer`

## Spatial Discretization

The angular coordinates are discretized on a periodic `(theta, zeta)` grid.
Each Legendre mode carries one dense spatial block of size

```text
N_fs = n_theta * n_zeta
```

Derivative operators are assembled from Fourier pseudospectral differentiation
matrices.

## Legendre Coupling

For each Legendre index `k`, NTX constructs:

- `L_k`: coupling to `f^(k-1)`
- `D_k`: diagonal block for `f^(k)`
- `U_k`: coupling to `f^(k+1)`

The source structure follows Escoto's formulation:

- `s1` contributes in modes `k = 0` and `k = 2`
- `s3` contributes in mode `k = 1`

## Nullspace Handling

The `k = 0` block has a constant nullspace. NTX removes it by replacing the
first row with the constraint

```text
f^(0)(theta = 0, zeta = 0) = 0
```

## Solver

NTX uses a dense block-tridiagonal Schur-complement solve:

1. Build the terminal block.
2. Sweep backward in `k` with forward elimination.
3. Store the low-order Schur-complement blocks needed for output.
4. Back-substitute modes `k = 0, 1, 2`.

This is the current production path in NTX.

## JAX Usage

The implementation is written as pure JAX/Numpy-style array code:

- `jax.jit` for repeated single-case solves
- `jax.vmap` for parameter scans
- x64 mode for physics runs
- dense linear solves through JAX SciPy wrappers

NTX now distinguishes between:

- a file-driven CLI layer for terminal runs and serialized outputs
- a differentiable imported core for JAX-native use
- a VMEC/Boozer imported path built around `vmec_jax` and `booz_xform_jax`

The imported core supports:

- `jit` over Boozer-surface arguments
- gradients with respect to `nu_hat`
- gradients with respect to `er_hat`
- gradients with respect to surface Fourier coefficients

The imported VMEC/Boozer path is:

1. solve or load a VMEC equilibrium in `vmec_jax`
2. construct Boozer inputs with `vmec_jax.booz_xform_inputs_from_state`
3. transform with `booz_xform_jax`
4. solve the monoenergetic equation in NTX from the resulting Boozer harmonics

This keeps the performance-oriented CLI path separate from the JAX-native
imported path without forcing the terminal runner to satisfy autodiff
constraints.

## Geometry

NTX supports two surface families:

- DKES-style Boozer harmonics with scalar `B_theta`, `B_zeta`, and `psi_p`
- VMEC `wout` Fourier data with spatially varying covariant and contravariant
  field components

The VMEC path evaluates:

- `B`
- `Jacobian`
- `B_sub_theta`, `B_sub_zeta`
- `B_sup_theta`, `B_sup_zeta`
- `r_n = sqrt(psi_n)`
- `r_hat = Aminor_p * r_n`
- `dpsi_hat/dr_hat`

on the requested angular grid before assembling the operator blocks.

## Electric-Field Normalization

NTX supports two equivalent electric-field inputs:

- `epsi_hat`
- `er_hat`

For Boozer / DKES surfaces:

```text
epsi_hat = er_hat / psi_p
```

For VMEC surfaces:

```text
r_n = sqrt(psi_n)
r_hat = Aminor_p * r_n
dpsi_hat/dr_hat = 2 * psi_a_hat * r_n / Aminor_p
epsi_hat = er_hat / (dpsi_hat/dr_hat)
```

The VMEC transport coefficients use the same `dpsi_hat/dr_hat` scale in the
post-processing formulas, so the stored coefficient normalization matches the
resolved field normalization shown at runtime.

## Diagnostics

Each solve returns:

- transport coefficients
- `residual_l2`
- `onsager_residual = |D13 + D31|`

The imported low-level interface also exposes:

- `solve_monoenergetic_internal(surface, grid, case) -> (Dij, f, s)`
- `solve_prepared_internal(prepared, case) -> (Dij, f, s)`
- `build_monoenergetic_database_arrays(surface, grid, nu_hat, ...)`
- `surface_from_vmec_jax_state(...)`
- `surface_from_vmec_jax_wout(...)`
- `build_ntx_neopax_scan(...)`
- `to_neopax_monoenergetic(...)`

where `f` and `s` are the retained low-order internal systems needed for the
monoenergetic coefficient evaluation, and the database builder returns
tensor-product in-memory scan arrays over `(nu_hat, scan_field)` for one
surface.

For file-driven runs, NTX also writes geometry statistics, surface metadata, and
algorithm metadata into the output `.npz`.
