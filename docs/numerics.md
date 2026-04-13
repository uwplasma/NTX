# Numerics And Algorithms

NTX is built around one numerical idea: the Legendre representation of the
monoenergetic drift-kinetic equation leads to a dense block-tridiagonal system
that can be solved in time linear in `N_xi` and cubic in the number of spatial
points `N_fs = N_theta N_zeta`.

This page documents the discretization and the exact algorithm implemented in
the source tree.

## Angular Discretization

The angular grid is uniform in `\theta` and `\zeta`:

```{math}
\theta_j = \frac{2\pi j}{N_\theta},
\qquad
\zeta_\ell = \frac{2\pi \ell}{N_\mathrm{fp} N_\zeta}.
```

This is created by `periodic_grid(...)` in
[`src/ntx/grids.py`](../src/ntx/grids.py).

### Spectral Derivatives

The first-derivative matrices are Fourier-collocation matrices assembled by
differentiating the discrete Fourier basis:

```{math}
D = F^{-1} (ik) F.
```

In code this is `fourier_derivative_matrix(...)` in
[`src/ntx/grids.py`](../src/ntx/grids.py). It works for even and odd grid sizes
and is easy to differentiate because it is built from JAX FFT primitives.

## Dense Spatial Blocks

For each Legendre mode `k`, NTX packs the coefficient fields

```{math}
(c_\theta, c_\zeta, c_0)
```

and constructs the dense operator block

```{math}
\mathcal B_k
=
\mathrm{diag}(c_\theta) D_\theta
+
\mathrm{diag}(c_\zeta) D_\zeta
+
\mathrm{diag}(c_0).
```

This happens in:

- `derivative_blocks(...)`
- `build_block(...)`
- `operator_blocks(...)`

in [`src/ntx/operators.py`](../src/ntx/operators.py).

## Forward Schur Recursion

NTX does not invert the full block-tridiagonal matrix directly. Instead it uses
the Schur-complement recursion described in the thesis.

Starting from the terminal mode,

```{math}
\Delta_{N_\xi} = D_{N_\xi},
\qquad
X_{N_\xi} = \Delta_{N_\xi}^{-1} L_{N_\xi},
```

and then for descending `k`

```{math}
\Delta_k = D_k - U_k X_{k+1},
\qquad
X_k = \Delta_k^{-1} L_k.
```

This is implemented in `_solve_modes(...)` in
[`src/ntx/solver.py`](../src/ntx/solver.py) with a `jax.lax.scan`.

### Why Only Modes 0, 1, And 2 Are Stored

The monoenergetic coefficients need only the first three Legendre modes.
Therefore NTX:

- runs the Schur recursion through all `k = N_\xi, \dots, 0`
- saves only `L_k`, `U_k`, and `\Delta_k` for `k = 0,1,2`
- back-substitutes only those low-order modes

That keeps the method dense and simple while avoiding storage of the full
Legendre spectrum in the standard transport workflow.

## Backward Substitution

Once the low-order Schur complements are known, NTX forms reduced right-hand
sides and solves

```{math}
\Delta_0 f^{(0)} = \sigma^{(0)},
\qquad
\Delta_1 f^{(1)} = \sigma^{(1)} - L_1 f^{(0)},
\qquad
\Delta_2 f^{(2)} = \sigma^{(2)} - L_2 f^{(1)}.
```

The code solves the transport and parallel-current systems together whenever
they share the same factorization. That is why `_solve_modes(...)` uses
stacked right-hand sides for several of the LU solves.

## Linear Algebra Choices

NTX uses:

- `jax.scipy.linalg.lu_factor`
- `jax.scipy.linalg.lu_solve`

throughout the dense solve. It does **not** form explicit inverses.

That choice is visible in:

- `_solve_modes(...)`
- `compile_prepared_solver(...)`

in [`src/ntx/solver.py`](../src/ntx/solver.py).

## Complexity

Let

```{math}
N_\mathrm{fs} = N_\theta N_\zeta.
```

Then the dense block solve scales as:

- linear in `N_\xi`
- cubic in `N_\mathrm{fs}`

or, schematically,

```{math}
\mathcal O(N_\xi N_\mathrm{fs}^3).
```

That is why NTX focuses on:

- keeping the Legendre recursion linear in `N_\xi`
- reusing prepared geometry for scans
- exposing separate serial-batched and multiprocess throughput lanes

## JAX Usage

NTX uses JAX in two distinct ways.

### Differentiable Imported Lane

The imported lane is designed for:

- `jit`
- `vmap`
- `grad`
- in-memory scans
- NEOPAX coupling

Key entry points:

- `solve_monoenergetic(...)`
- `solve_monoenergetic_scan(...)`
- `compile_prepared_solver(...)`
- `build_ntx_neopax_scan(...)`

### Non-Differentiable Throughput Lane

The multiprocess execution path in [`src/ntx/parallel.py`](../src/ntx/parallel.py)
is intentionally not differentiable. It exists to improve throughput on larger
production scans by running one worker process per device.

Use this lane when:

- scan size is large
- throughput matters more than differentiability
- device isolation is needed for multi-GPU reliability

## Serial, Threaded, And Multiprocess Scan Paths

NTX currently offers three scan styles.

### Serial Batched JAX

`solve_monoenergetic_scan(...)`

- default
- differentiable
- usually the fastest choice for small and medium scans

### Single-Process Device-Sharded Scan

`solve_monoenergetic_parallel_scan(...)`

- uses local healthy devices
- remains in one process
- useful when the local device stack is stable

### Multiprocess Scan

`solve_monoenergetic_multiprocess_scan(...)`

- one worker process per CPU or GPU device
- robust for multi-GPU isolation
- best treated as the throughput lane for larger jobs

The measured crossover behavior is summarized in
[Performance](performance.md).

## Convergence Guidance

The thesis shows that the hardest part of the solve is often `D31` at low
collisionality. NTX therefore follows the same practical logic:

1. converge `N_xi` first at low `\hat \nu`
2. then increase `N_theta` and `N_zeta` until `D31` is stable
3. check `|D13 + D31|` as an Onsager sanity test

The W7-X and CIEMAT-QI studies in the thesis suggest that low-collisionality
production runs can require `N_xi` in the `140-180` range and significantly
larger angular grids than the bundled examples.

The bundled examples are intentionally smaller because they are meant for:

- quick inspection
- tests
- documentation
- figure generation

not for final production convergence claims on arbitrary equilibria.
