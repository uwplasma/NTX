# Physics Model

NTX solves the monoenergetic neoclassical transport problem on a single flux
surface using the Legendre-space formulation developed in Javier Escoto's PhD
thesis, [Fast monoenergetic neoclassical transport coefficients in
stellarators](https://arxiv.org/abs/2510.27513).

This page documents the physical model that NTX actually implements, the
normalizations that appear in the code, and the source files in which each
piece is assembled.

## Scope Of The Model

NTX solves the monoenergetic drift-kinetic equation with:

- fixed particle speed
- Lorentz pitch-angle scattering
- Boozer-angle spatial coordinates on one flux surface
- an externally specified radial electric field through `epsi_hat` or `er_hat`
- two right-hand sides:
  - the radial-transport drive
  - the parallel-conductivity / bootstrap-current drive

The present code therefore computes the monoenergetic geometric coefficients

```{math}
\hat D_{11},\quad \hat D_{31},\quad \hat D_{13},\quad \hat D_{33},
\quad \hat D_{33,\mathrm{Sp}}.
```

These are the quantities exposed in [`src/ntx/transport.py`](../src/ntx/transport.py)
and surfaced in [`src/ntx/solver.py`](../src/ntx/solver.py).

NTX does **not** attempt to solve the full multi-species ambipolar problem by
itself. That layer is delegated to downstream tools such as
[NEOPAX](https://github.com/uwplasma/NEOPAX), which consume the NTX
monoenergetic tables.

## Coordinate And Field Representation

On a single flux surface, the magnetic-field strength is represented in Boozer
angles as

```{math}
B(\theta,\zeta)
=
\sum_{m,n}
B_{mn}
\cos\!\left(m\theta + n N_\mathrm{fp}\zeta\right)
```

for stellarator-symmetric surfaces, or with additional sine coefficients when
those are provided.

This is implemented in:

- [`src/ntx/geometry.py`](../src/ntx/geometry.py):
  `evaluate_fourier_series(...)`
- [`src/ntx/booz.py`](../src/ntx/booz.py):
  Boozer-file loading
- [`src/ntx/vmec.py`](../src/ntx/vmec.py):
  VMEC `wout` loading through `vmec_jax`

The spatial geometry object stored on the angular grid is
`GeometryOnGrid`, defined in [`src/ntx/geometry.py`](../src/ntx/geometry.py).

## Geometric Quantities Used By The Solver

For Boozer-surface inputs, NTX constructs

```{math}
\mathcal J
=
\frac{|B_\zeta + \iota B_\theta|}{B^2},
\qquad
B^\theta = \frac{\iota}{\mathcal J},
\qquad
B^\zeta = \frac{1}{\mathcal J}.
```

For VMEC-derived surfaces, NTX evaluates Fourier series for

- `B`
- `\mathcal J`
- `B_\theta`
- `B_\zeta`
- `B^\theta`
- `B^\zeta`

directly on the angular grid.

The magnetic-drift factor that enters the transport source is

```{math}
\hat v_m(\theta,\zeta)
=
\frac{B_\theta \,\partial_\zeta B - B_\zeta \,\partial_\theta B}
{\mathcal J B^3}.
```

This quantity is assembled in [`src/ntx/geometry.py`](../src/ntx/geometry.py)
inside `_boozer_geometry_on_grid(...)` and `_vmec_geometry_on_grid(...)`.

## Monoenergetic Legendre System

The unknown distribution is expanded in Legendre modes of pitch angle,

```{math}
f(\theta,\zeta,\xi)
=
\sum_{k=0}^{N_\xi} f^{(k)}(\theta,\zeta) P_k(\xi).
```

NTX then solves the block-tridiagonal hierarchy

```{math}
L_k f^{(k-1)} + D_k f^{(k)} + U_k f^{(k+1)} = s^{(k)}.
```

This is the exact operator form implemented in
[`src/ntx/operators.py`](../src/ntx/operators.py) and solved in
[`src/ntx/solver.py`](../src/ntx/solver.py).

The lower, diagonal, and upper blocks contain:

```{math}
L_k &=
\frac{k}{(2k-1)B}
\left(
B^\theta \partial_\theta + B^\zeta \partial_\zeta
\right)
\;+\;
\frac{k(k-1)}{2(2k-1)B^2}
\left(B^\zeta \partial_\zeta B + B^\theta \partial_\theta B\right), \\
D_k &=
\frac{\hat E_\psi}{\mathcal J \langle B^2\rangle}
\left(-B_\zeta \partial_\theta + B_\theta \partial_\zeta\right)
\;+\;
\frac{\hat \nu}{2}k(k+1), \\
U_k &=
\frac{k+1}{(2k+3)B}
\left(
B^\theta \partial_\theta + B^\zeta \partial_\zeta
\right)
\;+\;
\frac{-(k+1)(k+2)}{2(2k+3)B^2}
\left(B^\zeta \partial_\zeta B + B^\theta \partial_\theta B\right).
```

The exact coefficient arrays corresponding to these formulas are constructed in
`coefficients_for_k(...)` in [`src/ntx/operators.py`](../src/ntx/operators.py).

## Source Systems

NTX solves two linear systems per monoenergetic case:

1. a transport-driven system used for `D11` and `D31`
2. a parallel-current / conductivity system used for `D13` and `D33`

In code these are the source arrays `s1` and `s3` from
`source_modes(...)` in [`src/ntx/operators.py`](../src/ntx/operators.py).

Their Legendre support is sparse:

- `s1` occupies modes `k = 0` and `k = 2`
- `s3` occupies mode `k = 1`

That structure is central to the NTX solver design: only the low-order modes
`k = 0,1,2` are required in the backward substitution, even when the forward
Schur elimination proceeds to much larger `N_\xi`.

## Nullspace Constraint

The monoenergetic system has the usual constant-mode nullspace in the `k = 0`
equation. NTX removes it by replacing the first spatial row with

```{math}
f^{(0)}(\theta_0,\zeta_0)=0.
```

That operation is implemented by `apply_nullspace_condition(...)` in
[`src/ntx/operators.py`](../src/ntx/operators.py).

## Transport Coefficients

Once the low-order Legendre modes are known, NTX evaluates the monoenergetic
coefficients through flux-surface averages. In the code:

```{math}
\hat D_{11}
=
\frac{\left\langle
-2 \hat v_{m,0} f_{1,0}
-\frac{2}{5}\hat v_{m,2} f_{1,2}
\right\rangle}{\Psi_\mathrm{coeff}^2},
```

```{math}
\hat D_{31}
=
\frac{\left\langle \frac{2}{3} B f_{1,1}/B_0 \right\rangle}
{\Psi_\mathrm{coeff}},
\qquad
\hat D_{13}
=
\frac{\left\langle
-2 \hat v_{m,0} f_{3,0}
-\frac{2}{5}\hat v_{m,2} f_{3,2}
\right\rangle}
{\Psi_\mathrm{coeff} B_0},
```

```{math}
\hat D_{33}
=
\frac{\left\langle \frac{2}{3} B f_{3,1}/B_0 \right\rangle}{B_0}.
```

NTX also computes the Spitzer reference contribution

```{math}
\hat D_{33,\mathrm{Sp}}
=
\frac{2}{3\hat \nu}
\left\langle \frac{B^2}{B_0^2} \right\rangle,
```

which is the expression quoted in the thesis appendix and implemented in
`coefficients_from_modes(...)` in
[`src/ntx/transport.py`](../src/ntx/transport.py).

## Onsager Symmetry

The monoenergetic formulation satisfies the expected Onsager relation

```{math}
\hat D_{13} = -\hat D_{31}
```

under the assumptions of the model. NTX tracks the scalar residual

```{math}
|\hat D_{13} + \hat D_{31}|
```

through `onsager_error(...)` in
[`src/ntx/transport.py`](../src/ntx/transport.py).

This quantity is written into the CLI `.npz` outputs and exposed in the
high-level `TransportResult`.

## Electric-Field Normalization

NTX accepts either `epsi_hat` directly or `er_hat` together with a
surface-dependent normalization.

For Boozer / DKES-style surfaces,

```{math}
\hat E_\psi = \hat E_r / \psi_p.
```

For VMEC surfaces, NTX computes

```{math}
r_n = \sqrt{\psi_n},
\qquad
\hat r = a_\mathrm{minor} r_n,
\qquad
\frac{d\hat \psi}{d\hat r}
=
\frac{2 \hat \psi_a r_n}{a_\mathrm{minor}},
\qquad
\hat E_\psi = \hat E_r \left(\frac{d\hat \psi}{d\hat r}\right)^{-1}.
```

This conversion is implemented in:

- [`src/ntx/vmec.py`](../src/ntx/vmec.py) for file-based VMEC inputs
- [`src/ntx/solver.py`](../src/ntx/solver.py) in
  `MonoenergeticCase.resolved_epsi_hat(...)`

## What Downstream Tools Consume

Downstream bootstrap-current or ambipolar-transport tools typically consume
`D11`, `D13`, and `D33` over radial, collisionality, and electric-field scans.

NTX provides that through:

- `solve_monoenergetic_scan(...)`
- `build_ntx_neopax_scan(...)`
- `scan_to_neopax_arrays(...)`

in [`src/ntx/solver.py`](../src/ntx/solver.py) and
[`src/ntx/neopax.py`](../src/ntx/neopax.py).

## Closure-Model Gates

The monoenergetic coefficient pipeline and the downstream momentum-restoring
closure have to be validated separately.

For the closure side, the physically relevant gates are:

1. **Coefficient-side invariants stay closed first.**
   The monoenergetic solver must continue to satisfy the Onsager relation and
   the established database normalization bridge before any closure change is
   considered.
2. **The observable map stays fixed.**
   For the present Sonine/Laguerre basis, the corrected parallel flow is

   ```{math}
   U_{\parallel a} = n_a c_{a0},
   ```

   so higher-order closure work must change the solved moment system, not the
   final observable by an empirical remap.
3. **Finite-order closure must recover the current three-moment system exactly
   at `P=2`.**
   Any arbitrary-order implementation has to reproduce the legacy
   `P=2` Sonine basis, normalization factors, and source projections before it
   is used to interpret benchmark differences.
4. **The closure must preserve symmetry properties implied by the projected
   theory.**
   In practice this means retaining the Onsager/ambipolar structure emphasized
   by Sugama--Horton and Sugama--Nishimura, rather than inserting benchmark-fit
   constants into selected matrix entries.
5. **Benchmark transfer is mandatory.**
   A closure change is only acceptable if it improves the precise-QS fixed-field
   QA/QH current benchmark without regressing the already-validated integrated
   W7-X workflow.

These gates are now the design constraints for any higher-order closure work in
the imported `NTX+NEOPAX` path.
