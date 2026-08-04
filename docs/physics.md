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
  VMEC `wout` loading through `vmex`

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

## Drift-Kinetic Equation Treated By NTX

NTX solves the local monoenergetic drift-kinetic equation for the
non-adiabatic response on a fixed flux surface,

```{math}
\left[
\xi \frac{1}{B}\left(B^\theta\partial_\theta+B^\zeta\partial_\zeta\right)
+ \frac{\hat E_\psi}{\mathcal J\langle B^2\rangle}
\left(-B_\zeta\partial_\theta+B_\theta\partial_\zeta\right)
- \frac{1-\xi^2}{2B^2}
\left(B^\theta\partial_\theta B+B^\zeta\partial_\zeta B\right)\partial_\xi
- C_L
\right] f
= s.
```

Here `\xi = v_\parallel / v` is pitch angle and `C_L` is the Lorentz
pitch-angle-scattering operator,

```{math}
C_L f
=
\frac{\hat\nu}{2}
\partial_\xi\left[(1-\xi^2)\partial_\xi f\right].
```

The model is monoenergetic: speed is fixed, energy scattering is not retained,
and the full species/profile problem is outside this local solve. NTX treats
two source systems, one for radial transport and one for parallel flow. Those
monoenergetic responses are later integrated or mapped by profile and NEOPAX
workflows when a species-resolved bootstrap-current calculation is needed.

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

This quantity is written into the CLI NetCDF/NPZ/HDF5 outputs and exposed in the
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

The public database bridge defaults to the raw `D33` convention because that is
the branch that preserves the integrated W7-X transfer gate. The
`D33_spitzer` branch remains available as an explicit fixed-field closure
stress option, but it is not a global runtime default.

## Database Bridge And Bootstrap-Current Observable

The NTX-to-NEOPAX bridge is a normalization bridge between two well-defined
physics objects:

1. NTX computes monoenergetic geometric coefficients on a flux surface.
2. The downstream energy convolution evaluates species transport matrices from
   those coefficients and the thermodynamic forces.
3. The momentum-restoring closure solves a finite Sonine moment system for the
   corrected parallel-flow moments.
4. The bootstrap-current density is assembled from the species parallel flows.

No fitted scalar is used in this chain.

### Thermodynamic-Force Form

For species `a`, the reduced closure uses the linear-response form

```{math}
\begin{pmatrix}
\Gamma_a/n_a \\
Q_a/(n_a T_a) \\
U_{\parallel a}/n_a
\end{pmatrix}
=
-
\begin{pmatrix}
L_{11,a} & L_{12,a} & L_{13,a} \\
L_{21,a} & L_{22,a} & L_{23,a} \\
L_{31,a} & L_{32,a} & L_{33,a}
\end{pmatrix}
\begin{pmatrix}
A_{1,a} \\
A_{2,a} \\
A_3
\end{pmatrix}.
```

Here `A1` is the density/electrostatic drive, `A2` is the temperature-gradient
drive, and `A3` is the parallel-electric-field drive. This separation matters
for validation: the fixed-field bootstrap current is not determined by a single
monoenergetic coefficient, but by how the density and temperature drives enter
the row-3 response and the momentum-restoring Sonine solve.

For the finite-beta Redl stress audit, the analytic target is also stored as
source terms on the same current observable,

```{math}
\langle J\cdot B\rangle_\mathrm{Redl}
=
J_n + J_{T_e} + J_{T_i},
```

where `J_n` is the Redl density-gradient contribution and `J_Te + J_Ti` is the
effective temperature-gradient target. The source-channel audit compares these
terms to the frozen NTX+NEOPAX density/electric and effective-temperature
responses without inserting any fitted multiplier into the runtime closure.

The no-momentum parallel-flow branch is therefore

```{math}
U_{\parallel a}^{(0)}
=
-n_a\left(L_{31,a} A_{1,a}
        +L_{32,a} A_{2,a}
        +L_{33,a} A_3\right).
```

The momentum-restoring branch solves for Sonine coefficients `c_{ak}`. In the
current finite basis the observable is fixed by the moment definition:

```{math}
U_{\parallel a}=n_a c_{a0}.
```

This is why NTX keeps the observable map fixed in validation. Changing a fitted
linear combination of `c_{a0}, c_{a1}, ...` can improve one benchmark but would
change the physical moment definition. Any accepted closure improvement has to
change the projected moment equations or collision blocks, not the current
observable by a benchmark-tuned remap.

### Monoenergetic Coefficient Scaling

The database bridge follows the dimensional role of each coefficient in the
energy convolution:

```{math}
D_{11}^{\mathrm{db}} = D_{11}^{\mathrm{NTX}}\left(\frac{dr}{ds}\right)^2,
\qquad
D_{13}^{\mathrm{db}} = D_{13}^{\mathrm{NTX}}\frac{dr}{ds},
\qquad
D_{33}^{\mathrm{db}} = \nu D_{33}^{\mathrm{NTX}}.
```

The first factor appears twice in `D11` because radial particle/heat transport
contains two radial-gradient normalizations. The mixed `D13` coefficient
contains one radial and one parallel response, so it receives one `dr/ds`
factor. The parallel conductivity coefficient is stored as `nu * D33` because
the consumer reconstructs the monoenergetic conductivity integrand by dividing
by the collisionality used in the energy convolution. This is the exact mapping
implemented in [`src/ntx/_neopax.py`](../src/ntx/_neopax.py).

`D33_spitzer` is not a fitted alternate normalization. It is the analytic
Spitzer-conductivity contribution

```{math}
\hat D_{33,\mathrm{Sp}}
=
\frac{2}{3\hat \nu}
\left\langle\frac{B^2}{B_0^2}\right\rangle,
```

which is the constant-field conductivity limit of the monoenergetic system. The
fixed-field QA/QH stress audit can select this branch explicitly because the
low-order momentum-restoring closure is a parallel-flow conductivity model.
The public bridge still defaults to raw `D33`, because the integrated W7-X
database transfer is validated on that convention.

### Current Assembly And SFINCS Observable

The species current density from the closure is

```{math}
j_{\parallel a}=Z_a e\,U_{\parallel a}.
```

The archived fixed-field reference stores flux-surface-averaged parallel-flow
observables. In the SFINCS output convention,

```{math}
\mathrm{FSABjHat} = \sum_a Z_a\,\mathrm{FSABFlow}_a,
\qquad
\mathrm{FSABjHatOverB0}
=
\frac{\mathrm{FSABjHat}}{\mathrm{B0OverBBar}}.
```

The comparison therefore uses the archived `B0OverBBar` and the imported Boozer
handedness convention to convert the closure current into the same
flux-surface-averaged observable before comparing with the archived profile.
Both pieces are read from the reference output or from the imported geometry;
neither is fitted to the QA/QH current.

The other important implementation point is semantic rather than numerical:
`get_Neoclassical_Fluxes_With_Momentum_Correction(...)` returns the total
corrected `U_parallel`. It is not an increment to add to the no-momentum
`U_parallel`. Treating it as an increment double-counts the no-momentum branch
and produces an unphysical current. The shipped examples and validation scripts
now use the corrected return directly.

### Why The Fixed-Field Gate Is A Stress Gate

The precise-QS fixed-field comparison now passes the total-current stress gate
using only physics-derived operations:

- exact NTX coefficient normalizations,
- explicit `D33_spitzer` conductivity branch for the fixed-field stress model,
- SFINCS observable conversion from archived `B0OverBBar`,
- total corrected `U_parallel` semantics,
- no fitted constants and no species-dependent bridge factors.

It is still not promoted as species-current parity. Continuum drift-kinetic
solvers with fuller linearized Fokker-Planck physics can resolve
species-current components that a low-order momentum-restoring closure built
from monoenergetic coefficients is not guaranteed to reproduce coefficient by
coefficient. The promoted statement is therefore deliberately narrower:
the total bootstrap-current profile is inside the documented fixed-field stress
gate while the independent Redl comparison and the integrated W7-X raw-branch
transfer remain separate validation gates.

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
5. **The projected collision model must preserve the collisional invariants.**
   Particle number, parallel momentum, and energy must remain exact invariants
   of the finite-order collisional system, and the weighted collisional form
   should retain the self-adjoint structure that underlies Onsager symmetry and
   non-negative entropy production.
6. **Benchmark transfer is mandatory.**
   A closure change is only acceptable if it improves the precise-QS fixed-field
   QA/QH current benchmark without regressing the already-validated integrated
   W7-X workflow.

These gates are now the design constraints for any higher-order closure work in
the imported `NTX+NEOPAX` path. The runnable summary of the artifact-backed
gates is in [`physics-gates.md`](physics-gates.md) and
`python scripts/check_physics_gates.py`.
