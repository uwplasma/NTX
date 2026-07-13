# Resolution and convergence

An algebraically converged linear solve is not necessarily a converged
drift-kinetic discretization. NTX treats three checks separately:

1. the residual of the block equations;
2. Fourier sampling of the retained geometry harmonics;
3. convergence of transport coefficients under angular and Legendre refinement.

## Residual semantics

The production solve eliminates the high Legendre tail by Schur complements
and returns the three modes needed by the transport moments.
`TransportResult.residual_l2` is the root-mean-square residual of that complete
tail-eliminated system, evaluated from its pivoted LU factors. It includes the
high-mode Schur contribution to row 2 without reconstructing another mode.

The test suite also factors and reconstructs every Legendre mode on a small
system, applies every original block row independently, and requires both
residuals to reach roundoff. Full-mode reconstruction scales as
`O(N_xi N_fs^2)` storage, so it is an audit path rather than the default
production diagnostic.

Run that audit explicitly on a prepared system:

```python
from ntx import audit_prepared_residuals

residuals = audit_prepared_residuals(prepared, case)
print(residuals.tail_eliminated_l2)
print(residuals.full_system_l2)
print(residuals.retained_mode_max_abs_error)
```

## Geometry sampling

NTX evaluates each retained harmonic as

```{math}
\cos(m\theta+nN_{fp}\zeta)
```

on one toroidal field period. Consequently, alias-free collocation requires

```{math}
N_\theta \ge 2\max|m|+1,
\qquad
N_\zeta \ge 2\max|n|+1.
```

The odd floor avoids treating the real Nyquist mode of an even grid as a
resolved derivative mode. Inspect a proposed grid before a solve:

```python
from ntx import geometry_resolution_report

report = geometry_resolution_report(surface, grid)
report.require_resolved()
print(report.theta_oversampling, report.zeta_oversampling)
```

The adaptive convergence API enforces this gate. For a one-off solve, request
strict pre-solve validation explicitly:

```python
result = solve_monoenergetic(
    surface, grid, case, require_resolved_geometry=True
)
```

The default remains permissive so deliberately underresolved smoke diagnostics
and historical reference reproductions remain runnable, but they are not
research-grade convergence claims. If a surface is passed as a dynamic argument
to `jax.jit`, call the report host-side first because traced integer harmonic
arrays cannot be converted into a host report.

Nyquist adequacy is necessary, not sufficient. Products, reciprocals, and
derivatives of the magnetic field can demand angular oversampling beyond the
retained input bandwidth. NTX therefore warns below an oversampling ratio of
1.5 and still requires an angular convergence ladder.

## Adaptive coefficient gate

Use independent angular and pitch-angle ladders:

```python
from ntx import solve_monoenergetic_converged

audit = solve_monoenergetic_converged(
    surface,
    case,
    angular_resolutions=((13, 15), (17, 21), (21, 25), (25, 31)),
    legendre_orders=(16, 24, 32, 48, 63),
    rtol=1e-2,
    atol=(1e-10, 1e-10, 1e-10),  # D11, D31, D33
)
if audit.status != "converged":
    raise RuntimeError(audit.message)
```

For each coefficient `c`, a refinement is accepted when

```{math}
|c_h-c_{h^-}| \le a_c
+ r\max(|c_h|,|c_{h^-}|).
```

The absolute term is essential for symmetry-conditioned `D31` values near
zero. The default research policy requires two consecutive accepted angular
changes and two consecutive accepted Legendre changes. Exhausting a ladder
returns `unresolved`; the last result is retained for diagnosis but is not
silently promoted. Non-finite coefficients return `model-out-of-scope`.

These numerical gates follow Fourier collocation sampling theory and the
angular/Legendre convergence practice used for monoenergetic neoclassical
transport calculations. They do not replace independent-code validation or a
separate model-discrepancy tolerance.
