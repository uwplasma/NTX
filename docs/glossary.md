# Glossary

This page fixes the coordinate, normalization, grid, and workflow terminology
used by NTX. Equations and derivations are in [Physics](physics.md).

## Coordinates And Geometry

`psi`
: Toroidal magnetic-flux surface label. File-backed workflows retain the
  source convention and record the mapping used by the solver.

`s`
: Normalized toroidal flux, commonly `s = psi / psi_edge`.

`rho`
: Normalized radius `rho = sqrt(s)`. Derivatives with respect to `rho` are not
  interchangeable with derivatives with respect to physical radius.

`theta`, `zeta`
: Boozer poloidal and toroidal angles used by the local operator. Their Fourier
  sign and field-period conventions are part of the geometry-input contract.

`xi`
: Pitch-angle coordinate `xi = v_parallel / v`, with `-1 <= xi <= 1`.

`B_theta`, `B_zeta`
: Covariant Boozer components of the magnetic field.

`B^theta`, `B^zeta`
: Contravariant Boozer components of the magnetic field.

`J`
: Boozer-coordinate Jacobian `mathcal J`. The identity
  `mathcal J B^2 = B_zeta + iota B_theta` is an active physics gate.

## Normalized Parameters

`nu_hat`
: Monoenergetic collisionality entering the Lorentz pitch-angle-scattering
  operator. It is not a species-profile collision frequency until the
  documented profile mapping is applied.

`epsi_hat`
: Normalized radial-electric-field parameter multiplying tangential
  `E x B` precession in the local equation.

`D11`, `D31`, `D13`, `D33`
: NTX monoenergetic transport coefficients for radial and parallel-flow source
  systems. These are normalized coefficients, not dimensional particle, heat,
  or current fluxes.

`D33_spitzer`
: Parallel-conductivity reference coefficient in the retained collision model.

## Resolution And Diagnostics

`n_theta`, `n_zeta`
: Angular collocation counts on the flux surface. They must resolve retained
  Fourier harmonics and variable-coefficient products.

`n_xi`
: Highest retained Legendre index plus the implementation's documented mode
  convention. Always verify the exact convention in `GridSpec` when comparing
  inputs across codes.

angular oversampling
: Ratio between angular collocation and the strict Nyquist floor of retained
  harmonics. NTX warns below the measured recommended ratio; research claims
  still require two successive refinements.

full-system residual
: Residual of the assembled Legendre-space equations, including reconstructed
  eliminated modes and constraints.

Schur residual
: Residual of the reduced linear system actually solved. It diagnoses the
  numerical solve but does not replace the full-system residual.

Onsager diagnostic
: Reciprocity check between the cross coefficients under the documented source
  and normalization conventions.

## Workflow Terms

prepared geometry
: Geometry-dependent arrays and operator structure assembled once for repeated
  collisionality or electric-field solves.

compiled closure
: Lowered and JIT-compiled callable bound to a prepared system and static shape.
  Changing shapes or static options may trigger recompilation.

downstream closure
: Species, energy, and profile integration performed outside the local
  monoenergetic solve, including bootstrap-current workflows. A downstream
  result must not be described as a directly solved NTX coefficient.

physics gate
: Quantitative acceptance rule anchored to a model identity, convergence study,
  or independent reference. Each promoted gate owns a script, test, artifact,
  threshold, and documentation entry.

stress diagnostic
: Reproducible evidence used to expose a difficult regime without promoting a
  general accuracy claim.
