# Benchmark Matrix

NTX keeps validation claims in a maintained benchmark matrix. The matrix maps
each claim or monitored stress lane to:

- literature anchors,
- scripts,
- tests,
- committed artifacts,
- manuscript figures,
- and non-promoted future work that must not be promoted yet.

Generate the machine-readable artifact with:

```bash
python scripts/build_benchmark_matrix.py
```

The default output is:

```text
docs/_static/benchmark_matrix.json
```

The matrix has four maturity levels:

- `positive-gate`: a promoted validation or transfer claim.
- `stress-gate`: a monitored physics or workflow stress test.
- `software-gate`: a software-performance or maintenance gate.
- `planned-lane`: a literature-motivated lane that is intentionally not yet a
  validation claim.

Current machine-checked acceptance gates are:

| Gate | Scope | Primary artifact |
| --- | --- | --- |
| Monoenergetic validation summary | coefficient behavior, Onsager residuals, and Legendre convergence | `docs/_static/validation_summary.json` |
| Precise-QS Redl/SFINCS comparison | fixed-field Redl agreement on the interior benchmark window | `docs/_static/bootstrap_current_fixed_field_validation.json` |
| W7-X integrated transfer | imported workflow transfer on the rebuilt raw branch | `docs/_static/bootstrap_current_reference_audit_w7x.json` |
| Prepared derivative path | implicit-adjoint derivative agreement gate and timing evidence | `docs/_static/derivative_path_benchmark.json` |
| Geometry/boundary derivative agreement | finite-difference agreement on analytic, file-backed, boundary-projected, and explicit-relaxed derivative artifacts | `docs/_static/*derivative_benchmark.json` |
| Fixed-field `NTX+NEOPAX` closure stress | precise-QS total-current stress gate below `1e-1`, scoped away from species-current parity | `docs/_static/bootstrap_current_fixed_field_validation.json` |

Current software gates are:

| Gate | Scope | Primary artifact |
| --- | --- | --- |
| CPU/GPU throughput and strong-scaling characterization | serial, device-parallel, multiprocess, CPU, and GPU scan scaling on committed smoke, heavier, production, and fixed-workload strong-scaling grids | `docs/_static/performance_strong_scaling_production.png` |
| Prepared-geometry reuse profile | direct repeated solves, prepared geometry reuse, and compiled prepared steady-state reuse on one fixed geometry | `docs/_static/prepared_geometry_reuse_profile.json` |

Current stress gates are:

| Gate | Current Non-Promoted Scope |
| --- | --- |
| Fixed-field species-current closure parity | the total current passes the scoped stress gate, but species-resolved current decomposition and broader closure defaults remain reduced-closure issues |
| Synthetic inverse-design recovery | useful differentiable workflow check, but too small to be a research-grade geometry claim |
| Three-harmonic geometry-control derivatives | direct AD/finite-difference audit is now machine checked on an owned surface; reusable geometry families remain open |
| File-backed geometry-control derivatives | sample Boozer and VMEC files now pass machine-checked AD/finite-difference thresholds, but reusable geometry-family controls remain open |
| Boundary forward-mode current derivatives | low-dimensional boundary controls now pass the machine-checked boundary-projected finite-difference audit; self-consistent shipping claims use the explicit-relaxed lane |
| Implicit-equilibrium forward-mode derivatives | retained as a non-shipping diagnostic: equilibrium volume matches centered finite differences, but residual contraction and Boozer/NTX tangent parity do not pass |
| Explicit-relaxed boundary current derivatives | committed QA and QH cases now pass the machine-checked self-consistent forward-mode audit, but additional families plus reverse-mode equilibrium sensitivities remain open |
| Artifact-backed geometry-family breadth summary | analytic, file-backed, boundary-projected, explicit-relaxed, and implicit-volume derivative artifacts are summarized in one figure, while retired implicit Boozer/transport diagnostics are excluded from promoted geometry-family claims |
| VMEC geometry-family transport convergence | public VMEC example families now have a committed production-grid `D11/D31/D33` convergence stress artifact with `D13`/Onsager diagnostics retained; independent-code parity and radial/electric-field/collisionality promotion remain separate gates |
| Same-coordinate Boozer-file round trip | generated `boozmn` surfaces now reload on VMEC half-grid coordinates and reproduce the in-memory `vmex -> booz_xform_jax -> NTX` transport coefficients; VMEC-harmonic versus Boozer-coordinate comparisons remain representation audits |
| Finite-beta finalized-wout Boozer transfer | optimized finite-beta `wout` magnetic channels now transform and reload through the direct `boozmn` backend to roundoff on the same VMEC half-grid surfaces; the fully differentiable finite-beta state path remains non-shipping for unsupported current-profile representations |
| Profile uncertainty propagation | three-term radial-basis covariance propagation and Fisher/HVP consistency are machine checked; cross-geometry profile families remain open |
| Bootstrap-current optimization | machine-checked weighted-current improvement on the committed W7-X study, but not yet broad enough for a stellarator-design claim |
| Robust bootstrap-current optimization | useful robust-design stress test, but not yet broad enough for a promoted physics claim |
| Primitive-profile force reconstruction | literature-profile audit, currently monitored rather than promoted |
| Owned finite-beta JAX-native NTX+NEOPAX dataset provenance | finite-beta input/wout scan generation, physical VMEC edge-flux normalization in the Boozer path, and interpolation-path control are now artifact-backed; optimized finite-beta QH/QI Boozer reconstruction remains an explicit geometry-backend blocker |
| Owned finite-beta SFINCS-JAX generation contract | same-grid finite-beta SFINCS-JAX input generation, six-point completed HDF5 ladder ingestion including the profile-current stress neighborhood, exact radial interpolation, PAS `nuD` bridge, coefficient-level NTX comparison, a `35 x 43 x 48` production stress-radius resolution/harmonic-cutoff probe, a completed six-point production radial/collisionality coefficient ladder, and an accepted high-`Nxi` RHSMode=1 pitch stress gap below `1.5e-1` are artifact-backed |
| Owned finite-beta Redl and `NTX+NEOPAX` bootstrap-current stress | same finite-beta VMEC wout, Boozer transform, normalized-radius `B00(rho)` field convention, analytic profile contract, production radial/collisionality ladder, adaptive physical `nu/v` support, `D33_spitzer` audit branch, Sonine-order convergence sidecar, coefficient/profile localization sidecar, profile-current observable sidecar, current-conditioning sidecar, closure-quadrature sidecar, source-channel sidecar, profile source-response sidecar, closure-target driver sidecar, and production SFINCS-JAX coefficient ladder sidecar are artifact-backed; this is closed as a reduced-closure stress benchmark with the current high-order source response classified as mixed density/electric and temperature-gradient physics, not as a broad full-collision parity claim |

Planned lanes are not release blockers. They stay visible so future work has
clear promotion criteria instead of drifting into unsupported claims:

| Lane | Required Before Promotion |
| --- | --- |
| Full monoenergetic geometry-family reproduction | production-resolution independent-code parity for the available W7-X EIM/EJM, QI, QA/QH, and stellarator-family inputs; owned W7-X KJM input; radial/electric-field/collisionality ladders |
| Larger geometry-control autodiff | broaden the current analytic and file-backed audits into reusable geometry families; add direct autodiff, implicit-adjoint, and finite-difference agreement on that basis |
| Hidden-symmetry and omnigenous families | owned input families and convergence gates before adding research-grade figures |
| QI and piecewise-omnigenous low-bootstrap families | owned input families; `D11`, `D31`, `D33`, reduced bootstrap-current response, and radial-profile convergence; comparison to published qualitative ordering before any design claim |
| Implicit-equilibrium sensitivity transfer | restore only after the backend residual solve contracts and Boozer/NTX transport observables match centered finite differences |
| Performance and memory crossover maps | repeat the production grid on additional GPU nodes and add device-memory timelines for larger VMEC-family workloads |
