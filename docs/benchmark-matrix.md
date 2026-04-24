# Benchmark Matrix

NTX keeps validation claims in a maintained benchmark matrix. The matrix maps
each claim or monitored stress lane to:

- literature anchors,
- scripts,
- tests,
- committed artifacts,
- manuscript figures,
- and open work that must not be promoted yet.

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

Current software gates are:

| Gate | Scope | Primary artifact |
| --- | --- | --- |
| CPU/GPU throughput characterization | serial, multiprocess, CPU, and GPU scan scaling on committed smoke and heavier grids | `docs/_static/performance_scaling_heavy.png` |
| Prepared-geometry reuse profile | direct repeated solves, prepared geometry reuse, and compiled prepared steady-state reuse on one fixed geometry | `docs/_static/prepared_geometry_reuse_profile.json` |

Current stress gates are:

| Gate | Why It Stays Open |
| --- | --- |
| Fixed-field `NTX+NEOPAX` closure stress | the mismatch is a reduced-closure issue, not a solved parity claim |
| Synthetic inverse-design recovery | useful differentiable workflow check, but too small to be a research-grade geometry claim |
| Three-harmonic geometry-control derivatives | direct AD/finite-difference audit is now machine checked on an owned surface; reusable geometry families remain open |
| File-backed geometry-control derivatives | sample Boozer and VMEC files now pass machine-checked AD/finite-difference thresholds, but reusable geometry-family controls remain open |
| Boundary forward-mode current derivatives | low-dimensional boundary controls now pass the machine-checked boundary-projected finite-difference audit; self-consistent shipping claims use the explicit-relaxed lane |
| Implicit-equilibrium forward-mode derivatives | retained as a non-shipping diagnostic: equilibrium volume matches centered finite differences, but residual contraction and Boozer/NTX tangent parity do not pass |
| Explicit-relaxed boundary current derivatives | committed QA and QH cases now pass the machine-checked self-consistent forward-mode audit, but additional families plus reverse-mode equilibrium sensitivities remain open |
| Artifact-backed geometry-family breadth summary | analytic, file-backed, boundary-projected, explicit-relaxed, and implicit-volume derivative artifacts are summarized in one figure, while retired implicit Boozer/transport diagnostics are excluded from promoted geometry-family claims |
| VMEC geometry-family transport convergence | public VMEC example families now have a committed `D11/D31/D33` convergence stress artifact; independent-code parity and paper-resolution promotion remain separate gates |
| Profile uncertainty propagation | validates the current workflow mechanics, but needs broader profile bases |
| Bootstrap-current optimization | machine-checked weighted-current improvement on the committed W7-X study, but not yet broad enough for a stellarator-design claim |
| Robust bootstrap-current optimization | useful robust-design stress test, but not yet broad enough for a promoted physics claim |
| Primitive-profile force reconstruction | literature-profile audit, currently monitored rather than promoted |

Planned lanes that must stay visible are:

| Lane | Required Before Promotion |
| --- | --- |
| Full monoenergetic geometry-family reproduction | paper-resolution independent-code parity for the available W7-X EIM/EJM, QI, QA/QH, and stellarator-family inputs; owned W7-X KJM input; radial/electric-field/collisionality ladders |
| Larger geometry-control autodiff | broaden the current analytic and file-backed audits into reusable geometry families; add direct autodiff, implicit-adjoint, and finite-difference agreement on that basis |
| Hidden-symmetry and omnigenous families | owned input families and convergence gates before adding research-grade figures |
| QI and piecewise-omnigenous low-bootstrap families | owned input families; `D11`, `D31`, `D33`, bootstrap-current proxy, and radial-profile convergence; comparison to published qualitative ordering before any design claim |
| Implicit-equilibrium sensitivity transfer | restore only after the backend residual solve contracts and Boozer/NTX transport observables match centered finite differences |
| Performance and memory crossover maps | compile/steady-state split, resident memory, device memory, and CPU/GPU/multiprocess crossover on production grids |
| PyPI release readiness | standard-index dependency surface, wheel/sdist smoke tests, Trusted Publishing configuration, and release artifact provenance |
