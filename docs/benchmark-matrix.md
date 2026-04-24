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

Current promoted validation gates are:

| Gate | Scope | Primary artifact |
| --- | --- | --- |
| Monoenergetic validation summary | coefficient behavior, Onsager residuals, and Legendre convergence | `docs/_static/validation_summary.json` |
| Precise-QS Redl/SFINCS comparison | fixed-field Redl agreement on the interior benchmark window | `docs/_static/bootstrap_current_fixed_field_validation.json` |
| W7-X integrated transfer | imported workflow transfer on the rebuilt raw branch | `docs/_static/bootstrap_current_reference_audit_w7x.json` |
| Prepared derivative path | implicit-adjoint derivative agreement gate and timing evidence | `docs/_static/derivative_path_benchmark.json` |

Current stress gates are:

| Gate | Why It Stays Open |
| --- | --- |
| Fixed-field `NTX+NEOPAX` closure stress | the mismatch is a reduced-closure issue, not a solved parity claim |
| Synthetic inverse-design recovery | useful differentiable workflow check, but too small to be a research-grade geometry claim |
| Three-harmonic geometry-control derivatives | direct AD/finite-difference audit on an owned surface; real VMEC/Boozer geometry controls remain open |
| File-backed geometry-control derivatives | sample Boozer and VMEC files now pass AD/finite-difference checks, but reusable geometry-family controls remain open |
| Boundary forward-mode current derivatives | low-dimensional boundary controls now reach NTX and NTX+NEOPAX outputs on boundary-projected geometry; this stays as the fast precursor lane |
| Implicit-equilibrium forward-mode derivatives | the implicit residual solve reaches Boozer geometry and NTX transport on the committed QA case, but only equilibrium volume matches centered finite differences; Boozer, transport, integrated current, and reverse mode remain open |
| Explicit-relaxed boundary current derivatives | committed QA and QH cases now pass the self-consistent forward-mode audit, but additional families plus implicit/reverse-mode equilibrium sensitivities remain open |
| Profile uncertainty propagation | validates the current workflow mechanics, but needs broader profile bases |
| Robust bootstrap-current optimization | useful robust-design stress test, but not yet broad enough for a promoted physics claim |
| Primitive-profile force reconstruction | literature-profile audit, currently monitored rather than promoted |

Planned lanes that must stay visible are:

| Lane | Required Before Promotion |
| --- | --- |
| Full monoenergetic geometry-family reproduction | reusable W7-X EIM, W7-X KJM, and CIEMAT-QI inputs; `D11`, `D31`, `D33` parity; `N_xi`, `N_theta`, `N_zeta` convergence ladders |
| Larger geometry-control autodiff | broaden the current analytic and file-backed audits into reusable geometry families; add direct autodiff, implicit-adjoint, and finite-difference agreement on that basis |
| Hidden-symmetry and omnigenous families | owned input families and convergence gates before adding research-grade figures |
| QI and piecewise-omnigenous low-bootstrap families | owned input families; `D11`, `D31`, `D33`, bootstrap-current proxy, and radial-profile convergence; comparison to published qualitative ordering before any design claim |
| Implicit-equilibrium sensitivity transfer | Boozer and NTX transport observables must match centered finite differences, not only equilibrium volume |
| Performance and memory crossover maps | compile/steady-state split, resident memory, device memory, and CPU/GPU/multiprocess crossover on production grids |
| PyPI release readiness | standard-index dependency surface, wheel/sdist smoke tests, Trusted Publishing configuration, and release artifact provenance |
