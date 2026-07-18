# Research Roadmap

This page summarizes research directions and promotion criteria for users.
Implementation order, file ownership, and pull-request acceptance gates are
tracked in a private planning document.

## Current Foundation

NTX provides a JAX-native local monoenergetic solver with:

- analytic, DKES-style, VMEC, and Boozer geometry paths;
- prepared CPU/GPU scans and bounded batching;
- differentiable coefficient and profile workflows;
- NEOPAX-compatible database export and bootstrap-current workflows;
- analytical, convergence, imported-reference, and derivative physics gates;
- machine-readable benchmark and publication-artifact manifests.

Promoted claims and current numerical thresholds are listed in the
[benchmark matrix](benchmark-matrix.md). A completed diagnostic is not
automatically a validated research claim: promotion requires independent
physics evidence, convergence, reusable inputs, tests, and documented
provenance.

## Research Directions

| Direction | Current supported scope | Promotion requirement |
| --- | --- | --- |
| Optimization-grade derivatives | Direct AD, prepared adjoints, geometry controls, and explicit-relaxed boundary sensitivities | AD, centered finite difference, and prepared-adjoint agreement on reusable geometry families with bounded memory |
| Implicit-equilibrium derivatives | Non-shipping diagnostic | Contracted equilibrium residual plus Boozer and NTX transport tangent agreement, not equilibrium-volume agreement alone |
| Profile, UQ, and design workflows | Differentiable reduced profile/current examples | Broader profile bases, uncertainty models, robust objectives, and independent physics checks before stellarator-design claims |
| Geometry-family validation | Owned analytic, finite-beta, QA/QH, and selected file-backed stress families | Production radial, electric-field, collisionality, angular, and Legendre ladders with independent references |
| Bootstrap-current closure | Scoped fixed-field total-current stress comparison and integrated transfer workflow | Physics-derived species-resolved closure that improves broader QA/QH cases without regressing integrated transfer; no fitted bridge constants |
| Production throughput | Prepared reuse, CPU batching, device parallelism, and measured crossover artifacts | Reproducible production-grid runtime and memory maps on named CPU and GPU hardware |
| Physics expansion | Local monoenergetic Lorentz model | A separately derived, tested, and literature-anchored model; do not silently broaden the current solver's claims |

## Validation Standard

A research result is eligible for promotion only when it has:

1. a stated physical model, ordering, normalization, and validity boundary;
2. an owned or permanently accessible input with provenance;
3. angular, Legendre, radial, and parameter convergence appropriate to the
   observable;
4. analytical or independent numerical evidence with aligned physics settings;
5. residual, symmetry, conservation, and positivity checks where applicable;
6. direct tests of coordinate and normalization mappings;
7. autodiff agreement with centered finite differences for derivative claims;
8. runtime and memory measurements for performance claims;
9. a reusable script, regression test, machine-readable artifact, and
   publication-ready figure.

Stress diagnostics that miss one of these requirements remain useful, but their
scope must be explicit in the validation page and benchmark matrix.

## Numerical Priorities

Near-term numerical work should preserve the current physical operator while
improving evidence and ownership:

- converge variable-coefficient Fourier collocation with measured angular
  oversampling and successive refinement;
- keep true full-system and reduced Schur residuals distinct;
- reuse prepared geometry, lowered programs, and compiled closures across scans;
- avoid vectorization that increases memory without measured throughput benefit;
- specialize adjoints only when primal and tangent regression gates remain
  unchanged;
- split large modules along stable public ownership boundaries, not by moving
  complexity into untested helpers.

The [numerics](numerics.md), [convergence](convergence.md),
[autodiff](autodiff.md), and [performance](performance.md) pages define the
current implementation contracts.

## Physics Basis

The forward formulation follows the Legendre-space monoenergetic treatment in
Javier Escoto's thesis, [arXiv:2510.27513](https://arxiv.org/abs/2510.27513).
Research directions are also informed by:

- neoclassical adjoint optimization,
  [arXiv:1904.06430](https://arxiv.org/abs/1904.06430);
- differentiable plasma workflows,
  [arXiv:2410.11161](https://arxiv.org/abs/2410.11161);
- direct neoclassical ion-transport optimization,
  [arXiv:2406.04147](https://arxiv.org/abs/2406.04147);
- near-axis quasi-isodynamic verification,
  [JPP 2025](https://doi.org/10.1017/S0022377825000157);
- zero-bootstrap-current piecewise omnigenity,
  [arXiv:2505.02546](https://arxiv.org/abs/2505.02546);
- hidden-symmetry optimization,
  [arXiv:2502.09350](https://arxiv.org/abs/2502.09350).

The expanded bibliography and the exact role of each source are maintained in
[Literature](literature.md).

## Source Ownership

| Concern | Primary package area |
| --- | --- |
| Operator and source assembly | `ntx.operators` |
| Preparation, solve, and scans | `ntx.solver` |
| Geometry evaluation and imports | `ntx.geometry` |
| Stable input/output contracts | `ntx.io` and `ntx.inputfiles` |
| Profile and current workflows | `ntx.profiles` |
| NEOPAX bridge | `ntx.neopax` |
| Physics gates and artifacts | `ntx.validation` |

See the [source-code map](source-map.md) before changing ownership. Public APIs
should remain stable while internal modules are reduced to coherent, testable
areas.

## Next Milestone

The next milestone is not a single larger claim. It is a sequence of bounded
pull requests from the private plan: finish user-facing documentation hygiene, continue
stable source ownership, strengthen reusable geometry and profile derivative
families, and promote only the independent geometry/current comparisons that
pass their full convergence gates.
