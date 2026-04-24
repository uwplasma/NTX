from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

BenchmarkMaturity = Literal[
    "positive-gate",
    "stress-gate",
    "software-gate",
    "planned-lane",
]
BenchmarkLane = Literal[
    "monoenergetic",
    "bootstrap-current",
    "integrated-workflow",
    "autodiff",
    "profile-workflow",
    "performance",
    "geometry-breadth",
]


@dataclass(frozen=True)
class BenchmarkEntry:
    """A maintained map from a research claim to code, tests, and artifacts."""

    id: str
    lane: BenchmarkLane
    maturity: BenchmarkMaturity
    title: str
    claim_scope: str
    literature_anchors: tuple[str, ...]
    scripts: tuple[str, ...]
    tests: tuple[str, ...]
    artifacts: tuple[str, ...]
    manuscript_figures: tuple[str, ...]
    docs: tuple[str, ...]
    open_work: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class BenchmarkPathStatus:
    kind: Literal["script", "test", "artifact", "doc"]
    path: str
    exists: bool

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class BenchmarkEvaluation:
    entry: BenchmarkEntry
    path_status: tuple[BenchmarkPathStatus, ...]

    @property
    def missing_required_paths(self) -> tuple[str, ...]:
        if self.entry.maturity == "planned-lane":
            return ()
        return tuple(status.path for status in self.path_status if not status.exists)

    @property
    def status(self) -> Literal["complete", "incomplete", "planned"]:
        if self.entry.maturity == "planned-lane":
            return "planned"
        return "complete" if not self.missing_required_paths else "incomplete"

    def as_dict(self) -> dict[str, object]:
        return {
            "entry": self.entry.as_dict(),
            "status": self.status,
            "missing_required_paths": list(self.missing_required_paths),
            "path_status": [status.as_dict() for status in self.path_status],
        }


def benchmark_matrix() -> tuple[BenchmarkEntry, ...]:
    """Return the maintained NTX benchmark matrix."""

    return (
        BenchmarkEntry(
            id="monoenergetic_validation_summary",
            lane="monoenergetic",
            maturity="positive-gate",
            title="Monoenergetic coefficient convergence and symmetry",
            claim_scope=(
                "NTX reproduces the expected monoenergetic coefficient behavior, "
                "Onsager residual control, and Legendre convergence on owned "
                "DKES-style and VMEC surfaces."
            ),
            literature_anchors=(
                "Escoto et al. 2024 monoenergetic convergence and benchmarking",
                "Escoto PhD thesis monoenergetic formulation",
                "Helander, Parra, and Newton 2017 low-collisionality scaling",
            ),
            scripts=("examples/validation_summary.py",),
            tests=(
                "tests/test_validation_summary_example.py",
                "tests/test_make_publication_figures.py",
            ),
            artifacts=(
                "docs/_static/validation_summary.png",
                "docs/_static/validation_summary.pdf",
                "docs/_static/validation_summary.json",
            ),
            manuscript_figures=("validation_summary",),
            docs=("docs/validation.md", "docs/manuscript.md"),
        ),
        BenchmarkEntry(
            id="precise_qs_redl_sfincs",
            lane="bootstrap-current",
            maturity="positive-gate",
            title="Precise-QS Redl current against archived fixed-field reference",
            claim_scope=(
                "The Redl path closes the archived precise-QS fixed-field "
                "interior-window gate and is separated from the reduced closure "
                "stress metric."
            ),
            literature_anchors=(
                "Landreman and Paul 2022 precise-QS benchmark family",
                "Redl et al. bootstrap-current fit",
                "SFINCS fixed-field archive used by the benchmark",
            ),
            scripts=(
                "examples/precise_qs_redl_sfincs_audit.py",
                "examples/bootstrap_current_fixed_field_validation.py",
            ),
            tests=(
                "tests/test_precise_qs_redl_sfincs_audit.py",
                "tests/test_fixed_field_parallel_flow_audit.py",
            ),
            artifacts=(
                "docs/_static/bootstrap_current_fixed_field_validation.png",
                "docs/_static/bootstrap_current_fixed_field_validation.pdf",
                "docs/_static/bootstrap_current_fixed_field_validation.json",
            ),
            manuscript_figures=("closure_validation_report",),
            docs=("docs/physics-gates.md", "docs/validation.md"),
        ),
        BenchmarkEntry(
            id="fixed_field_ntx_neopax_closure_stress",
            lane="bootstrap-current",
            maturity="stress-gate",
            title="Fixed-field current closure stress test",
            claim_scope=(
                "The fixed-field NTX+NEOPAX comparison is retained as a monitored "
                "closure stress test, not as a promoted monoenergetic parity gate."
            ),
            literature_anchors=(
                "Landreman and Paul 2022 precise-QS benchmark family",
                "momentum-restoring closure literature for parallel-flow models",
            ),
            scripts=(
                "examples/bootstrap_current_fixed_field_validation.py",
                "examples/fixed_field_momentum_correction_diagnostic.py",
                "examples/momentum_correction_mapping_audit.py",
            ),
            tests=(
                "tests/test_fixed_field_momentum_correction_diagnostic.py",
                "tests/test_momentum_correction_mapping_audit.py",
            ),
            artifacts=(
                "docs/_static/bootstrap_current_fixed_field_validation.png",
                "docs/_static/bootstrap_current_fixed_field_validation.pdf",
                "docs/_static/bootstrap_current_fixed_field_validation.json",
            ),
            manuscript_figures=("closure_validation_report",),
            docs=("docs/physics-gates.md", "docs/validation.md"),
            open_work=(
                "derive and implement a transferable momentum-restoring closure",
                "require no regression on the integrated W7-X workflow",
            ),
        ),
        BenchmarkEntry(
            id="w7x_integrated_transfer",
            lane="integrated-workflow",
            maturity="positive-gate",
            title="Imported W7-X workflow transfer",
            claim_scope=(
                "The rebuilt raw-branch NTX database transfers through the "
                "imported W7-X workflow without exceeding the current release gate."
            ),
            literature_anchors=(
                "W7-X benchmark profile used by the imported workflow",
                "neoclassical bootstrap-current profile validation practice",
            ),
            scripts=("examples/bootstrap_current_reference_audit_w7x.py",),
            tests=(
                "tests/test_bootstrap_current_reference_audit_w7x.py",
                "tests/test_w7x_reference_benchmark.py",
            ),
            artifacts=(
                "docs/_static/bootstrap_current_reference_audit_w7x.png",
                "docs/_static/bootstrap_current_reference_audit_w7x.pdf",
                "docs/_static/bootstrap_current_reference_audit_w7x.json",
            ),
            manuscript_figures=("bootstrap_current_reference_audit_w7x",),
            docs=("docs/physics-gates.md", "docs/validation.md", "docs/manuscript.md"),
        ),
        BenchmarkEntry(
            id="prepared_derivative_path",
            lane="autodiff",
            maturity="positive-gate",
            title="Prepared implicit-adjoint derivative path",
            claim_scope=(
                "Prepared derivatives agree with direct reverse-mode and give a "
                "measured speedup on the committed electric-field scan."
            ),
            literature_anchors=(
                "Paul et al. 2019 adjoint neoclassical optimization",
                "McGreivy 2024 differentiable programming for plasma workflows",
            ),
            scripts=("examples/derivative_path_benchmark.py",),
            tests=("tests/test_derivative_path_benchmark_example.py",),
            artifacts=(
                "docs/_static/derivative_path_benchmark.png",
                "docs/_static/derivative_path_benchmark.pdf",
                "docs/_static/derivative_path_benchmark.json",
            ),
            manuscript_figures=("derivative_path_benchmark",),
            docs=("docs/autodiff.md", "docs/manuscript.md"),
            open_work=(
                "reduce memory and factorization overhead on larger geometry-control scans",
            ),
        ),
        BenchmarkEntry(
            id="geometry_control_derivative_benchmark",
            lane="autodiff",
            maturity="stress-gate",
            title="Three-harmonic geometry-control derivative audit",
            claim_scope=(
                "Direct geometry-control autodiff agrees with centered finite "
                "differences for three controlled Boozer harmonics on an owned "
                "analytic surface."
            ),
            literature_anchors=(
                "Paul et al. 2019 adjoint neoclassical optimization",
                "McGreivy 2024 differentiable programming for plasma workflows",
                "Escoto thesis monoenergetic formulation",
            ),
            scripts=("examples/geometry_control_derivative_benchmark.py",),
            tests=(
                "tests/test_geometry_control_derivative_benchmark_example.py",
                "tests/test_make_publication_figures.py",
            ),
            artifacts=(
                "docs/_static/geometry_control_derivative_benchmark.png",
                "docs/_static/geometry_control_derivative_benchmark.pdf",
                "docs/_static/geometry_control_derivative_benchmark.json",
            ),
            manuscript_figures=("geometry_control_derivative_benchmark",),
            docs=("docs/autodiff.md", "docs/manuscript.md"),
            open_work=(
                "transfer this audit to reusable VMEC/Boozer geometry-control families",
                "compare geometry pullbacks with an implicit-adjoint path once implemented",
            ),
        ),
        BenchmarkEntry(
            id="file_backed_geometry_control_derivative_benchmark",
            lane="autodiff",
            maturity="stress-gate",
            title="File-backed geometry-control derivative audit",
            claim_scope=(
                "Direct geometry-control autodiff agrees with centered finite "
                "differences on repository-owned Boozer and VMEC sample surfaces."
            ),
            literature_anchors=(
                "Paul et al. 2019 adjoint neoclassical optimization",
                "McGreivy 2024 differentiable programming for plasma workflows",
                "Escoto thesis monoenergetic formulation",
            ),
            scripts=("examples/file_backed_geometry_control_derivative_benchmark.py",),
            tests=("tests/test_file_backed_geometry_control_derivative_benchmark_example.py",),
            artifacts=(
                "docs/_static/file_backed_geometry_control_derivative_benchmark.png",
                "docs/_static/file_backed_geometry_control_derivative_benchmark.pdf",
                "docs/_static/file_backed_geometry_control_derivative_benchmark.json",
            ),
            manuscript_figures=("file_backed_geometry_control_derivative_benchmark",),
            docs=("docs/autodiff.md", "docs/manuscript.md"),
            open_work=(
                "extend from repository sample files to broader reusable geometry families",
                "compare geometry pullbacks with an implicit-adjoint path once implemented",
                "measure memory and factorization reuse on larger geometry-control scans",
            ),
        ),
        BenchmarkEntry(
            id="boundary_forward_mode_current_derivative_benchmark",
            lane="autodiff",
            maturity="stress-gate",
            title="Boundary-to-output forward-mode derivative audit",
            claim_scope=(
                "Low-dimensional boundary controls propagate through "
                "boundary-projected vmec_jax geometry, booz_xform_jax, NTX "
                "coefficients, and an NTX+NEOPAX integrated-current objective "
                "under forward-mode autodiff."
            ),
            literature_anchors=(
                "vmec_jax explicit differentiable boundary workflows",
                "booz_xform_jax JAX-native Boozer transform",
                "McGreivy 2024 differentiable programming for plasma workflows",
            ),
            scripts=("examples/boundary_forward_mode_current_derivative_benchmark.py",),
            tests=("tests/test_boundary_forward_mode_current_derivative_benchmark_example.py",),
            artifacts=(
                "docs/_static/boundary_forward_mode_current_derivative_benchmark.png",
                "docs/_static/boundary_forward_mode_current_derivative_benchmark.pdf",
                "docs/_static/boundary_forward_mode_current_derivative_benchmark.json",
            ),
            manuscript_figures=(),
            docs=("docs/autodiff.md", "docs/research-roadmap.md"),
            open_work=(
                (
                    "keep this fast projected-geometry lane as a low-cost precursor "
                    "to the equilibrium-relaxed benchmark"
                ),
                (
                    "validate broader non-axisymmetric benchmark families "
                    "beyond the repository sample input"
                ),
                (
                    "determine whether reverse-mode can be repaired or whether "
                    "the boundary-control lane should stay forward-mode only"
                ),
            ),
        ),
        BenchmarkEntry(
            id="implicit_equilibrium_forward_mode_derivative_benchmark",
            lane="autodiff",
            maturity="stress-gate",
            title="Implicit-equilibrium forward-mode derivative audit",
            claim_scope=(
                "The implicit fixed-boundary vmec_jax residual solve reaches "
                "Boozer geometry and an NTX transport observable under "
                "forward-mode autodiff on the committed QA case, but only the "
                "equilibrium-volume derivative currently matches centered "
                "finite differences on this lane."
            ),
            literature_anchors=(
                "vmec_jax implicit fixed-boundary differentiation",
                "booz_xform_jax JAX-native Boozer transform",
                "McGreivy 2024 differentiable programming for plasma workflows",
            ),
            scripts=("examples/implicit_equilibrium_forward_mode_derivative_benchmark.py",),
            tests=("tests/test_implicit_equilibrium_forward_mode_derivative_benchmark_example.py",),
            artifacts=(
                "docs/_static/implicit_equilibrium_forward_mode_derivative_benchmark.png",
                "docs/_static/implicit_equilibrium_forward_mode_derivative_benchmark.pdf",
                "docs/_static/implicit_equilibrium_forward_mode_derivative_benchmark.json",
            ),
            manuscript_figures=("implicit_equilibrium_forward_mode_derivative_benchmark",),
            docs=("docs/autodiff.md", "docs/research-roadmap.md", "docs/manuscript.md"),
            open_work=(
                "recover Boozer-scalar parity on the implicit geometry path",
                "recover NTX transport parity on the implicit geometry path",
                (
                    "extend the implicit forward-mode lane from NTX transport to "
                    "NTX+NEOPAX integrated current"
                ),
                "broaden beyond the committed QA case to additional geometry families",
                "repair reverse mode through the implicit vmec_jax -> booz_xform_jax path",
            ),
        ),
        BenchmarkEntry(
            id="explicit_relaxed_boundary_current_derivative_benchmark",
            lane="autodiff",
            maturity="stress-gate",
            title="Explicit-relaxed boundary-to-current derivative audit",
            claim_scope=(
                "Low-dimensional boundary controls propagate through an "
                "explicitly relaxed fixed-boundary vmec_jax solve, "
                "booz_xform_jax, NTX coefficients, and an NTX+NEOPAX "
                "integrated-current objective under forward-mode autodiff on "
                "committed QA and QH family cases, while preserving the "
                "ordinary primal volume."
            ),
            literature_anchors=(
                "vmec_jax explicit differentiable boundary workflows",
                "booz_xform_jax JAX-native Boozer transform",
                "McGreivy 2024 differentiable programming for plasma workflows",
                "Landreman and Paul 2022 precise-QS benchmark family",
            ),
            scripts=("examples/explicit_relaxed_boundary_current_derivative_benchmark.py",),
            tests=("tests/test_explicit_relaxed_boundary_current_derivative_benchmark_example.py",),
            artifacts=(
                "docs/_static/explicit_relaxed_boundary_current_derivative_benchmark.png",
                "docs/_static/explicit_relaxed_boundary_current_derivative_benchmark.pdf",
                "docs/_static/explicit_relaxed_boundary_current_derivative_benchmark.json",
            ),
            manuscript_figures=("explicit_relaxed_boundary_current_derivative_benchmark",),
            docs=("docs/autodiff.md", "docs/research-roadmap.md", "docs/manuscript.md"),
            open_work=(
                (
                    "widen beyond the committed QA and QH cases to additional "
                    "geometry families"
                ),
                (
                    "establish whether the implicit-equilibrium path can recover "
                    "the same boundary sensitivities"
                ),
                (
                    "determine whether reverse-mode can be repaired on the "
                    "equilibrium-relaxed boundary-control lane"
                ),
            ),
        ),
        BenchmarkEntry(
            id="autodiff_inverse_problem",
            lane="autodiff",
            maturity="stress-gate",
            title="Synthetic inverse-design recovery",
            claim_scope=(
                "A differentiable profile/geometry control can recover a generated "
                "target on a small owned inverse problem."
            ),
            literature_anchors=(
                "adjoint stellarator-optimization validation practice",
                "differentiable programming verification by generated targets",
            ),
            scripts=("examples/autodiff_inverse_problem.py",),
            tests=("tests/test_autodiff.py", "tests/test_make_publication_figures.py"),
            artifacts=(
                "docs/_static/autodiff_inverse_problem.png",
                "docs/_static/autodiff_inverse_problem.pdf",
            ),
            manuscript_figures=("autodiff_inverse_problem",),
            docs=("docs/autodiff.md",),
            open_work=("extend from scalar controls to larger geometry-control families",),
        ),
        BenchmarkEntry(
            id="autodiff_profile_uncertainty",
            lane="autodiff",
            maturity="stress-gate",
            title="Linearized profile uncertainty against Monte Carlo",
            claim_scope=(
                "Linearized covariance propagation is checked against a small "
                "Monte Carlo ensemble on the same differentiable profile map."
            ),
            literature_anchors=(
                "differentiable programming uncertainty propagation",
                "profile-level neoclassical workflow sensitivity analysis",
            ),
            scripts=("examples/autodiff_profile_uncertainty.py",),
            tests=(
                "tests/test_autodiff.py",
                "tests/test_autodiff_profile_uncertainty_example.py",
            ),
            artifacts=(
                "docs/_static/autodiff_profile_uncertainty.png",
                "docs/_static/autodiff_profile_uncertainty.pdf",
                "docs/_static/autodiff_profile_uncertainty.json",
            ),
            manuscript_figures=("autodiff_profile_uncertainty",),
            docs=("docs/autodiff.md",),
            open_work=("add Fisher or Hessian-vector probes on a larger profile basis",),
        ),
        BenchmarkEntry(
            id="robust_bootstrap_current_optimization",
            lane="autodiff",
            maturity="stress-gate",
            title="Robust bootstrap-current optimization under control uncertainty",
            claim_scope=(
                "The robust-design objective improves under a prescribed Gaussian "
                "control uncertainty on a deterministic owned benchmark."
            ),
            literature_anchors=(
                "adjoint stellarator optimization",
                "direct neoclassical ion-transport optimization",
                "robust optimization under parameter uncertainty",
            ),
            scripts=("examples/bootstrap_current_robust_optimization.py",),
            tests=(
                "tests/test_autodiff.py",
                "tests/test_bootstrap_current_robust_optimization_example.py",
            ),
            artifacts=(
                "docs/_static/bootstrap_current_robust_optimization.png",
                "docs/_static/bootstrap_current_robust_optimization.pdf",
                "docs/_static/bootstrap_current_robust_optimization.json",
            ),
            manuscript_figures=("bootstrap_current_robust_optimization",),
            docs=("docs/autodiff.md",),
            open_work=(
                "promote only after testing on broader geometry controls and profile families",
            ),
        ),
        BenchmarkEntry(
            id="profile_force_reconstruction",
            lane="profile-workflow",
            maturity="stress-gate",
            title="Primitive profile to force reconstruction",
            claim_scope=(
                "The primitive-profile builder is audited on the precise-QS "
                "profile family and kept as a monitored profile stress benchmark."
            ),
            literature_anchors=(
                "Landreman and Paul 2022 precise-QS density and temperature profiles",
                "thermodynamic-force formulation of neoclassical transport",
            ),
            scripts=("examples/profile_force_reconstruction_audit.py",),
            tests=("tests/test_profile_force_reconstruction_audit_example.py",),
            artifacts=(
                "docs/_static/profile_force_reconstruction_audit.png",
                "docs/_static/profile_force_reconstruction_audit.pdf",
                "docs/_static/profile_force_reconstruction_audit.json",
            ),
            manuscript_figures=("profile_force_reconstruction_audit",),
            docs=("docs/profiles.md",),
            open_work=("tighten profile-level physical diagnostics for long-radius studies",),
        ),
        BenchmarkEntry(
            id="performance_scaling",
            lane="performance",
            maturity="software-gate",
            title="CPU/GPU throughput characterization",
            claim_scope=(
                "Serial, multiprocess, CPU, and GPU execution modes are "
                "characterized on committed smoke and heavier-grid cases."
            ),
            literature_anchors=(
                "JAX performance practice for compiled scientific workloads",
                "multi-process scan execution for independent monoenergetic cases",
            ),
            scripts=("examples/performance_scaling.py", "scripts/benchmark_scaling.py"),
            tests=(
                "tests/test_performance_scaling_example.py",
                "tests/test_benchmark_scaling_script.py",
            ),
            artifacts=(
                "docs/_static/performance_scaling_smoke.png",
                "docs/_static/performance_scaling_smoke.pdf",
                "docs/_static/performance_scaling_cpu_smoke.json",
                "docs/_static/performance_scaling_gpu_smoke.json",
                "docs/_static/performance_scaling_heavy.png",
                "docs/_static/performance_scaling_heavy.pdf",
                "docs/_static/performance_scaling_cpu_heavy.json",
                "docs/_static/performance_scaling_gpu_heavy.json",
            ),
            manuscript_figures=("performance_scaling_heavy",),
            docs=("docs/performance.md", "docs/gpu.md"),
            open_work=(
                "map production-grid crossover points more systematically",
                "improve prepared-geometry reuse across larger scan campaigns",
            ),
        ),
        BenchmarkEntry(
            id="geometry_breadth_hidden_symmetry",
            lane="geometry-breadth",
            maturity="planned-lane",
            title="Hidden-symmetry and omnigenous geometry families",
            claim_scope=(
                "Future research workflows should broaden validation beyond the "
                "current W7-X-centered set."
            ),
            literature_anchors=(
                "near-axis quasi-isodynamic construction and verification",
                "hidden-symmetry optimization literature",
                "piecewise-omnigenous optimization literature",
            ),
            scripts=(),
            tests=(),
            artifacts=(),
            manuscript_figures=(),
            docs=("docs/research-roadmap.md",),
            open_work=(
                "identify reusable public inputs for hidden-symmetry studies",
                "add VMEC/Boozer family examples once inputs are owned",
                "define convergence gates before promoting figures",
            ),
        ),
        BenchmarkEntry(
            id="full_monoenergetic_geometry_family",
            lane="monoenergetic",
            maturity="planned-lane",
            title="Full literature monoenergetic geometry-family reproduction",
            claim_scope=(
                "The compact validation summary should be broadened into the "
                "full literature geometry-family reproduction before claiming "
                "coverage of all benchmark cases."
            ),
            literature_anchors=(
                "Escoto et al. 2024 W7-X EIM, W7-X KJM, and CIEMAT-QI benchmarks",
                "Escoto thesis convergence ladders for D11, D31, and D33",
            ),
            scripts=(),
            tests=(),
            artifacts=(),
            manuscript_figures=(),
            docs=("docs/benchmark-matrix.md", "docs/validation.md"),
            open_work=(
                "own or regenerate reusable inputs for W7-X EIM, W7-X KJM, and CIEMAT-QI",
                "add D11, D31, and D33 parity plots for those families",
                "add N_xi, N_theta, and N_zeta convergence ladders",
                "include zero and finite radial-electric-field cases where applicable",
            ),
        ),
        BenchmarkEntry(
            id="large_geometry_control_autodiff",
            lane="autodiff",
            maturity="planned-lane",
            title="Large geometry-control autodiff validation",
            claim_scope=(
                "The scalar derivative and optimization examples should be "
                "extended to larger geometry-control families before promoting "
                "optimization-grade stellarator-design claims."
            ),
            literature_anchors=(
                "Paul et al. 2019 adjoint neoclassical optimization",
                "McGreivy 2024 differentiable programming for plasma workflows",
                "direct neoclassical transport optimization literature",
            ),
            scripts=(),
            tests=(),
            artifacts=(),
            manuscript_figures=(),
            docs=("docs/autodiff.md", "docs/research-roadmap.md"),
            open_work=(
                (
                    "broaden the current analytic and file-backed audits into "
                    "reusable geometry families"
                ),
                "compare direct autodiff, implicit-adjoint, and finite-difference probes",
                "measure memory and factorization reuse under real scan workloads",
                "add publication-ready derivative and optimization figures",
            ),
        ),
    )
