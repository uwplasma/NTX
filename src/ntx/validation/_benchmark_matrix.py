"""Benchmark-matrix entries: monoenergetic, profiles, bootstrap, autodiff, performance.

Every entry in the published benchmark matrix except the geometry rows, which
are bulkier and live in _benchmark_matrix_geometry. Shared row types live in
_benchmark_matrix_types, which both import.
"""

from __future__ import annotations

from ._benchmark_matrix_types import BenchmarkEntry
from ._benchmark_matrix_geometry import geometry_breadth_benchmark_entries

__all__ = [
    "autodiff_active_benchmark_entries",
    "autodiff_derivative_benchmark_entries",
    "autodiff_design_benchmark_entries",
    "autodiff_design_planned_benchmark_entries",
    "autodiff_planned_benchmark_entries",
    "benchmark_matrix",
    "bootstrap_current_benchmark_entries",
    "integrated_workflow_benchmark_entries",
    "monoenergetic_active_benchmark_entries",
    "monoenergetic_planned_benchmark_entries",
    "performance_benchmark_entries",
    "profile_workflow_benchmark_entries",
]


# --- _benchmark_matrix_autodiff_derivatives: Benchmark-matrix entries for derivative-accuracy claims. ---

def autodiff_derivative_benchmark_entries() -> tuple[BenchmarkEntry, ...]:
    return (
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
                "boundary-projected vmex geometry, booz_xform_jax, NTX "
                "coefficients, and an NTX+NEOPAX integrated-current objective "
                "under forward-mode autodiff."
            ),
            literature_anchors=(
                "vmex explicit differentiable boundary workflows",
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
            manuscript_figures=("boundary_forward_mode_current_derivative_benchmark",),
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
            title="Implicit-equilibrium non-shipping derivative diagnostic",
            claim_scope=(
                "The implicit fixed-boundary residual-forward path is retained "
                "as an artifact-backed diagnostic, but it is closed as "
                "non-shipping because residual contraction and Boozer/NTX "
                "surface-transport tangent parity do not pass on the committed "
                "QA case. The explicit-relaxed lane is the supported "
                "differentiable equilibrium path."
            ),
            literature_anchors=(
                "vmex implicit fixed-boundary differentiation",
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
                (
                    "restore an implicit-equilibrium derivative lane only after "
                    "residual contraction and Boozer/NTX tangent parity pass"
                ),
                "broaden beyond the committed QA case to additional geometry families",
                "keep explicit-relaxed derivatives as the shipping equilibrium route",
            ),
        ),
        BenchmarkEntry(
            id="explicit_relaxed_boundary_current_derivative_benchmark",
            lane="autodiff",
            maturity="stress-gate",
            title="Explicit-relaxed boundary-to-current derivative audit",
            claim_scope=(
                "Low-dimensional boundary controls propagate through an "
                "explicitly relaxed fixed-boundary vmex solve, "
                "booz_xform_jax, NTX coefficients, and an NTX+NEOPAX "
                "integrated-current objective under forward-mode autodiff on "
                "committed QA and QH family cases, while preserving the "
                "ordinary primal volume."
            ),
            literature_anchors=(
                "vmex explicit differentiable boundary workflows",
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
    )


# --- _benchmark_matrix_autodiff_design: Benchmark-matrix entries for design-sensitivity and optimization claims. ---

def autodiff_design_benchmark_entries() -> tuple[BenchmarkEntry, ...]:
    return (
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
            title="Radial-basis profile uncertainty and Fisher/HVP audit",
            claim_scope=(
                "Linearized covariance propagation, Monte Carlo sampling, and "
                "a Fisher/Gauss-Newton versus Hessian-vector-product probe are "
                "checked on the same differentiable radial-basis profile map."
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
            open_work=(
                "broaden from the committed radial-basis audit to cross-geometry profile families",
            ),
        ),
        BenchmarkEntry(
            id="bootstrap_current_optimization",
            lane="autodiff",
            maturity="stress-gate",
            title="Differentiable bootstrap-current optimization",
            claim_scope=(
                "The differentiable bootstrap-current objective improves a "
                "weighted reduced current response on the committed W7-X "
                "study while remaining scoped below a broad stellarator-design "
                "claim."
            ),
            literature_anchors=(
                "Paul et al. 2019 adjoint neoclassical optimization",
                "direct neoclassical ion-transport optimization literature",
                "differentiable programming verification by generated objectives",
            ),
            scripts=("examples/bootstrap_current_optimization.py",),
            tests=(
                "tests/test_autodiff.py",
                "tests/test_bootstrap_current_optimization_example.py",
            ),
            artifacts=(
                "docs/_static/bootstrap_current_optimization.png",
                "docs/_static/bootstrap_current_optimization.pdf",
                "docs/_static/bootstrap_current_optimization.json",
            ),
            manuscript_figures=("bootstrap_current_optimization",),
            docs=("docs/autodiff.md", "docs/examples.md", "docs/manuscript.md"),
            open_work=(
                "promote only after broader geometry-family controls are added",
                "tie future optimization claims to reusable derivative-audit gates",
            ),
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
    )


def autodiff_design_planned_benchmark_entries() -> tuple[BenchmarkEntry, ...]:
    return (
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


# --- _benchmark_matrix_autodiff: Benchmark-matrix entries for the differentiation lane. ---

def autodiff_active_benchmark_entries() -> tuple[BenchmarkEntry, ...]:
    return (
        *autodiff_derivative_benchmark_entries(),
        *autodiff_design_benchmark_entries(),
    )


def autodiff_planned_benchmark_entries() -> tuple[BenchmarkEntry, ...]:
    return autodiff_design_planned_benchmark_entries()


# --- _benchmark_matrix_bootstrap: Benchmark-matrix entries for the bootstrap-current lane. ---

def bootstrap_current_benchmark_entries() -> tuple[BenchmarkEntry, ...]:
    return (
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
                "scripts/build_closure_validation_report.py",
            ),
            tests=(
                "tests/test_precise_qs_redl_sfincs_audit.py",
                "tests/test_fixed_field_parallel_flow_audit.py",
            ),
            artifacts=(
                "docs/_static/bootstrap_current_fixed_field_validation.png",
                "docs/_static/bootstrap_current_fixed_field_validation.pdf",
                "docs/_static/bootstrap_current_fixed_field_validation.json",
                "docs/_static/closure_validation_report.png",
                "docs/_static/closure_validation_report.pdf",
                "docs/_static/closure_validation_report.json",
                "docs/_static/closure_validation_report.txt",
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
                "The fixed-field NTX+NEOPAX total-current comparison passes the "
                "documented reduced-closure stress gate with the low-order "
                "Spitzer-conductivity branch; species-current parity and broader "
                "closure defaults remain separate claims."
            ),
            literature_anchors=(
                "Landreman and Paul 2022 precise-QS benchmark family",
                "momentum-restoring closure literature for parallel-flow models",
            ),
            scripts=(
                "examples/bootstrap_current_fixed_field_validation.py",
                "examples/fixed_field_momentum_correction_diagnostic.py",
                "examples/momentum_correction_mapping_audit.py",
                "scripts/build_closure_validation_report.py",
            ),
            tests=(
                "tests/test_fixed_field_momentum_correction_diagnostic.py",
                "tests/test_momentum_correction_mapping_audit.py",
            ),
            artifacts=(
                "docs/_static/bootstrap_current_fixed_field_validation.png",
                "docs/_static/bootstrap_current_fixed_field_validation.pdf",
                "docs/_static/bootstrap_current_fixed_field_validation.json",
                "docs/_static/closure_validation_report.png",
                "docs/_static/closure_validation_report.pdf",
                "docs/_static/closure_validation_report.json",
                "docs/_static/closure_validation_report.txt",
                "docs/_static/closure_pmax_convergence.png",
                "docs/_static/closure_pmax_convergence.pdf",
                "docs/_static/closure_pmax_convergence.json",
            ),
            manuscript_figures=("closure_validation_report",),
            docs=("docs/physics-gates.md", "docs/validation.md"),
            open_work=(
                "derive any broader default from the same moment-equation closure",
                "keep the integrated W7-X raw-branch transfer regression separate",
            ),
        ),
    )


# --- _benchmark_matrix_integrated: Benchmark-matrix entries for end-to-end integrated workflows. ---

def integrated_workflow_benchmark_entries() -> tuple[BenchmarkEntry, ...]:
    return (
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
    )


# --- _benchmark_matrix_monoenergetic: Benchmark-matrix entries for the monoenergetic transport lane. ---

def monoenergetic_active_benchmark_entries() -> tuple[BenchmarkEntry, ...]:
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
    )


def monoenergetic_planned_benchmark_entries() -> tuple[BenchmarkEntry, ...]:
    return (
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
    )


# --- _benchmark_matrix_performance: Benchmark-matrix entries for throughput and scaling claims. ---

def performance_benchmark_entries() -> tuple[BenchmarkEntry, ...]:
    return (
        BenchmarkEntry(
            id="performance_scaling",
            lane="performance",
            maturity="software-gate",
            title="CPU/GPU throughput characterization",
            claim_scope=(
                "Serial, multiprocess, CPU, and GPU execution modes are "
                "characterized on committed smoke, heavier-grid, and "
                "production-grid cases, with a separate fixed-workload "
                "strong-scaling artifact."
            ),
            literature_anchors=(
                "JAX performance practice for compiled scientific workloads",
                "multi-process scan execution for independent monoenergetic cases",
            ),
            scripts=(
                "examples/performance_scaling.py",
                "examples/performance_strong_scaling.py",
                "scripts/benchmark_scaling.py",
                "scripts/benchmark_strong_scaling.py",
            ),
            tests=(
                "tests/test_performance_scaling_example.py",
                "tests/test_benchmark_scaling_script.py",
            ),
            artifacts=(
                "docs/_static/performance_scaling_smoke.png",
                "docs/_static/performance_scaling_smoke.pdf",
                "docs/_static/performance_scaling_smoke.json",
                "docs/_static/performance_scaling_cpu_smoke.json",
                "docs/_static/performance_scaling_gpu_smoke.json",
                "docs/_static/performance_scaling_heavy.png",
                "docs/_static/performance_scaling_heavy.pdf",
                "docs/_static/performance_scaling_heavy.json",
                "docs/_static/performance_scaling_cpu_heavy.json",
                "docs/_static/performance_scaling_gpu_heavy.json",
                "docs/_static/performance_scaling_production.png",
                "docs/_static/performance_scaling_production.pdf",
                "docs/_static/performance_scaling_production.json",
                "docs/_static/performance_scaling_cpu_production.json",
                "docs/_static/performance_scaling_gpu_production.json",
                "docs/_static/performance_strong_scaling_production.png",
                "docs/_static/performance_strong_scaling_production.pdf",
                "docs/_static/performance_strong_scaling_production.json",
                "docs/_static/performance_strong_scaling_cpu_production.json",
                "docs/_static/performance_strong_scaling_gpu_production.json",
            ),
            manuscript_figures=(
                "performance_scaling_production",
                "performance_strong_scaling_production",
            ),
            docs=("docs/performance.md", "docs/gpu.md"),
            open_work=(
                "repeat the production and strong-scaling matrices on additional GPU nodes",
                "add device-memory timelines for larger VMEC-family workloads",
            ),
        ),
        BenchmarkEntry(
            id="prepared_geometry_reuse_profile",
            lane="performance",
            maturity="software-gate",
            title="Prepared-geometry and compiled-solver reuse profile",
            claim_scope=(
                "Measures repeated monoenergetic solves with direct calls, "
                "prepared geometry reuse, and a compiled prepared solver on "
                "one fixed geometry. This is a performance and reproducibility "
                "gate, not a physics-validation claim."
            ),
            literature_anchors=(
                "JAX performance practice for compiled scientific workloads",
                "reuse of fixed-geometry DKE operators across monoenergetic scans",
            ),
            scripts=("examples/prepared_geometry_reuse_profile.py",),
            tests=("tests/test_prepared_geometry_reuse_profile_example.py",),
            artifacts=(
                "docs/_static/prepared_geometry_reuse_profile.png",
                "docs/_static/prepared_geometry_reuse_profile.pdf",
                "docs/_static/prepared_geometry_reuse_profile.json",
            ),
            manuscript_figures=("prepared_geometry_reuse_profile",),
            docs=("docs/performance.md", "docs/numerics.md", "docs/manuscript.md"),
            open_work=(
                "repeat the profile on larger production VMEC-family scans",
                "evaluate reusable factorization or linear-operator approaches after profiling",
            ),
        ),
    )


# --- _benchmark_matrix_profiles: Benchmark-matrix entries for profile and closure workflows. ---

def profile_workflow_benchmark_entries() -> tuple[BenchmarkEntry, ...]:
    return (
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
            id="profile_basis_optimization",
            lane="profile-workflow",
            maturity="stress-gate",
            title="Radial-basis profile-control optimization",
            claim_scope=(
                "A three-function radial basis control is optimized through the "
                "profile closure, with objective, residual, and control metrics "
                "stored as a monitored workflow artifact."
            ),
            literature_anchors=(
                "profile-level neoclassical transport sensitivity workflows",
                "differentiable programming verification by controlled objectives",
            ),
            scripts=("examples/profile_basis_optimization.py",),
            tests=("tests/test_profile_basis_optimization_example.py",),
            artifacts=(
                "docs/_static/profile_basis_optimization.png",
                "docs/_static/profile_basis_optimization.pdf",
                "docs/_static/profile_basis_optimization.json",
            ),
            manuscript_figures=("profile_basis_optimization",),
            docs=("docs/profiles.md", "docs/manuscript.md"),
            open_work=(
                "broaden the radial basis to reusable physics-motivated profile families",
                "promote only after cross-geometry profile/current validation",
            ),
        ),
    )


# --- _benchmark_matrix_entries: Assembles the benchmark matrix from its per-lane entry modules. ---

def benchmark_matrix() -> tuple[BenchmarkEntry, ...]:
    """Return the maintained NTX benchmark matrix."""

    return (
        *monoenergetic_active_benchmark_entries(),
        *bootstrap_current_benchmark_entries(),
        *integrated_workflow_benchmark_entries(),
        *autodiff_active_benchmark_entries(),
        *profile_workflow_benchmark_entries(),
        *performance_benchmark_entries(),
        *geometry_breadth_benchmark_entries(),
        *monoenergetic_planned_benchmark_entries(),
        *autodiff_planned_benchmark_entries(),
    )
