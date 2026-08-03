"""Benchmark-matrix entries for derivative-accuracy claims."""

from __future__ import annotations

from ._benchmark_matrix_types import BenchmarkEntry


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


__all__ = ["autodiff_derivative_benchmark_entries"]
