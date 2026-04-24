from __future__ import annotations

from ._benchmark_matrix_types import BenchmarkEntry


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


__all__ = [
    "monoenergetic_active_benchmark_entries",
    "monoenergetic_planned_benchmark_entries",
]
