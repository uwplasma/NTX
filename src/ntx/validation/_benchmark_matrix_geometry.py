from __future__ import annotations

from ._benchmark_matrix_types import BenchmarkEntry


def geometry_breadth_benchmark_entries() -> tuple[BenchmarkEntry, ...]:
    return (
        BenchmarkEntry(
            id="geometry_family_breadth_summary",
            lane="geometry-breadth",
            maturity="stress-gate",
            title="Artifact-backed geometry-family derivative breadth summary",
            claim_scope=(
                "Summarizes committed analytic, file-backed, boundary-projected, "
                "explicit-relaxed, and implicit-equilibrium diagnostic artifacts "
                "without promoting a full hidden-symmetry, omnigenous, or broad "
                "W7-X/QI validation claim."
            ),
            literature_anchors=(
                "Paul et al. 2019 adjoint neoclassical optimization",
                "McGreivy 2024 differentiable programming for plasma workflows",
                "Landreman and Paul 2022 precise-QS benchmark family",
                "omnigenous and quasi-isodynamic geometry-breadth literature",
            ),
            scripts=("examples/geometry_family_breadth_summary.py",),
            tests=("tests/test_geometry_family_breadth_summary.py",),
            artifacts=(
                "docs/_static/geometry_family_breadth_summary.png",
                "docs/_static/geometry_family_breadth_summary.pdf",
                "docs/_static/geometry_family_breadth_summary.json",
            ),
            manuscript_figures=("geometry_family_breadth_summary",),
            docs=(
                "docs/benchmark-matrix.md",
                "docs/autodiff.md",
                "docs/manuscript.md",
                "docs/research-roadmap.md",
            ),
            open_work=(
                "broaden committed cases to reusable W7-X EIM/KJM, QI, and omnigenous inputs",
                (
                    "add D11/D31/D33 parity and convergence ladders before full "
                    "geometry-family promotion"
                ),
                (
                    "restore implicit-equilibrium derivatives only after residual "
                    "contraction and surface/transport tangent parity pass"
                ),
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
    )


__all__ = ["geometry_breadth_benchmark_entries"]
