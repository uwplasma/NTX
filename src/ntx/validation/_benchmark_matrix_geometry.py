from __future__ import annotations

from ._benchmark_matrix_types import BenchmarkEntry


def geometry_breadth_benchmark_entries() -> tuple[BenchmarkEntry, ...]:
    return (
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
