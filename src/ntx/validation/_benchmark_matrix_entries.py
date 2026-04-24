from __future__ import annotations

from ._benchmark_matrix_autodiff import (
    autodiff_active_benchmark_entries,
    autodiff_planned_benchmark_entries,
)
from ._benchmark_matrix_bootstrap import bootstrap_current_benchmark_entries
from ._benchmark_matrix_geometry import geometry_breadth_benchmark_entries
from ._benchmark_matrix_integrated import integrated_workflow_benchmark_entries
from ._benchmark_matrix_monoenergetic import (
    monoenergetic_active_benchmark_entries,
    monoenergetic_planned_benchmark_entries,
)
from ._benchmark_matrix_performance import performance_benchmark_entries
from ._benchmark_matrix_profiles import profile_workflow_benchmark_entries
from ._benchmark_matrix_types import BenchmarkEntry


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


__all__ = ["benchmark_matrix"]
