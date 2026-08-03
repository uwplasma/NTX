"""Benchmark-matrix entries for the differentiation lane."""

from __future__ import annotations

from ._benchmark_matrix_autodiff_derivatives import (
    autodiff_derivative_benchmark_entries,
)
from ._benchmark_matrix_autodiff_design import (
    autodiff_design_benchmark_entries,
    autodiff_design_planned_benchmark_entries,
)
from ._benchmark_matrix_types import BenchmarkEntry


def autodiff_active_benchmark_entries() -> tuple[BenchmarkEntry, ...]:
    return (
        *autodiff_derivative_benchmark_entries(),
        *autodiff_design_benchmark_entries(),
    )


def autodiff_planned_benchmark_entries() -> tuple[BenchmarkEntry, ...]:
    return autodiff_design_planned_benchmark_entries()


__all__ = [
    "autodiff_active_benchmark_entries",
    "autodiff_planned_benchmark_entries",
]
