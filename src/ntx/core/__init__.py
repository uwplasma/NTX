"""Core monoenergetic solver namespace.

This namespace is a stable import surface for solver, scan, and transport
helpers while the historical flat modules remain supported.
"""

from ..solver import (
    CompiledPreparedSolver,
    MonoenergeticCase,
    PreparedMonoenergeticSystem,
    TransportResult,
    compile_prepared_solver,
    healthy_parallel_device_count,
    local_parallel_device_count,
    prepare_monoenergetic_system,
    solve_monoenergetic,
    solve_monoenergetic_internal,
    solve_monoenergetic_parallel_scan,
    solve_monoenergetic_scan,
    solve_prepared,
    solve_prepared_coefficient_vector,
    solve_prepared_coefficient_vector_vjp,
    solve_prepared_internal,
)
from ..transport import coefficients_from_modes, onsager_error

__all__ = [
    "CompiledPreparedSolver",
    "MonoenergeticCase",
    "PreparedMonoenergeticSystem",
    "TransportResult",
    "coefficients_from_modes",
    "compile_prepared_solver",
    "healthy_parallel_device_count",
    "local_parallel_device_count",
    "onsager_error",
    "prepare_monoenergetic_system",
    "solve_monoenergetic",
    "solve_monoenergetic_internal",
    "solve_monoenergetic_parallel_scan",
    "solve_monoenergetic_scan",
    "solve_prepared",
    "solve_prepared_coefficient_vector",
    "solve_prepared_coefficient_vector_vjp",
    "solve_prepared_internal",
]
