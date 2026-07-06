"""Dense JAX block-tridiagonal monoenergetic DKE solver."""

from ._solver_context import _operator_context
from ._solver_core import (
    prepare_monoenergetic_system,
    solve_monoenergetic,
    solve_monoenergetic_internal,
)
from ._solver_prepared import (
    compile_prepared_solver,
    solve_prepared,
    solve_prepared_coefficient_vector,
    solve_prepared_coefficient_vector_derivative_vjp,
    solve_prepared_coefficient_vector_iterative_jvp,
    solve_prepared_coefficient_vector_iterative_vjp,
    solve_prepared_coefficient_vector_jvp,
    solve_prepared_coefficient_vector_lowdot_two_pullbacks,
    solve_prepared_coefficient_vector_recompute_vjp,
    solve_prepared_coefficient_vector_vjp,
    solve_prepared_internal,
)
from ._solver_scan import (
    _resolved_scan_inputs,
    healthy_parallel_device_count,
    healthy_parallel_devices,
    local_parallel_device_count,
    solve_monoenergetic_parallel_scan,
    solve_monoenergetic_scan,
    solve_scan,
)
from ._solver_types import (
    CompiledPreparedSolver,
    MonoenergeticCase,
    PreparedMonoenergeticSystem,
    TransportResult,
)

__all__ = [
    "CompiledPreparedSolver",
    "MonoenergeticCase",
    "PreparedMonoenergeticSystem",
    "TransportResult",
    "compile_prepared_solver",
    "healthy_parallel_device_count",
    "healthy_parallel_devices",
    "local_parallel_device_count",
    "prepare_monoenergetic_system",
    "solve_monoenergetic",
    "solve_monoenergetic_internal",
    "solve_monoenergetic_parallel_scan",
    "solve_monoenergetic_scan",
    "solve_prepared",
    "solve_prepared_coefficient_vector",
    "solve_prepared_coefficient_vector_derivative_vjp",
    "solve_prepared_coefficient_vector_iterative_jvp",
    "solve_prepared_coefficient_vector_iterative_vjp",
    "solve_prepared_coefficient_vector_jvp",
    "solve_prepared_coefficient_vector_lowdot_two_pullbacks",
    "solve_prepared_coefficient_vector_recompute_vjp",
    "solve_prepared_coefficient_vector_vjp",
    "solve_prepared_internal",
    "solve_scan",
    "_operator_context",
    "_resolved_scan_inputs",
]
