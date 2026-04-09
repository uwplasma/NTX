"""JAX-native monoenergetic neoclassical transport solver."""

from .config import enable_x64
from .database import (
    MonoenergeticDatabaseArrays,
    build_monoenergetic_database_arrays,
    stack_monoenergetic_database_arrays,
)
from .geometry import BoozerSurface, VmecSurface, example_surface
from .grids import GridSpec
from .inputfiles import load_run_config, run_from_input_file
from .io import (
    load_boozer_modes_csv,
    load_dkes_surface,
    load_magnetic_configuration_surface,
    load_vmec_surface,
)
from .sfincs_geometry import compare_vmec_geometry_to_sfincs
from .solver import (
    CompiledPreparedSolver,
    MonoenergeticCase,
    PreparedMonoenergeticSystem,
    TransportResult,
    compile_prepared_solver,
    prepare_monoenergetic_system,
    solve_monoenergetic,
    solve_monoenergetic_internal,
    solve_monoenergetic_scan,
    solve_prepared,
    solve_prepared_internal,
)
from .transport import onsager_error

__all__ = [
    "BoozerSurface",
    "CompiledPreparedSolver",
    "GridSpec",
    "MonoenergeticCase",
    "MonoenergeticDatabaseArrays",
    "PreparedMonoenergeticSystem",
    "TransportResult",
    "VmecSurface",
    "build_monoenergetic_database_arrays",
    "compile_prepared_solver",
    "compare_vmec_geometry_to_sfincs",
    "enable_x64",
    "example_surface",
    "load_run_config",
    "load_boozer_modes_csv",
    "load_dkes_surface",
    "load_magnetic_configuration_surface",
    "load_vmec_surface",
    "onsager_error",
    "prepare_monoenergetic_system",
    "run_from_input_file",
    "solve_monoenergetic",
    "solve_monoenergetic_internal",
    "solve_monoenergetic_scan",
    "solve_prepared",
    "solve_prepared_internal",
    "stack_monoenergetic_database_arrays",
]
