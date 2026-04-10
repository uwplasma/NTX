"""JAX-native monoenergetic neoclassical transport solver."""

from importlib.metadata import PackageNotFoundError, version

from .booz import BoozmnSurface, load_boozmn_surface
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
from .neopax import (
    NeopaxMonoenergeticArrays,
    NeopaxScan,
    build_ntx_neopax_scan,
    build_ntx_neopax_scan_from_surfaces,
    build_reference_vmec_scan,
    load_neopax_reference_scan,
    scan_to_neopax_arrays,
    to_neopax_monoenergetic,
    write_neopax_scan_hdf5,
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
from .vmec_jax_backend import surface_from_vmec_jax_state, surface_from_vmec_jax_wout
from .vmec_jax_vmec import surface_from_vmec_jax_vmec_wout, surface_from_vmec_jax_vmec_wout_file
from .vmec_reference import (
    VmecReferenceFactors,
    load_vmec_surface_reference,
    vmec_reference_factors,
)

try:
    __version__ = version("ntx")
except PackageNotFoundError:
    __version__ = "0.1.0"

__all__ = [
    "__version__",
    "BoozmnSurface",
    "BoozerSurface",
    "CompiledPreparedSolver",
    "GridSpec",
    "MonoenergeticCase",
    "MonoenergeticDatabaseArrays",
    "VmecReferenceFactors",
    "NeopaxMonoenergeticArrays",
    "NeopaxScan",
    "PreparedMonoenergeticSystem",
    "TransportResult",
    "VmecSurface",
    "build_monoenergetic_database_arrays",
    "build_reference_vmec_scan",
    "build_ntx_neopax_scan",
    "build_ntx_neopax_scan_from_surfaces",
    "compile_prepared_solver",
    "compare_vmec_geometry_to_sfincs",
    "enable_x64",
    "example_surface",
    "load_boozmn_surface",
    "load_run_config",
    "load_boozer_modes_csv",
    "load_dkes_surface",
    "load_magnetic_configuration_surface",
    "load_neopax_reference_scan",
    "load_vmec_surface",
    "load_vmec_surface_reference",
    "vmec_reference_factors",
    "onsager_error",
    "prepare_monoenergetic_system",
    "run_from_input_file",
    "scan_to_neopax_arrays",
    "solve_monoenergetic",
    "solve_monoenergetic_internal",
    "solve_monoenergetic_scan",
    "solve_prepared",
    "solve_prepared_internal",
    "stack_monoenergetic_database_arrays",
    "surface_from_vmec_jax_state",
    "surface_from_vmec_jax_wout",
    "surface_from_vmec_jax_vmec_wout",
    "surface_from_vmec_jax_vmec_wout_file",
    "to_neopax_monoenergetic",
    "write_neopax_scan_hdf5",
]
