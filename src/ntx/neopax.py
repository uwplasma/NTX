"""Explicit NTX-to-NEOPAX mapping helpers."""

from __future__ import annotations

from ._neopax_bridge import (
    _surface_reference_bridge,
    scan_to_neopax_arrays,
    to_neopax_monoenergetic,
)
from ._neopax_field import (
    build_differentiable_neopax_field,
    build_differentiable_neopax_field_from_vmec_booz_files,
    build_differentiable_neopax_field_from_vmex_boundary_params,
    build_differentiable_neopax_field_from_vmex_state,
)
from ._neopax_fluxes import get_differentiable_neopax_fluxes
from ._neopax_io import (
    load_neopax_reference_scan,
    neopax_scan_requires_rebuild,
    write_neopax_scan_hdf5,
)
from ._neopax_scan import (
    build_ntx_neopax_scan,
    build_ntx_neopax_scan_from_surfaces,
    build_ntx_neopax_scan_from_vmex_boundary_params,
    build_ntx_neopax_scan_from_vmex_state,
)
from ._neopax_scan_coefficients import (
    NeopaxScanCoefficientBlocks,
    NeopaxScanCoefficientPrimalRecord,
    pullback_neopax_scan_coefficient_blocks_from_primal_record,
    pullback_neopax_scan_coefficient_blocks_from_primal_record_batched,
)
from ._neopax_types import DifferentiableNeopaxField, NeopaxMonoenergeticArrays, NeopaxScan

__all__ = [
    "DifferentiableNeopaxField",
    "NeopaxMonoenergeticArrays",
    "NeopaxScan",
    "NeopaxScanCoefficientBlocks",
    "NeopaxScanCoefficientPrimalRecord",
    "build_differentiable_neopax_field",
    "build_differentiable_neopax_field_from_vmec_booz_files",
    "build_differentiable_neopax_field_from_vmex_boundary_params",
    "build_differentiable_neopax_field_from_vmex_state",
    "build_ntx_neopax_scan",
    "build_ntx_neopax_scan_from_vmex_boundary_params",
    "build_ntx_neopax_scan_from_vmex_state",
    "build_ntx_neopax_scan_from_surfaces",
    "pullback_neopax_scan_coefficient_blocks_from_primal_record",
    "pullback_neopax_scan_coefficient_blocks_from_primal_record_batched",
    "get_differentiable_neopax_fluxes",
    "load_neopax_reference_scan",
    "neopax_scan_requires_rebuild",
    "scan_to_neopax_arrays",
    "to_neopax_monoenergetic",
    "write_neopax_scan_hdf5",
    "_surface_reference_bridge",
]
