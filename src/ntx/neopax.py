"""Explicit NTX-to-NEOPAX mapping helpers."""

from __future__ import annotations

from ._neopax import (
    DifferentiableNeopaxField,
    NeopaxMonoenergeticArrays,
    NeopaxScan,
    _surface_reference_bridge,
    build_differentiable_neopax_field,
    build_differentiable_neopax_field_from_vmec_booz_files,
    build_differentiable_neopax_field_from_vmex_boundary_params,
    build_differentiable_neopax_field_from_vmex_state,
    get_differentiable_neopax_fluxes,
    load_neopax_reference_scan,
    neopax_scan_requires_rebuild,
    scan_to_neopax_arrays,
    to_neopax_monoenergetic,
    write_neopax_scan_hdf5,
)
from ._neopax_scan import (
    build_ntx_neopax_scan,
    build_ntx_neopax_scan_from_surfaces,
    build_ntx_neopax_scan_from_vmex_boundary_params,
    build_ntx_neopax_scan_from_vmex_state,
)

__all__ = [
    "DifferentiableNeopaxField",
    "NeopaxMonoenergeticArrays",
    "NeopaxScan",
    "build_differentiable_neopax_field",
    "build_differentiable_neopax_field_from_vmec_booz_files",
    "build_differentiable_neopax_field_from_vmex_boundary_params",
    "build_differentiable_neopax_field_from_vmex_state",
    "build_ntx_neopax_scan",
    "build_ntx_neopax_scan_from_vmex_boundary_params",
    "build_ntx_neopax_scan_from_vmex_state",
    "build_ntx_neopax_scan_from_surfaces",
    "get_differentiable_neopax_fluxes",
    "load_neopax_reference_scan",
    "neopax_scan_requires_rebuild",
    "scan_to_neopax_arrays",
    "to_neopax_monoenergetic",
    "write_neopax_scan_hdf5",
    "_surface_reference_bridge",
]
