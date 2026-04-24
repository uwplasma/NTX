"""Compatibility facade for `vmec_jax -> booz_xform_jax -> NTX` workflows."""

from __future__ import annotations

from ._vmec_jax_boozer import (
    _apply_boozer_sign_convention,
    _booz_xform_bundle_from_vmec_jax_state,
    _import_booz_xform_jax_api,
    _import_vmec_jax,
    _prepend_checkout,
)
from ._vmec_jax_boundary import (
    VmecJaxBoundaryContext,
    build_vmec_jax_boundary_context,
    initial_guess_vmec_jax_boundary_state,
    relax_vmec_jax_boundary_state_explicit,
    solve_vmec_jax_boundary_state,
)
from ._vmec_jax_surfaces import (
    surface_from_vmec_jax_state,
    surface_from_vmec_jax_wout,
    surfaces_from_vmec_jax_boundary_params,
    surfaces_from_vmec_jax_state,
)

__all__ = [
    "VmecJaxBoundaryContext",
    "build_vmec_jax_boundary_context",
    "initial_guess_vmec_jax_boundary_state",
    "relax_vmec_jax_boundary_state_explicit",
    "solve_vmec_jax_boundary_state",
    "surface_from_vmec_jax_state",
    "surface_from_vmec_jax_wout",
    "surfaces_from_vmec_jax_boundary_params",
    "surfaces_from_vmec_jax_state",
    "_apply_boozer_sign_convention",
    "_booz_xform_bundle_from_vmec_jax_state",
    "_import_booz_xform_jax_api",
    "_import_vmec_jax",
    "_prepend_checkout",
]
