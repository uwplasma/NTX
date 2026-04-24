"""Explicit NTX-to-NEOPAX mapping helpers."""

from __future__ import annotations

from collections.abc import Callable

import jax.numpy as jnp
from jax import Array

from ._neopax_bridge import (
    _surface_reference_bridge,
    _surface_transport_scale,
    scan_to_neopax_arrays,
    to_neopax_monoenergetic,
)
from ._neopax_field import (
    build_differentiable_neopax_field,
    build_differentiable_neopax_field_from_vmec_jax_boundary_params,
    build_differentiable_neopax_field_from_vmec_jax_state,
)
from ._neopax_fluxes import get_differentiable_neopax_fluxes
from ._neopax_io import (
    load_neopax_reference_scan,
    neopax_scan_requires_rebuild,
    write_neopax_scan_hdf5,
)
from ._neopax_types import DifferentiableNeopaxField, NeopaxMonoenergeticArrays, NeopaxScan
from .geometry import BoozerSurface, VmecSurface
from .grids import GridSpec
from .solver import solve_monoenergetic_scan
from .vmec_jax_backend import (
    VmecJaxBoundaryContext,
    surfaces_from_vmec_jax_boundary_params,
    surfaces_from_vmec_jax_state,
)

__all__ = [
    "DifferentiableNeopaxField",
    "NeopaxMonoenergeticArrays",
    "NeopaxScan",
    "build_differentiable_neopax_field",
    "build_differentiable_neopax_field_from_vmec_jax_boundary_params",
    "build_differentiable_neopax_field_from_vmec_jax_state",
    "build_ntx_neopax_scan",
    "build_ntx_neopax_scan_from_vmec_jax_boundary_params",
    "build_ntx_neopax_scan_from_vmec_jax_state",
    "build_ntx_neopax_scan_from_surfaces",
    "get_differentiable_neopax_fluxes",
    "load_neopax_reference_scan",
    "neopax_scan_requires_rebuild",
    "scan_to_neopax_arrays",
    "to_neopax_monoenergetic",
    "write_neopax_scan_hdf5",
    "_surface_reference_bridge",
]


def build_ntx_neopax_scan(
    surface_loader: Callable[[float], BoozerSurface | VmecSurface],
    *,
    rho: Array,
    nu_v: Array,
    Es: Array | None = None,
    Er: Array | None = None,
    drds: Array,
    grid: GridSpec,
    source_name: str | None = None,
) -> NeopaxScan:
    """Build a NEOPAX-style scan from NTX surfaces.

    Parameters
    ----------
    surface_loader:
        Callable receiving one `rho` value and returning the corresponding NTX
        surface object.
    rho, nu_v, Es, Er, drds:
        Arrays following the same conventions as NEOPAX's reference HDF5 files.
    grid:
        NTX angular and Legendre resolution for the solve.
    """

    rho_arr = jnp.asarray(rho)
    nu_arr = jnp.asarray(nu_v)
    drds_arr = jnp.asarray(drds)
    if drds_arr.shape[0] != rho_arr.shape[0]:
        raise ValueError("drds must have the same length as rho")
    if Es is None and Er is None:
        raise ValueError("set at least one of Es or Er")

    surfaces = tuple(surface_loader(float(rho_value)) for rho_value in rho_arr)

    if Es is None:
        er_arr = jnp.asarray(Er)
        transport_scale = jnp.asarray(
            [_surface_transport_scale(surface) for surface in surfaces],
            dtype=grid.jax_dtype,
        )
        es_arr = er_arr / transport_scale[:, None]
    else:
        es_arr = jnp.asarray(Es)

    if Er is None:
        transport_scale = jnp.asarray(
            [_surface_transport_scale(surface) for surface in surfaces],
            dtype=grid.jax_dtype,
        )
        er_arr = es_arr * transport_scale[:, None]
    else:
        er_arr = jnp.asarray(Er)

    if es_arr.shape != er_arr.shape:
        raise ValueError("Es and Er must have the same shape")
    if es_arr.shape[0] != rho_arr.shape[0]:
        raise ValueError("Es/Er first dimension must match rho")

    return build_ntx_neopax_scan_from_surfaces(
        surfaces,
        rho=rho_arr,
        nu_v=nu_arr,
        Es=es_arr,
        Er=er_arr,
        drds=drds_arr,
        grid=grid,
        source_name=source_name,
    )


def build_ntx_neopax_scan_from_surfaces(
    surfaces: tuple[BoozerSurface | VmecSurface, ...],
    *,
    rho: Array,
    nu_v: Array,
    Es: Array | None = None,
    Er: Array | None = None,
    drds: Array,
    grid: GridSpec,
    source_name: str | None = None,
) -> NeopaxScan:
    """Build a NEOPAX-style scan from an explicit tuple of NTX surfaces.

    This is the intended imported path when the caller already has surface
    objects in memory and wants to avoid a Python callback boundary.
    """

    rho_arr = jnp.asarray(rho)
    nu_arr = jnp.asarray(nu_v)
    drds_arr = jnp.asarray(drds)
    if len(surfaces) != rho_arr.shape[0]:
        raise ValueError("number of surfaces must match rho length")
    if drds_arr.shape[0] != rho_arr.shape[0]:
        raise ValueError("drds must have the same length as rho")
    if Es is None and Er is None:
        raise ValueError("set at least one of Es or Er")

    transport_scale = jnp.asarray(
        [_surface_transport_scale(surface) for surface in surfaces],
        dtype=grid.jax_dtype,
    )

    if Es is None:
        er_arr = jnp.asarray(Er)
        es_arr = er_arr / transport_scale[:, None]
    else:
        es_arr = jnp.asarray(Es)

    if Er is None:
        er_arr = es_arr * transport_scale[:, None]
    else:
        er_arr = jnp.asarray(Er)

    if es_arr.shape != er_arr.shape:
        raise ValueError("Es and Er must have the same shape")
    if es_arr.shape[0] != rho_arr.shape[0]:
        raise ValueError("Es/Er first dimension must match rho")

    d11_list = []
    d13_list = []
    d33_list = []
    d33_spitzer_list = []
    b00_list = []
    boozer_i_list = []
    boozer_g_list = []
    iota_list = []
    fac_11_list = []
    fac_31_list = []
    fac_33_list = []
    sfincs_to_dkes_11_list = []
    sfincs_to_dkes_31_list = []
    sfincs_to_dkes_33_list = []
    for surface, es_row in zip(surfaces, es_arr, strict=True):
        nu_grid, es_grid = jnp.meshgrid(nu_arr, es_row, indexing="ij")
        coeffs = solve_monoenergetic_scan(surface, grid, nu_grid, epsi_hat=es_grid)
        d11_list.append(coeffs["D11"])
        d13_list.append(coeffs["D13"])
        d33_list.append(coeffs["D33"])
        d33_spitzer_list.append(coeffs["D33_spitzer"])
        bridge = _surface_reference_bridge(surface)
        b00_list.append(bridge["b00"])
        boozer_i_list.append(bridge["boozer_i"])
        boozer_g_list.append(bridge["boozer_g"])
        iota_list.append(bridge["iota"])
        fac_11_list.append(bridge["fac_11"])
        fac_31_list.append(bridge["fac_31"])
        fac_33_list.append(bridge["fac_33"])
        sfincs_to_dkes_11_list.append(bridge["fac_sfincs_to_dkes_11"])
        sfincs_to_dkes_31_list.append(bridge["fac_sfincs_to_dkes_31"])
        sfincs_to_dkes_33_list.append(bridge["fac_sfincs_to_dkes_33"])

    return NeopaxScan(
        rho=rho_arr,
        nu_v=nu_arr,
        Er=er_arr,
        Es=es_arr,
        drds=drds_arr,
        D11=jnp.stack(d11_list),
        D13=jnp.stack(d13_list),
        D33=jnp.stack(d33_list),
        D33_spitzer=jnp.stack(d33_spitzer_list),
        b00=jnp.asarray(b00_list),
        boozer_i=jnp.asarray(boozer_i_list),
        boozer_g=jnp.asarray(boozer_g_list),
        iota=jnp.asarray(iota_list),
        fac_reference_to_sfincs_11=jnp.asarray(fac_11_list),
        fac_reference_to_sfincs_31=jnp.asarray(fac_31_list),
        fac_reference_to_sfincs_33=jnp.asarray(fac_33_list),
        fac_sfincs_to_dkes_11=jnp.asarray(sfincs_to_dkes_11_list),
        fac_sfincs_to_dkes_31=jnp.asarray(sfincs_to_dkes_31_list),
        fac_sfincs_to_dkes_33=jnp.asarray(sfincs_to_dkes_33_list),
        source_name=source_name,
    )


def build_ntx_neopax_scan_from_vmec_jax_state(
    *,
    state,
    static,
    indata,
    signgs: int,
    rho: Array,
    nu_v: Array,
    Es: Array | None = None,
    Er: Array | None = None,
    drds: Array,
    grid: GridSpec,
    source_name: str | None = None,
    mboz: int = 12,
    nboz: int = 12,
    psi_p: float = 1.0,
    min_bmn_to_load: float = 0.0,
) -> NeopaxScan:
    """Build a NEOPAX-style scan directly from an in-memory `vmec_jax` state."""

    rho_arr = jnp.asarray(rho)
    s_values = tuple(float(rho_value**2) for rho_value in rho_arr)
    surfaces = surfaces_from_vmec_jax_state(
        state=state,
        static=static,
        indata=indata,
        signgs=signgs,
        s_values=s_values,
        mboz=mboz,
        nboz=nboz,
        psi_p=psi_p,
        min_bmn_to_load=min_bmn_to_load,
    )
    return build_ntx_neopax_scan_from_surfaces(
        surfaces,
        rho=rho_arr,
        nu_v=nu_v,
        Es=Es,
        Er=Er,
        drds=drds,
        grid=grid,
        source_name=source_name,
    )


def build_ntx_neopax_scan_from_vmec_jax_boundary_params(
    context: VmecJaxBoundaryContext,
    params,
    *,
    rho: Array,
    nu_v: Array,
    Es: Array | None = None,
    Er: Array | None = None,
    drds: Array,
    grid: GridSpec,
    source_name: str | None = None,
    vmec_project: bool = True,
    max_iter: int = 50,
    step_size: float = 1.0,
    ftol: float | None = None,
    implicit=None,
    mboz: int = 12,
    nboz: int = 12,
    psi_p: float = 1.0,
    min_bmn_to_load: float = 0.0,
) -> NeopaxScan:
    """Solve a fixed boundary and build a NEOPAX-style scan from the result."""

    rho_arr = jnp.asarray(rho)
    s_values = tuple(float(rho_value**2) for rho_value in rho_arr)
    surfaces = surfaces_from_vmec_jax_boundary_params(
        context,
        params,
        s_values=s_values,
        vmec_project=vmec_project,
        max_iter=max_iter,
        step_size=step_size,
        ftol=ftol,
        implicit=implicit,
        mboz=mboz,
        nboz=nboz,
        psi_p=psi_p,
        min_bmn_to_load=min_bmn_to_load,
    )
    return build_ntx_neopax_scan_from_surfaces(
        surfaces,
        rho=rho_arr,
        nu_v=nu_v,
        Es=Es,
        Er=Er,
        drds=drds,
        grid=grid,
        source_name=source_name,
    )
