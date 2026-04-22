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
from ._neopax_io import (
    load_neopax_reference_scan,
    neopax_scan_requires_rebuild,
    write_neopax_scan_hdf5,
)
from ._neopax_types import NeopaxMonoenergeticArrays, NeopaxScan
from .geometry import BoozerSurface, VmecSurface
from .grids import GridSpec
from .solver import solve_monoenergetic_scan

__all__ = [
    "NeopaxMonoenergeticArrays",
    "NeopaxScan",
    "build_ntx_neopax_scan",
    "build_ntx_neopax_scan_from_surfaces",
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
