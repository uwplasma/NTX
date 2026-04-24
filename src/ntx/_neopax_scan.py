"""NEOPAX scan assembly from NTX surfaces and imported VMEC-JAX states."""

from __future__ import annotations

from collections.abc import Callable

import jax.numpy as jnp
from jax import Array

from ._neopax_scan_coefficients import solve_neopax_scan_coefficient_blocks
from ._neopax_scan_fields import normalize_neopax_scan_field_channels
from ._neopax_types import NeopaxScan
from .geometry import BoozerSurface, VmecSurface
from .grids import GridSpec
from .vmec_jax_backend import (
    VmecJaxBoundaryContext,
    surfaces_from_vmec_jax_boundary_params,
    surfaces_from_vmec_jax_state,
)


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
    drds_arr = jnp.asarray(drds)
    if drds_arr.shape[0] != rho_arr.shape[0]:
        raise ValueError("drds must have the same length as rho")
    if Es is None and Er is None:
        raise ValueError("set at least one of Es or Er")

    surfaces = tuple(surface_loader(float(rho_value)) for rho_value in rho_arr)
    return build_ntx_neopax_scan_from_surfaces(
        surfaces,
        rho=rho_arr,
        nu_v=nu_v,
        Es=Es,
        Er=Er,
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

    channels = normalize_neopax_scan_field_channels(
        surfaces,
        rho=rho,
        nu_v=nu_v,
        Es=Es,
        Er=Er,
        drds=drds,
        grid=grid,
    )
    blocks = solve_neopax_scan_coefficient_blocks(
        surfaces,
        Es=channels.Es,
        nu_v=channels.nu_v,
        grid=grid,
    )

    return NeopaxScan(
        rho=channels.rho,
        nu_v=channels.nu_v,
        Er=channels.Er,
        Es=channels.Es,
        drds=channels.drds,
        D11=blocks.D11,
        D13=blocks.D13,
        D33=blocks.D33,
        D33_spitzer=blocks.D33_spitzer,
        b00=blocks.b00,
        boozer_i=blocks.boozer_i,
        boozer_g=blocks.boozer_g,
        iota=blocks.iota,
        fac_reference_to_sfincs_11=blocks.fac_reference_to_sfincs_11,
        fac_reference_to_sfincs_31=blocks.fac_reference_to_sfincs_31,
        fac_reference_to_sfincs_33=blocks.fac_reference_to_sfincs_33,
        fac_sfincs_to_dkes_11=blocks.fac_sfincs_to_dkes_11,
        fac_sfincs_to_dkes_31=blocks.fac_sfincs_to_dkes_31,
        fac_sfincs_to_dkes_33=blocks.fac_sfincs_to_dkes_33,
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
