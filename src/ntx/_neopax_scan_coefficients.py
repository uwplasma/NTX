"""Coefficient and normalization-block assembly for NEOPAX scans."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
from jax import Array

from ._neopax_bridge import _surface_reference_bridge
from .geometry import BoozerSurface, VmecSurface
from .grids import GridSpec
from .solver import solve_monoenergetic_scan


@dataclass(frozen=True)
class NeopaxScanCoefficientBlocks:
    """Solved monoenergetic blocks plus reference-normalization metadata."""

    D11: Array
    D13: Array
    D33: Array
    D33_spitzer: Array
    b00: Array
    boozer_i: Array
    boozer_g: Array
    iota: Array
    fac_reference_to_sfincs_11: Array
    fac_reference_to_sfincs_31: Array
    fac_reference_to_sfincs_33: Array
    fac_sfincs_to_dkes_11: Array
    fac_sfincs_to_dkes_31: Array
    fac_sfincs_to_dkes_33: Array


def solve_neopax_scan_coefficient_blocks(
    surfaces: tuple[BoozerSurface | VmecSurface, ...],
    *,
    Es: Array,
    nu_v: Array,
    grid: GridSpec,
) -> NeopaxScanCoefficientBlocks:
    """Solve all surface/electric-field blocks used by a NEOPAX scan."""

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
    for surface, es_row in zip(surfaces, Es, strict=True):
        nu_grid, es_grid = jnp.meshgrid(nu_v, es_row, indexing="ij")
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

    return NeopaxScanCoefficientBlocks(
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
    )


__all__ = ["NeopaxScanCoefficientBlocks", "solve_neopax_scan_coefficient_blocks"]
