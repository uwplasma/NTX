"""Field-channel normalization for NTX-to-NEOPAX scan assembly."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
from jax import Array

from ._neopax_bridge import _surface_transport_scale
from .geometry import BoozerSurface, VmecSurface
from .grids import GridSpec


@dataclass(frozen=True)
class NeopaxScanFieldChannels:
    """Validated radial, collisionality, and electric-field scan channels."""

    rho: Array
    nu_v: Array
    Es: Array
    Er: Array
    drds: Array


def normalize_neopax_scan_field_channels(
    surfaces: tuple[BoozerSurface | VmecSurface, ...],
    *,
    rho: Array,
    nu_v: Array,
    Es: Array | None,
    Er: Array | None,
    drds: Array,
    grid: GridSpec,
) -> NeopaxScanFieldChannels:
    """Validate and complete the electric-field channels for a NEOPAX scan."""

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

    return NeopaxScanFieldChannels(
        rho=rho_arr,
        nu_v=nu_arr,
        Es=es_arr,
        Er=er_arr,
        drds=drds_arr,
    )


__all__ = ["NeopaxScanFieldChannels", "normalize_neopax_scan_field_channels"]
