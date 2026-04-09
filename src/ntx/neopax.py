"""Explicit NTX-to-NEOPAX mapping helpers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import jax.numpy as jnp
from jax import Array

from .geometry import BoozerSurface, VmecSurface
from .grids import GridSpec
from .solver import solve_monoenergetic_scan


@dataclass(frozen=True)
class NeopaxScan:
    """Monoenergetic scan data shaped for NEOPAX."""

    rho: Array
    nu_v: Array
    Er: Array
    Es: Array
    drds: Array
    D11: Array
    D13: Array
    D33: Array
    source_name: str | None = None


def load_neopax_reference_scan(path: str | Path) -> NeopaxScan:
    """Load a NEOPAX/REFERENCE_EXECUTABLE-style HDF5 monoenergetic table."""

    import h5py

    h5_path = Path(path).expanduser().resolve()
    with h5py.File(h5_path, "r") as handle:
        return NeopaxScan(
            rho=jnp.asarray(handle["rho"][()]),
            nu_v=jnp.asarray(handle["nu_v"][()]),
            Er=jnp.asarray(handle["Er"][()]),
            Es=jnp.asarray(handle["Es"][()]),
            drds=jnp.asarray(handle["drds"][()]),
            D11=jnp.asarray(handle["D11"][()]),
            D13=jnp.asarray(handle["D13"][()]),
            D33=jnp.asarray(handle["D33"][()]),
            source_name=h5_path.name,
        )


def build_ntx_neopax_scan(
    surface_loader: Callable[[float], BoozerSurface | VmecSurface],
    *,
    rho: Array,
    nu_v: Array,
    Es: Array,
    Er: Array,
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
    es_arr = jnp.asarray(Es)
    er_arr = jnp.asarray(Er)
    drds_arr = jnp.asarray(drds)
    if es_arr.shape != er_arr.shape:
        raise ValueError("Es and Er must have the same shape")
    if es_arr.shape[0] != rho_arr.shape[0]:
        raise ValueError("Es/Er first dimension must match rho")
    if drds_arr.shape[0] != rho_arr.shape[0]:
        raise ValueError("drds must have the same length as rho")

    d11_list = []
    d13_list = []
    d33_list = []
    for idx, rho_value in enumerate(rho_arr):
        surface = surface_loader(float(rho_value))
        nu_grid, es_grid = jnp.meshgrid(nu_arr, es_arr[idx], indexing="ij")
        coeffs = solve_monoenergetic_scan(surface, grid, nu_grid, epsi_hat=es_grid)
        d11_list.append(coeffs["D11"])
        d13_list.append(coeffs["D13"])
        d33_list.append(coeffs["D33"])

    return NeopaxScan(
        rho=rho_arr,
        nu_v=nu_arr,
        Er=er_arr,
        Es=es_arr,
        drds=drds_arr,
        D11=jnp.stack(d11_list),
        D13=jnp.stack(d13_list),
        D33=jnp.stack(d33_list),
        source_name=source_name,
    )


def to_neopax_monoenergetic(scan: NeopaxScan, *, a_b: float):
    """Construct `NEOPAX.Monoenergetic` from NTX scan data.

    This mapping follows NEOPAX's current REFERENCE_EXECUTABLE database conventions exactly,
    including the stored `drds` and `nu_v` rescalings.
    """

    try:
        import NEOPAX
    except ImportError as exc:  # pragma: no cover - exercised when NEOPAX exists locally
        raise ImportError("NEOPAX is required for `to_neopax_monoenergetic`") from exc

    rho = jnp.asarray(scan.rho)
    nu_v = jnp.asarray(scan.nu_v)
    er = jnp.asarray(scan.Er)
    drds = jnp.asarray(scan.drds)
    d11 = jnp.asarray(scan.D11)
    d13 = jnp.asarray(scan.D13)
    d33 = jnp.asarray(scan.D33)

    er_list = jnp.zeros((rho.shape[0], er.shape[1]), dtype=er.dtype)
    d11_scaled = d11 * drds[:, None, None] ** 2
    d13_scaled = d13 * drds[:, None, None]
    d33_scaled = d33 * nu_v[None, :, None]
    er0 = er[0]
    for j in range(rho.shape[0]):
        er_list = er_list.at[j].set(jnp.log10(jnp.maximum(1.0e-8, jnp.abs(er0) / (a_b * rho[j]))))

    return NEOPAX.Monoenergetic(
        a_b=float(a_b),
        rho=rho,
        nu_log=jnp.log10(nu_v),
        Er_list=er_list,
        D11_log=jnp.log10(d11_scaled),
        D13=d13_scaled,
        D33=d33_scaled,
    )
