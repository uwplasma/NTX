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
from .vmec_reference_executable import load_vmec_surface_reference_executable_reference, reference_executable_vmec_factors


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
    D31: Array | None = None
    Er_tilde: Array | None = None
    Er_to_Ertilde: Array | None = None
    dr_tildedr: Array | None = None
    dr_tildeds: Array | None = None
    a_b: float | None = None
    psia: float | None = None
    b00: Array | None = None
    r00: Array | None = None
    boozer_i: Array | None = None
    boozer_g: Array | None = None
    iota: Array | None = None
    fac_reference_executable_to_sfincs_11: Array | None = None
    fac_reference_executable_to_sfincs_31: Array | None = None
    fac_reference_executable_to_sfincs_33: Array | None = None
    fac_sfincs_to_dkes_11: Array | None = None
    fac_sfincs_to_dkes_31: Array | None = None
    fac_sfincs_to_dkes_33: Array | None = None
    fac_dkes_to_d11star: Array | None = None
    fac_dkes_to_d31star: Array | None = None
    fac_dkes_to_d33star: Array | None = None
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
            D31=_optional_dataset(handle, "D31"),
            Er_tilde=_optional_dataset(handle, "Er_tilde"),
            Er_to_Ertilde=_optional_dataset(handle, "Er_to_Ertilde"),
            dr_tildedr=_optional_dataset(handle, "dr_tildedr"),
            dr_tildeds=_optional_dataset(handle, "dr_tildeds"),
            b00=_optional_dataset(handle, "B00"),
            r00=_optional_dataset(handle, "R00"),
            boozer_i=_optional_dataset(handle, "I"),
            boozer_g=_optional_dataset(handle, "G"),
            iota=_optional_dataset(handle, "iota"),
            fac_reference_executable_to_sfincs_11=_optional_dataset(handle, "Fac_REFERENCE_EXECUTABLE_TO_SFINCS_11"),
            fac_reference_executable_to_sfincs_31=_optional_dataset(handle, "Fac_REFERENCE_EXECUTABLE_TO_SFINCS_31"),
            fac_reference_executable_to_sfincs_33=_optional_dataset(handle, "Fac_REFERENCE_EXECUTABLE_TO_SFINCS_33"),
            fac_sfincs_to_dkes_11=_optional_dataset(handle, "Fac_SFINCS_TO_DKES_11"),
            fac_sfincs_to_dkes_31=_optional_dataset(handle, "Fac_SFINCS_TO_DKES_31"),
            fac_sfincs_to_dkes_33=_optional_dataset(handle, "Fac_SFINCS_TO_DKES_33"),
            fac_dkes_to_d11star=_optional_dataset(handle, "Fac_DKES_TO_D11star"),
            fac_dkes_to_d31star=_optional_dataset(handle, "Fac_DKES_TO_D31star"),
            fac_dkes_to_d33star=_optional_dataset(handle, "Fac_DKES_TO_D33star"),
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


def build_reference_executable_reference_vmec_scan(
    vmec_path: str | Path,
    booz_path: str | Path,
    *,
    rho: Array,
    nu_v: Array,
    er_tilde: Array,
    nt: int = 25,
    nz: int = 25,
    nl: int = 64,
    min_bmn_to_load: float = 0.0,
    source_name: str | None = None,
) -> NeopaxScan:
    """Build the W7-X VMEC monoenergetic database using Eduardo's REFERENCE_EXECUTABLE conventions."""

    rho_arr = jnp.asarray(rho)
    nu_arr = jnp.asarray(nu_v)
    er_tilde_arr = jnp.asarray(er_tilde)
    grid = GridSpec(n_theta=int(nt), n_zeta=int(nz), n_xi=int(nl) - 1)
    factors = reference_executable_vmec_factors(vmec_path, booz_path, rho_arr)

    es = er_tilde_arr[None, :] * factors.dr_tildeds[:, None] * factors.b00[:, None]
    er = er_tilde_arr[None, :] * factors.dr_tildedr[:, None] * factors.b00[:, None]
    er_to_er_tilde = jnp.broadcast_to(
        1.0 / factors.dr_tildedr[:, None],
        es.shape,
    )

    d11_list = []
    d31_list = []
    d13_list = []
    d33_list = []
    for rho_value, es_row in zip(rho_arr, es, strict=True):
        surface = load_vmec_surface_reference_executable_reference(
            vmec_path,
            s=float(rho_value**2),
            min_bmn_to_load=min_bmn_to_load,
        )
        nu_grid, es_grid = jnp.meshgrid(nu_arr, es_row, indexing="ij")
        coeffs = solve_monoenergetic_scan(surface, grid, nu_grid, epsi_hat=es_grid)
        d11_list.append(coeffs["D11"])
        d31_list.append(coeffs["D31"])
        d13_list.append(coeffs["D13"])
        d33_list.append(coeffs["D33"])

    return NeopaxScan(
        rho=rho_arr,
        nu_v=nu_arr,
        Er=er,
        Es=es,
        drds=factors.drds,
        D11=jnp.stack(d11_list),
        D31=jnp.stack(d31_list),
        D13=jnp.stack(d13_list),
        D33=jnp.stack(d33_list),
        Er_tilde=er_tilde_arr,
        Er_to_Ertilde=er_to_er_tilde,
        dr_tildedr=factors.dr_tildedr,
        dr_tildeds=factors.dr_tildeds,
        a_b=factors.a_b,
        psia=factors.psia,
        b00=factors.b00,
        r00=factors.r00,
        boozer_i=factors.boozer_i,
        boozer_g=factors.boozer_g,
        iota=factors.iota,
        fac_reference_executable_to_sfincs_11=factors.fac_reference_executable_to_sfincs_11,
        fac_reference_executable_to_sfincs_31=factors.fac_reference_executable_to_sfincs_31,
        fac_reference_executable_to_sfincs_33=factors.fac_reference_executable_to_sfincs_33,
        fac_sfincs_to_dkes_11=factors.fac_sfincs_to_dkes_11,
        fac_sfincs_to_dkes_31=factors.fac_sfincs_to_dkes_31,
        fac_sfincs_to_dkes_33=factors.fac_sfincs_to_dkes_33,
        fac_dkes_to_d11star=factors.fac_dkes_to_d11star,
        fac_dkes_to_d31star=factors.fac_dkes_to_d31star,
        fac_dkes_to_d33star=factors.fac_dkes_to_d33star,
        source_name=source_name,
    )


def write_neopax_scan_hdf5(scan: NeopaxScan, path: str | Path) -> Path:
    """Write a NEOPAX/REFERENCE_EXECUTABLE-style HDF5 file from a scan payload."""

    import h5py

    output_path = Path(path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as handle:
        _write_dataset(handle, "rho", scan.rho)
        _write_dataset(handle, "nu_v", scan.nu_v)
        _write_dataset(handle, "Er", scan.Er)
        _write_dataset(handle, "Es", scan.Es)
        _write_dataset(handle, "drds", scan.drds)
        _write_dataset(handle, "D11", scan.D11)
        _write_dataset(handle, "D13", scan.D13)
        _write_dataset(handle, "D33", scan.D33)
        _write_dataset(handle, "D31", scan.D31)
        _write_dataset(handle, "Er_tilde", scan.Er_tilde)
        _write_dataset(handle, "Er_to_Ertilde", scan.Er_to_Ertilde)
        _write_dataset(handle, "dr_tildedr", scan.dr_tildedr)
        _write_dataset(handle, "dr_tildeds", scan.dr_tildeds)
        _write_dataset(handle, "B00", scan.b00)
        _write_dataset(handle, "R00", scan.r00)
        _write_dataset(handle, "I", scan.boozer_i)
        _write_dataset(handle, "G", scan.boozer_g)
        _write_dataset(handle, "iota", scan.iota)
        _write_dataset(handle, "Fac_REFERENCE_EXECUTABLE_TO_SFINCS_11", scan.fac_reference_executable_to_sfincs_11)
        _write_dataset(handle, "Fac_REFERENCE_EXECUTABLE_TO_SFINCS_31", scan.fac_reference_executable_to_sfincs_31)
        _write_dataset(handle, "Fac_REFERENCE_EXECUTABLE_TO_SFINCS_33", scan.fac_reference_executable_to_sfincs_33)
        _write_dataset(handle, "Fac_SFINCS_TO_DKES_11", scan.fac_sfincs_to_dkes_11)
        _write_dataset(handle, "Fac_SFINCS_TO_DKES_31", scan.fac_sfincs_to_dkes_31)
        _write_dataset(handle, "Fac_SFINCS_TO_DKES_33", scan.fac_sfincs_to_dkes_33)
        _write_dataset(handle, "Fac_DKES_TO_D11star", scan.fac_dkes_to_d11star)
        _write_dataset(handle, "Fac_DKES_TO_D31star", scan.fac_dkes_to_d31star)
        _write_dataset(handle, "Fac_DKES_TO_D33star", scan.fac_dkes_to_d33star)
        if scan.a_b is not None:
            handle.attrs["a_b"] = float(scan.a_b)
        if scan.psia is not None:
            handle.attrs["psia"] = float(scan.psia)
        if scan.source_name is not None:
            handle.attrs["source_name"] = scan.source_name
    return output_path


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


def _optional_dataset(handle, name: str):
    if name not in handle:
        return None
    return jnp.asarray(handle[name][()])


def _write_dataset(handle, name: str, values) -> None:
    if values is None:
        return
    handle[name] = jnp.asarray(values)
