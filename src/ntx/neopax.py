"""Explicit NTX-to-NEOPAX mapping helpers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import jax.numpy as jnp
from jax import Array, tree_util

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
    fac_reference_to_sfincs_11: Array | None = None
    fac_reference_to_sfincs_31: Array | None = None
    fac_reference_to_sfincs_33: Array | None = None
    fac_sfincs_to_dkes_11: Array | None = None
    fac_sfincs_to_dkes_31: Array | None = None
    fac_sfincs_to_dkes_33: Array | None = None
    fac_dkes_to_d11star: Array | None = None
    fac_dkes_to_d31star: Array | None = None
    fac_dkes_to_d33star: Array | None = None
    source_name: str | None = None


tree_util.register_dataclass(
    NeopaxScan,
    data_fields=(
        "rho",
        "nu_v",
        "Er",
        "Es",
        "drds",
        "D11",
        "D13",
        "D33",
        "D31",
        "Er_tilde",
        "Er_to_Ertilde",
        "dr_tildedr",
        "dr_tildeds",
        "a_b",
        "psia",
        "b00",
        "r00",
        "boozer_i",
        "boozer_g",
        "iota",
        "fac_reference_to_sfincs_11",
        "fac_reference_to_sfincs_31",
        "fac_reference_to_sfincs_33",
        "fac_sfincs_to_dkes_11",
        "fac_sfincs_to_dkes_31",
        "fac_sfincs_to_dkes_33",
        "fac_dkes_to_d11star",
        "fac_dkes_to_d31star",
        "fac_dkes_to_d33star",
    ),
    meta_fields=("source_name",),
)


@dataclass(frozen=True)
class NeopaxMonoenergeticArrays:
    """Pure-array NEOPAX mapping payload for differentiable imported workflows."""

    a_b: Array
    rho: Array
    nu_log: Array
    Er_list: Array
    D11_log: Array
    D13: Array
    D33: Array


tree_util.register_dataclass(
    NeopaxMonoenergeticArrays,
    data_fields=("a_b", "rho", "nu_log", "Er_list", "D11_log", "D13", "D33"),
    meta_fields=(),
)


def load_neopax_reference_scan(path: str | Path) -> NeopaxScan:
    """Load a NEOPAX-style HDF5 monoenergetic table."""

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
            fac_reference_to_sfincs_11=_optional_dataset(handle, "Fac_REFERENCE_TO_SFINCS_11"),
            fac_reference_to_sfincs_31=_optional_dataset(handle, "Fac_REFERENCE_TO_SFINCS_31"),
            fac_reference_to_sfincs_33=_optional_dataset(handle, "Fac_REFERENCE_TO_SFINCS_33"),
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

    surfaces = tuple(surface_loader(float(rho_value)) for rho_value in rho_arr)
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
    Es: Array,
    Er: Array,
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
    es_arr = jnp.asarray(Es)
    er_arr = jnp.asarray(Er)
    drds_arr = jnp.asarray(drds)
    if len(surfaces) != rho_arr.shape[0]:
        raise ValueError("number of surfaces must match rho length")
    if es_arr.shape != er_arr.shape:
        raise ValueError("Es and Er must have the same shape")
    if es_arr.shape[0] != rho_arr.shape[0]:
        raise ValueError("Es/Er first dimension must match rho")
    if drds_arr.shape[0] != rho_arr.shape[0]:
        raise ValueError("drds must have the same length as rho")

    d11_list = []
    d13_list = []
    d33_list = []
    for surface, es_row in zip(surfaces, es_arr, strict=True):
        nu_grid, es_grid = jnp.meshgrid(nu_arr, es_row, indexing="ij")
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

def write_neopax_scan_hdf5(scan: NeopaxScan, path: str | Path) -> Path:
    """Write a NEOPAX-style HDF5 file from a scan payload."""

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
        _write_dataset(handle, "Fac_REFERENCE_TO_SFINCS_11", scan.fac_reference_to_sfincs_11)
        _write_dataset(handle, "Fac_REFERENCE_TO_SFINCS_31", scan.fac_reference_to_sfincs_31)
        _write_dataset(handle, "Fac_REFERENCE_TO_SFINCS_33", scan.fac_reference_to_sfincs_33)
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


def scan_to_neopax_arrays(scan: NeopaxScan, *, a_b: float | Array) -> NeopaxMonoenergeticArrays:
    """Map NTX scan data into the pure arrays consumed by `NEOPAX.Monoenergetic`.

    This path is JAX-friendly and is the right place to keep imported,
    differentiable workflows before constructing the external NEOPAX object.
    """

    rho = jnp.asarray(scan.rho)
    nu_v = jnp.asarray(scan.nu_v)
    er = jnp.asarray(scan.Er)
    drds = jnp.asarray(scan.drds)
    d11 = jnp.asarray(scan.D11)
    d13 = jnp.asarray(scan.D13)
    d33 = jnp.asarray(scan.D33)
    a_b_value = jnp.asarray(a_b)

    er0 = er[0]
    er_list = jnp.stack(
        [
            jnp.log10(jnp.maximum(1.0e-8, jnp.abs(er0) / (a_b_value * rho_value)))
            for rho_value in rho
        ]
    )
    return NeopaxMonoenergeticArrays(
        a_b=a_b_value,
        rho=rho,
        nu_log=jnp.log10(nu_v),
        Er_list=er_list,
        D11_log=jnp.log10(d11 * drds[:, None, None] ** 2),
        D13=d13 * drds[:, None, None],
        D33=d33 * nu_v[None, :, None],
    )


def to_neopax_monoenergetic(scan: NeopaxScan, *, a_b: float):
    """Construct `NEOPAX.Monoenergetic` from NTX scan data.

    This mapping follows the current NEOPAX monoenergetic database conventions,
    including the stored `drds` and `nu_v` rescalings.
    """

    try:
        import NEOPAX
    except ImportError as exc:  # pragma: no cover - exercised when NEOPAX exists locally
        raise ImportError("NEOPAX is required for `to_neopax_monoenergetic`") from exc

    arrays = scan_to_neopax_arrays(scan, a_b=a_b)

    return NEOPAX.Monoenergetic(
        a_b=float(a_b),
        rho=arrays.rho,
        nu_log=arrays.nu_log,
        Er_list=arrays.Er_list,
        D11_log=arrays.D11_log,
        D13=arrays.D13,
        D33=arrays.D33,
    )


def _optional_dataset(handle, name: str):
    if name not in handle:
        return None
    return jnp.asarray(handle[name][()])


def _write_dataset(handle, name: str, values) -> None:
    if values is None:
        return
    handle[name] = jnp.asarray(values)
