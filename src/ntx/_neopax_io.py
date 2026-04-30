"""NEOPAX-style reference scan I/O."""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np

from ._neopax_types import NEOPAX_SCAN_FORMAT_VERSION, NeopaxScan


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
            D33_spitzer=_optional_dataset(handle, "D33_spitzer"),
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
            fac_monkes_to_sfincs_11=_optional_dataset(handle, "Fac_MONKES_TO_SFINCS_11"),
            fac_monkes_to_sfincs_31=_optional_dataset(handle, "Fac_MONKES_TO_SFINCS_31"),
            fac_monkes_to_sfincs_33=_optional_dataset(handle, "Fac_MONKES_TO_SFINCS_33"),
            fac_sfincs_to_dkes_11=_optional_dataset(handle, "Fac_SFINCS_TO_DKES_11"),
            fac_sfincs_to_dkes_31=_optional_dataset(handle, "Fac_SFINCS_TO_DKES_31"),
            fac_sfincs_to_dkes_33=_optional_dataset(handle, "Fac_SFINCS_TO_DKES_33"),
            fac_dkes_to_d11star=_optional_dataset(handle, "Fac_DKES_TO_D11star"),
            fac_dkes_to_d31star=_optional_dataset(handle, "Fac_DKES_TO_D31star"),
            fac_dkes_to_d33star=_optional_dataset(handle, "Fac_DKES_TO_D33star"),
            source_name=str(handle.attrs.get("source_name", h5_path.name)),
        )


def neopax_scan_requires_rebuild(path: str | Path) -> bool:
    """Return whether a cached NEOPAX-style scan is missing required fields."""

    import h5py

    h5_path = Path(path).expanduser().resolve()
    if not h5_path.exists():
        return True
    with h5py.File(h5_path, "r") as handle:
        format_version = int(handle.attrs.get("format_version", 0))
        return format_version < NEOPAX_SCAN_FORMAT_VERSION or "D33_spitzer" not in handle


def write_neopax_scan_hdf5(scan: NeopaxScan, path: str | Path) -> Path:
    """Write a NEOPAX-style HDF5 file from a scan payload."""

    import h5py

    output_path = Path(path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as handle:
        for name, values in _scan_datasets(scan):
            _write_dataset(handle, name, values)
        if scan.a_b is not None:
            handle.attrs["a_b"] = float(scan.a_b)
        if scan.psia is not None:
            handle.attrs["psia"] = float(scan.psia)
        if scan.source_name is not None:
            handle.attrs["source_name"] = scan.source_name
        handle.attrs["format_version"] = NEOPAX_SCAN_FORMAT_VERSION
    return output_path


def _optional_dataset(handle, name: str):
    if name not in handle:
        return None
    return jnp.asarray(handle[name][()])


def _write_dataset(handle, name: str, values) -> None:
    if values is None:
        return
    handle.create_dataset(name, data=np.asarray(values), track_times=False)


def _scan_datasets(scan: NeopaxScan):
    return (
        ("rho", scan.rho),
        ("nu_v", scan.nu_v),
        ("Er", scan.Er),
        ("Es", scan.Es),
        ("drds", scan.drds),
        ("D11", scan.D11),
        ("D13", scan.D13),
        ("D33", scan.D33),
        ("D33_spitzer", scan.D33_spitzer),
        ("D31", scan.D31),
        ("Er_tilde", scan.Er_tilde),
        ("Er_to_Ertilde", scan.Er_to_Ertilde),
        ("dr_tildedr", scan.dr_tildedr),
        ("dr_tildeds", scan.dr_tildeds),
        ("B00", scan.b00),
        ("R00", scan.r00),
        ("I", scan.boozer_i),
        ("G", scan.boozer_g),
        ("iota", scan.iota),
        ("Fac_REFERENCE_TO_SFINCS_11", scan.fac_reference_to_sfincs_11),
        ("Fac_REFERENCE_TO_SFINCS_31", scan.fac_reference_to_sfincs_31),
        ("Fac_REFERENCE_TO_SFINCS_33", scan.fac_reference_to_sfincs_33),
        ("Fac_MONKES_TO_SFINCS_11", scan.fac_monkes_to_sfincs_11),
        ("Fac_MONKES_TO_SFINCS_31", scan.fac_monkes_to_sfincs_31),
        ("Fac_MONKES_TO_SFINCS_33", scan.fac_monkes_to_sfincs_33),
        ("Fac_SFINCS_TO_DKES_11", scan.fac_sfincs_to_dkes_11),
        ("Fac_SFINCS_TO_DKES_31", scan.fac_sfincs_to_dkes_31),
        ("Fac_SFINCS_TO_DKES_33", scan.fac_sfincs_to_dkes_33),
        ("Fac_DKES_TO_D11star", scan.fac_dkes_to_d11star),
        ("Fac_DKES_TO_D31star", scan.fac_dkes_to_d31star),
        ("Fac_DKES_TO_D33star", scan.fac_dkes_to_d33star),
    )
