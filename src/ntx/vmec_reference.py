"""Reference-oriented VMEC helpers for NTX database and benchmark workflows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

import jax.numpy as jnp
import numpy as np

from .geometry import VmecSurface


@dataclass(frozen=True)
class VmecReferenceFactors:
    """Conversion factors used by the NTX reference-database VMEC workflow."""

    rho: jnp.ndarray
    a_b: float
    psia: float
    b00: jnp.ndarray
    r00: jnp.ndarray
    iota: jnp.ndarray
    boozer_i: jnp.ndarray
    boozer_g: jnp.ndarray
    drds: jnp.ndarray
    dr_tildedr: jnp.ndarray
    dr_tildeds: jnp.ndarray
    fac_reference_to_sfincs_11: jnp.ndarray
    fac_reference_to_sfincs_31: jnp.ndarray
    fac_reference_to_sfincs_33: jnp.ndarray
    fac_sfincs_to_dkes_11: jnp.ndarray
    fac_sfincs_to_dkes_31: jnp.ndarray
    fac_sfincs_to_dkes_33: jnp.ndarray
    fac_dkes_to_d11star: jnp.ndarray
    fac_dkes_to_d31star: jnp.ndarray
    fac_dkes_to_d33star: jnp.ndarray


class _VmecWoutData(TypedDict):
    ns: int
    nfp: int
    mpol: int
    ntor: int
    Aminor_p: float
    volume_p: float
    phi: np.ndarray
    iotaf: np.ndarray
    xm_nyq: np.ndarray
    xn_nyq: np.ndarray
    gmnc: np.ndarray
    bmnc: np.ndarray
    bsupumnc: np.ndarray
    bsupvmnc: np.ndarray
    bsubumnc: np.ndarray
    bsubvmnc: np.ndarray


class _BoozData(TypedDict):
    bmnc_b: np.ndarray
    rmnc_b: np.ndarray
    buco_b: np.ndarray
    bvco_b: np.ndarray
    ixm_b: np.ndarray
    ixn_b: np.ndarray
    jlist: np.ndarray


def load_vmec_surface_reference(
    path: str | Path,
    *,
    s: float,
    min_bmn_to_load: float = 0.0,
) -> VmecSurface:
    """Load one VMEC surface using the reference database conventions.

    This path is comparison-oriented. It reproduces the conventions used by the
    external W7-X reference database closely enough to make pointwise audits and
    downstream NEOPAX database comparisons meaningful.
    """

    wout_path = Path(path).expanduser().resolve()
    if not wout_path.exists():
        raise FileNotFoundError(str(wout_path))
    if not 0.0 <= float(s) <= 1.0:
        raise ValueError("s must be between 0 and 1")

    data = _read_vmec_wout_netcdf(wout_path)
    ns = int(data["ns"])
    if ns < 2:
        raise ValueError("VMEC input must contain at least two radial surfaces")

    s_full = np.linspace(0.0, 1.0, ns, dtype=np.float64)
    hs = 1.0 / (ns - 1)
    s_half = s_full[:-1] + 0.5 * hs

    bmnc = _interp_mode_columns(s_half, data["bmnc"][1:, :], s)
    gmnc = _interp_mode_columns(s_half, data["gmnc"][1:, :], s)
    bsupumnc = _interp_mode_columns(s_half, data["bsupumnc"][1:, :], s)
    bsupvmnc = _interp_mode_columns(s_half, data["bsupvmnc"][1:, :], s)
    bsubumnc = _interp_mode_columns(s_half, data["bsubumnc"][1:, :], s)
    bsubvmnc = _interp_mode_columns(s_half, data["bsubvmnc"][1:, :], s)
    iota = -float(_interp_profile(s_full, data["iotaf"], s))

    b0 = float(np.max(np.abs(bmnc)))
    if b0 == 0.0:
        raise ValueError("selected VMEC surface has zero magnetic-field strength")

    include = np.abs(bmnc) >= float(min_bmn_to_load) * b0
    zero_mode = (data["xm_nyq"] == 0) & (data["xn_nyq"] == 0)
    if np.any(zero_mode):
        include[np.argmax(zero_mode)] = True

    aminor_p = float(data["Aminor_p"])
    r_n = float(np.sqrt(float(s)))
    r_hat = float(aminor_p * r_n)

    return VmecSurface(
        path=wout_path,
        requested_psi_n=float(s),
        psi_n=float(s),
        nfp=int(data["nfp"]),
        ns=ns,
        mpol=int(data["mpol"]),
        ntor=int(data["ntor"]),
        total_mode_count=int(data["xm_nyq"].size),
        loaded_mode_count=int(np.count_nonzero(include)),
        iota=iota,
        m=jnp.asarray(data["xm_nyq"][include], dtype=jnp.int32),
        n=jnp.asarray(np.rint(-data["xn_nyq"][include] / float(data["nfp"])).astype(np.int32)),
        b_cos=jnp.asarray(bmnc[include], dtype=jnp.float64),
        jacobian_cos=jnp.asarray(gmnc[include], dtype=jnp.float64),
        b_sub_theta_cos=jnp.asarray(bsubumnc[include], dtype=jnp.float64),
        b_sub_zeta_cos=jnp.asarray(bsubvmnc[include], dtype=jnp.float64),
        b_sup_theta_cos=jnp.asarray(bsupumnc[include], dtype=jnp.float64),
        b_sup_zeta_cos=jnp.asarray(bsupvmnc[include], dtype=jnp.float64),
        b0=b0,
        psi_a_hat=float(abs(data["phi"][-1]) / (2.0 * np.pi)),
        phi_edge=float(data["phi"][-1]),
        r_n=r_n,
        r_hat=r_hat,
        dpsi_hat_dr_hat=1.0,
        dr_hat_dpsi_hat=1.0,
        aminor_p=aminor_p,
        psi_p=None,
        transport_psi_scale=1.0,
    )


def vmec_reference_factors(
    vmec_path: str | Path,
    booz_path: str | Path,
    rho: jnp.ndarray | np.ndarray,
) -> VmecReferenceFactors:
    """Return the Boozer-side conversion factors used by the reference workflow."""

    wout = _read_vmec_wout_netcdf(Path(vmec_path).expanduser().resolve())
    booz = _read_booz_netcdf(Path(booz_path).expanduser().resolve())

    rho_arr = np.asarray(rho, dtype=np.float64)
    if rho_arr.ndim != 1:
        raise ValueError("rho must be a 1D array")

    ns = int(wout["ns"])
    s_half = np.asarray([(index - 0.5) / (ns - 1) for index in range(ns)], dtype=np.float64)
    rho_half = np.sqrt(np.clip(s_half, 0.0, None))
    s_full = np.linspace(0.0, 1.0, ns, dtype=np.float64)
    rho_full = np.sqrt(s_full)
    jlist = np.asarray(booz["jlist"], dtype=np.int64)
    rho_booz_b = rho_half[jlist - 1]
    rho_booz_r = rho_full[jlist - 1]

    psia = float(abs(wout["phi"][-1]) / (2.0 * np.pi))
    volume_p = float(wout["volume_p"])
    r00_mode = _mode_index(booz["ixm_b"], booz["ixn_b"], 0, 0)
    r0_b = float(booz["rmnc_b"][-1, r00_mode])
    a_b = float(np.sqrt(volume_p / (2.0 * np.pi**2 * r0_b)))

    b00 = np.asarray(_interp_profile(rho_booz_b, booz["bmnc_b"][:, r00_mode], rho_arr))
    r00 = np.asarray(_interp_profile(rho_booz_r, booz["rmnc_b"][:, r00_mode], rho_arr))
    boozer_i = np.asarray(_interp_profile(rho_half[1:], booz["buco_b"][1:], rho_arr))
    boozer_g = np.asarray(_interp_profile(rho_half[1:], booz["bvco_b"][1:], rho_arr))
    iota = np.asarray(_interp_profile(rho_full, wout["iotaf"], rho_arr))

    drds = a_b / (2.0 * rho_arr)
    dr_tildedr = 2.0 * psia / (a_b**2 * b00)
    dr_tildeds = dr_tildedr * drds
    dpsidrtilde = rho_arr * a_b * b00

    fac_reference_to_sfincs_11 = (
        8.0
        * (boozer_g + iota * boozer_i)
        * b00
        * psia**2
        / (np.sqrt(np.pi) * boozer_g**2)
    )
    fac_reference_to_sfincs_31 = 4.0 * b00 * psia / (np.sqrt(np.pi) * boozer_g)
    fac_reference_to_sfincs_33 = -2.0 * b00 / ((boozer_g + iota * boozer_i) * np.sqrt(np.pi))

    fac_sfincs_to_dkes_11 = 1.0 / (
        8.0
        * (boozer_g + iota * boozer_i)
        / (boozer_g**2 * b00 * np.sqrt(np.pi))
        * dpsidrtilde**2
    )
    fac_sfincs_to_dkes_31 = 1.0 / (4.0 * dpsidrtilde / (boozer_g * np.sqrt(np.pi)))
    fac_sfincs_to_dkes_33 = 1.0 / fac_reference_to_sfincs_33

    epsilon_t = rho_arr * a_b / r00
    fac_dkes_to_d11star = -8.0 * iota * r00 / np.pi
    fac_dkes_to_d31star = -(3.0 / 1.46) * iota * np.sqrt(epsilon_t) / 2.0
    fac_dkes_to_d33star = np.ones_like(rho_arr)

    return VmecReferenceFactors(
        rho=jnp.asarray(rho_arr),
        a_b=a_b,
        psia=psia,
        b00=jnp.asarray(b00),
        r00=jnp.asarray(r00),
        iota=jnp.asarray(iota),
        boozer_i=jnp.asarray(boozer_i),
        boozer_g=jnp.asarray(boozer_g),
        drds=jnp.asarray(drds),
        dr_tildedr=jnp.asarray(dr_tildedr),
        dr_tildeds=jnp.asarray(dr_tildeds),
        fac_reference_to_sfincs_11=jnp.asarray(fac_reference_to_sfincs_11),
        fac_reference_to_sfincs_31=jnp.asarray(fac_reference_to_sfincs_31),
        fac_reference_to_sfincs_33=jnp.asarray(fac_reference_to_sfincs_33),
        fac_sfincs_to_dkes_11=jnp.asarray(fac_sfincs_to_dkes_11),
        fac_sfincs_to_dkes_31=jnp.asarray(fac_sfincs_to_dkes_31),
        fac_sfincs_to_dkes_33=jnp.asarray(fac_sfincs_to_dkes_33),
        fac_dkes_to_d11star=jnp.asarray(fac_dkes_to_d11star),
        fac_dkes_to_d31star=jnp.asarray(fac_dkes_to_d31star),
        fac_dkes_to_d33star=jnp.asarray(fac_dkes_to_d33star),
    )


def _read_vmec_wout_netcdf(path: Path) -> _VmecWoutData:
    try:
        from netCDF4 import Dataset
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "comparison VMEC helpers require netCDF4. Install it with "
            "`pip install netCDF4` or `pip install -e '.[io]'`."
        ) from exc

    with Dataset(path, mode="r") as handle:
        return {
            "ns": int(_filled(handle.variables["ns"])),
            "nfp": int(_filled(handle.variables["nfp"])),
            "mpol": int(_filled(handle.variables["mpol"])),
            "ntor": int(_filled(handle.variables["ntor"])),
            "Aminor_p": float(_filled(handle.variables["Aminor_p"])),
            "volume_p": float(_filled(handle.variables["volume_p"])),
            "phi": np.asarray(_filled(handle.variables["phi"]), dtype=np.float64),
            "iotaf": np.asarray(_filled(handle.variables["iotaf"]), dtype=np.float64),
            "xm_nyq": np.asarray(_filled(handle.variables["xm_nyq"]), dtype=np.int32),
            "xn_nyq": np.asarray(_filled(handle.variables["xn_nyq"]), dtype=np.int32),
            "gmnc": np.asarray(_filled(handle.variables["gmnc"]), dtype=np.float64),
            "bmnc": np.asarray(_filled(handle.variables["bmnc"]), dtype=np.float64),
            "bsupumnc": np.asarray(_filled(handle.variables["bsupumnc"]), dtype=np.float64),
            "bsupvmnc": np.asarray(_filled(handle.variables["bsupvmnc"]), dtype=np.float64),
            "bsubumnc": np.asarray(_filled(handle.variables["bsubumnc"]), dtype=np.float64),
            "bsubvmnc": np.asarray(_filled(handle.variables["bsubvmnc"]), dtype=np.float64),
        }


def _read_booz_netcdf(path: Path) -> _BoozData:
    try:
        from netCDF4 import Dataset
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "comparison Boozer helpers require netCDF4. Install it with "
            "`pip install netCDF4` or `pip install -e '.[io]'`."
        ) from exc

    with Dataset(path, mode="r") as handle:
        return {
            "bmnc_b": np.asarray(_filled(handle.variables["bmnc_b"]), dtype=np.float64),
            "rmnc_b": np.asarray(_filled(handle.variables["rmnc_b"]), dtype=np.float64),
            "buco_b": np.asarray(_filled(handle.variables["buco_b"]), dtype=np.float64),
            "bvco_b": np.asarray(_filled(handle.variables["bvco_b"]), dtype=np.float64),
            "ixm_b": np.asarray(_filled(handle.variables["ixm_b"]), dtype=np.int32),
            "ixn_b": np.asarray(_filled(handle.variables["ixn_b"]), dtype=np.int32),
            "jlist": np.asarray(_filled(handle.variables["jlist"]), dtype=np.int32),
        }


def _filled(variable) -> np.ndarray | float | int:
    values = variable[:]
    return values.filled() if hasattr(values, "filled") else values


def _interp_mode_columns(x: np.ndarray, values: np.ndarray, xq: float) -> np.ndarray:
    if values.ndim != 2:
        raise ValueError("expected a 2D `(radius, mode)` array")
    return np.asarray(_interp_profile(x, values, xq), dtype=np.float64)


def _interp_profile(x: np.ndarray, values: np.ndarray, xq):
    """Match the cubic interpolation used in the reference database workflow."""

    import interpax

    return interpax.interp1d(
        jnp.asarray(xq, dtype=jnp.float64),
        jnp.asarray(x, dtype=jnp.float64),
        jnp.asarray(values, dtype=jnp.float64),
        method="cubic",
        extrap=True,
    )


def _mode_index(m: np.ndarray, n: np.ndarray, m_target: int, n_target: int) -> int:
    matches = np.where((m == m_target) & (n == n_target))[0]
    if matches.size == 0:
        raise ValueError(f"mode ({m_target}, {n_target}) not found in Boozer file")
    return int(matches[0])
