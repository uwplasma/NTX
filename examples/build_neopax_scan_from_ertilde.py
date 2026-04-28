#!/usr/bin/env python3
"""Build a NEOPAX-style monoenergetic scan from an Er_tilde grid.

This mirrors the legacy MONKES DKES-like database workflow more closely than
the NTX rebuild/audit examples: the user chooses rho, nu_v, and Er_tilde,
provides VMEC + Boozer files, and NTX computes the coefficient tables and
conversion metadata from scratch before writing a NEOPAX-style HDF5 file.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import interpax
import jax.numpy as jnp
import numpy as np
from netCDF4 import Dataset

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ntx import GridSpec, NeopaxScan, solve_monoenergetic_scan, write_neopax_scan_hdf5  # noqa: E402
from ntx._checkout_paths import find_neopax_root  # noqa: E402
from ntx.vmec_jax_vmec import surface_from_vmec_jax_vmec_wout_file  # noqa: E402


NEOPAX_ROOT = find_neopax_root()

# ---------------------------------------------------------------------------
# User inputs
# ---------------------------------------------------------------------------
WOUT_PATH = (
    NEOPAX_ROOT / "tests" / "inputs" / "wout_W7-X_standard_configuration.nc"
    if NEOPAX_ROOT is not None
    else Path("/missing/wout_W7-X_standard_configuration.nc")
)
BOOZMN_PATH = (
    NEOPAX_ROOT / "tests" / "inputs" / "boozmn_wout_W7-X_standard_configuration.nc"
    if NEOPAX_ROOT is not None
    else Path("/missing/boozmn_wout_W7-X_standard_configuration.nc")
)
OUTPUT_PATH = ROOT / "examples" / "outputs" / "neopax_scan_from_ertilde" / "ntx_scan_from_ertilde.h5"

RHO = jnp.array([0.12247, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875], dtype=jnp.float64)
NU_V = jnp.array(
    [
        3.0e-7,
        1.0e-6,
        3.0e-6,
        1.0e-5,
        3.0e-5,
        1.0e-4,
        3.0e-4,
        1.0e-3,
        3.0e-3,
        1.0e-2,
        3.0e-2,
        1.0e-1,
        3.0e-1,
        1.0e0,
        3.0e0,
        1.0e1,
    ],
    dtype=jnp.float64,
)
ER_TILDE = jnp.array(
    [0.0, 1.0e-6, 3.0e-6, 1.0e-5, 3.0e-5, 1.0e-4, 3.0e-4, 1.0e-3, 3.0e-3, 1.0e-2, 3.0e-2, 1.0e-1],
    dtype=jnp.float64,
)
GRID = GridSpec(n_theta=25, n_zeta=25, n_xi=64)
ONSAGER_WARN_THRESHOLD = 1.0e-6


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a NEOPAX-style NTX monoenergetic scan from an Er_tilde grid."
    )
    parser.add_argument(
        "--wout",
        type=Path,
        default=WOUT_PATH,
        help="Path to the VMEC wout file.",
    )
    parser.add_argument(
        "--booz",
        type=Path,
        default=BOOZMN_PATH,
        help="Path to the Boozer boozmn/boozermn file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_PATH,
        help="Path to the output NEOPAX-style HDF5 file.",
    )
    parser.add_argument(
        "--n-theta",
        type=int,
        default=GRID.n_theta,
        help="Poloidal grid resolution.",
    )
    parser.add_argument(
        "--n-zeta",
        type=int,
        default=GRID.n_zeta,
        help="Toroidal grid resolution.",
    )
    parser.add_argument(
        "--n-xi",
        type=int,
        default=GRID.n_xi,
        help="Pitch-angle / Legendre resolution.",
    )
    return parser.parse_args()


def _filled(variable) -> np.ndarray:
    values = variable[:]
    if hasattr(values, "filled"):
        values = values.filled()
    return np.asarray(values, dtype=float)


def _interpolator(x, y):
    return interpax.Interpolator1D(jnp.asarray(x, dtype=jnp.float64), jnp.asarray(y, dtype=jnp.float64), extrap=True)


def _load_vmec_boozer_channels(wout_path: Path, boozmn_path: Path, rho: jnp.ndarray) -> dict[str, jnp.ndarray | float]:
    with Dataset(wout_path, mode="r") as vfile:
        ns = int(np.asarray(vfile.variables["ns"][:]).reshape(-1)[0])
        s_full = jnp.linspace(0.0, 1.0, ns)
        s_half = jnp.asarray([(i - 0.5) / (ns - 1) for i in range(ns)], dtype=jnp.float64)
        rho_half = jnp.sqrt(s_half)
        rho_full = jnp.sqrt(s_full)

        volume_p = float(np.asarray(vfile.variables["volume_p"][:]).reshape(-1)[-1])
        vp = _filled(vfile.variables["vp"])
        phi = _filled(vfile.variables["phi"])
        iotaf = _filled(vfile.variables["iotaf"])
        psia = float(jnp.abs(phi[-1]) / (2.0 * jnp.pi))

    with Dataset(boozmn_path, mode="r") as bfile:
        bmnc_b = _filled(bfile.variables["bmnc_b"])
        rmnc_b = _filled(bfile.variables["rmnc_b"])
        xm_b = _filled(bfile.variables["ixm_b"])
        xn_b = _filled(bfile.variables["ixn_b"])
        buco = _filled(bfile.variables["buco_b"])
        bvco = _filled(bfile.variables["bvco_b"])

    zero_mode = np.where((xm_b == 0) & (xn_b == 0))[0]
    if zero_mode.size == 0:
        raise ValueError("could not find Boozer (m,n)=(0,0) mode in boozmn file")
    mode00 = int(zero_mode[0])

    r0_b = float(rmnc_b[-1, mode00])
    a_b = float(np.sqrt(volume_p / (2.0 * np.pi**2 * r0_b)))

    b00 = _interpolator(rho_half[1:], bmnc_b[:, mode00])
    r00 = _interpolator(rho_full[1:], rmnc_b[:, mode00])
    boozer_i = _interpolator(rho_half[1:], buco[1:])
    boozer_g = _interpolator(rho_half[1:], bvco[1:])
    iota = _interpolator(rho_full, iotaf)

    b00_rho = b00(rho)
    r00_rho = r00(rho)
    i_rho = boozer_i(rho)
    g_rho = boozer_g(rho)
    iota_rho = iota(rho)

    dpsidrtilde = rho * a_b * b00_rho
    drds = a_b / (2.0 * rho)
    dr_tildedr = 2.0 * psia / (a_b**2 * b00_rho)
    dr_tildeds = dr_tildedr * drds

    fac_monkes_to_sfincs_11 = 8.0 * (g_rho + iota_rho * i_rho) * b00_rho * psia**2 / (jnp.sqrt(jnp.pi) * g_rho**2)
    fac_monkes_to_sfincs_31 = 4.0 * b00_rho * psia / (jnp.sqrt(jnp.pi) * g_rho)
    fac_monkes_to_sfincs_33 = -2.0 * b00_rho / ((g_rho + iota_rho * i_rho) * jnp.sqrt(jnp.pi))

    fac_sfincs_to_dkes_11 = 1.0 / (
        8.0 * (g_rho + iota_rho * i_rho) * dpsidrtilde**2 / (g_rho**2 * b00_rho * jnp.sqrt(jnp.pi))
    )
    fac_sfincs_to_dkes_31 = 1.0 / (4.0 * dpsidrtilde / (g_rho * jnp.sqrt(jnp.pi)))
    fac_sfincs_to_dkes_33 = 1.0 / (-2.0 * b00_rho / ((g_rho + iota_rho * i_rho) * jnp.sqrt(jnp.pi)))

    epsilon_t = rho * a_b / r00_rho
    fac_dkes_to_d11star = -(8.0 / jnp.pi) * iota_rho * r00_rho
    fac_dkes_to_d31star = -(3.0 / 1.46) * iota_rho * jnp.sqrt(epsilon_t) / 2.0
    fac_dkes_to_d33star = jnp.asarray(1.0, dtype=jnp.float64)

    return {
        "a_b": a_b,
        "psia": psia,
        "b00": b00_rho,
        "r00": r00_rho,
        "boozer_i": i_rho,
        "boozer_g": g_rho,
        "iota": iota_rho,
        "drds": drds,
        "dr_tildedr": dr_tildedr,
        "dr_tildeds": dr_tildeds,
        "fac_monkes_to_sfincs_11": fac_monkes_to_sfincs_11,
        "fac_monkes_to_sfincs_31": fac_monkes_to_sfincs_31,
        "fac_monkes_to_sfincs_33": fac_monkes_to_sfincs_33,
        "fac_sfincs_to_dkes_11": fac_sfincs_to_dkes_11,
        "fac_sfincs_to_dkes_31": fac_sfincs_to_dkes_31,
        "fac_sfincs_to_dkes_33": fac_sfincs_to_dkes_33,
        "fac_dkes_to_d11star": fac_dkes_to_d11star,
        "fac_dkes_to_d31star": fac_dkes_to_d31star,
        "fac_dkes_to_d33star": fac_dkes_to_d33star,
    }


def _build_field_channels(
    rho: jnp.ndarray,
    er_tilde: jnp.ndarray,
    b00: jnp.ndarray,
    dr_tildedr: jnp.ndarray,
    dr_tildeds: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    er = er_tilde[None, :] * dr_tildedr[:, None] * b00[:, None]
    es = er_tilde[None, :] * dr_tildeds[:, None] * b00[:, None]
    er_to_ertilde = jnp.broadcast_to(1.0 / dr_tildedr[:, None], er.shape)
    return er, es, er_to_ertilde


def _surface_loader(wout_path: Path, rho_value: float):
    return surface_from_vmec_jax_vmec_wout_file(wout_path, s=float(rho_value**2))


def _report_scan_warnings(scan: NeopaxScan) -> None:
    d11 = np.asarray(scan.D11, dtype=float)
    d13 = np.asarray(scan.D13, dtype=float)
    d31 = np.asarray(scan.D31, dtype=float) if scan.D31 is not None else None
    d33 = np.asarray(scan.D33, dtype=float)

    any_issue = False

    finite_mask = np.isfinite(d11) & np.isfinite(d13) & np.isfinite(d33)
    if d31 is not None:
        finite_mask = finite_mask & np.isfinite(d31)
    if not bool(np.all(finite_mask)):
        bad = int(np.size(finite_mask) - np.count_nonzero(finite_mask))
        print(f"warning: found {bad} non-finite coefficient entries in the scan")
        any_issue = True

    negative_d11 = np.argwhere(d11 < 0.0)
    if negative_d11.size > 0:
        print(f"warning: found {negative_d11.shape[0]} entries with D11 < 0")
        for idx in negative_d11[:10]:
            ir, inu, ier = (int(v) for v in idx)
            print(
                "  "
                f"rho={float(scan.rho[ir]):.5f}, "
                f"nu_v={float(scan.nu_v[inu]):.6e}, "
                f"Er_tilde={float(scan.Er_tilde[ier]):.6e}, "
                f"D11={d11[ir, inu, ier]:.6e}"
            )
        if negative_d11.shape[0] > 10:
            print(f"  ... plus {negative_d11.shape[0] - 10} more negative-D11 entries")
        any_issue = True

    if d31 is not None:
        onsager = np.abs(d13 + d31)
        max_onsager = float(np.max(onsager))
        if max_onsager > ONSAGER_WARN_THRESHOLD:
            worst = np.unravel_index(int(np.argmax(onsager)), onsager.shape)
            ir, inu, ier = (int(v) for v in worst)
            print(
                "warning: Onsager mismatch exceeded threshold "
                f"({ONSAGER_WARN_THRESHOLD:.1e}); max |D13 + D31| = {max_onsager:.6e}"
            )
            print(
                "  "
                f"rho={float(scan.rho[ir]):.5f}, "
                f"nu_v={float(scan.nu_v[inu]):.6e}, "
                f"Er_tilde={float(scan.Er_tilde[ier]):.6e}, "
                f"D13={d13[ir, inu, ier]:.6e}, "
                f"D31={d31[ir, inu, ier]:.6e}"
            )
            any_issue = True

    if not any_issue:
        print("scan sanity checks: no negative D11, no non-finite values, Onsager mismatch within threshold")


def build_scan(*, wout_path: Path, boozmn_path: Path, grid: GridSpec) -> NeopaxScan:
    channels = _load_vmec_boozer_channels(wout_path, boozmn_path, RHO)
    er, es, er_to_ertilde = _build_field_channels(
        RHO,
        ER_TILDE,
        channels["b00"],
        channels["dr_tildedr"],
        channels["dr_tildeds"],
    )

    n_r = int(RHO.shape[0])
    n_nu = int(NU_V.shape[0])
    n_er = int(ER_TILDE.shape[0])
    d11 = jnp.zeros((n_r, n_nu, n_er), dtype=jnp.float64)
    d13 = jnp.zeros((n_r, n_nu, n_er), dtype=jnp.float64)
    d31 = jnp.zeros((n_r, n_nu, n_er), dtype=jnp.float64)
    d33 = jnp.zeros((n_r, n_nu, n_er), dtype=jnp.float64)
    d33_spitzer = jnp.zeros((n_r, n_nu, n_er), dtype=jnp.float64)

    for idx, rho_value in enumerate(np.asarray(RHO)):
        surface = _surface_loader(wout_path, float(rho_value))
        nu_grid, es_grid = jnp.meshgrid(NU_V, es[idx], indexing="ij")
        coeffs = solve_monoenergetic_scan(surface, grid, nu_grid, epsi_hat=es_grid)
        d11 = d11.at[idx].set(coeffs["D11"])
        d13 = d13.at[idx].set(coeffs["D13"])
        d31 = d31.at[idx].set(coeffs["D31"])
        d33 = d33.at[idx].set(coeffs["D33"])
        d33_spitzer = d33_spitzer.at[idx].set(coeffs["D33_spitzer"])

    return NeopaxScan(
        rho=RHO,
        nu_v=NU_V,
        Er=er,
        Es=es,
        drds=channels["drds"],
        D11=d11,
        D13=d13,
        D33=d33,
        D33_spitzer=d33_spitzer,
        D31=d31,
        Er_tilde=ER_TILDE,
        Er_to_Ertilde=er_to_ertilde,
        dr_tildedr=channels["dr_tildedr"],
        dr_tildeds=channels["dr_tildeds"],
        a_b=channels["a_b"],
        psia=channels["psia"],
        b00=channels["b00"],
        r00=channels["r00"],
        boozer_i=channels["boozer_i"],
        boozer_g=channels["boozer_g"],
        iota=channels["iota"],
        fac_reference_to_sfincs_11=channels["fac_monkes_to_sfincs_11"],
        fac_reference_to_sfincs_31=channels["fac_monkes_to_sfincs_31"],
        fac_reference_to_sfincs_33=channels["fac_monkes_to_sfincs_33"],
        fac_monkes_to_sfincs_11=channels["fac_monkes_to_sfincs_11"],
        fac_monkes_to_sfincs_31=channels["fac_monkes_to_sfincs_31"],
        fac_monkes_to_sfincs_33=channels["fac_monkes_to_sfincs_33"],
        fac_sfincs_to_dkes_11=channels["fac_sfincs_to_dkes_11"],
        fac_sfincs_to_dkes_31=channels["fac_sfincs_to_dkes_31"],
        fac_sfincs_to_dkes_33=channels["fac_sfincs_to_dkes_33"],
        fac_dkes_to_d11star=channels["fac_dkes_to_d11star"],
        fac_dkes_to_d31star=channels["fac_dkes_to_d31star"],
        fac_dkes_to_d33star=channels["fac_dkes_to_d33star"],
        source_name="ntx_scan_from_ertilde",
    )


def main() -> None:
    args = _parse_args()
    wout_path = args.wout.expanduser().resolve()
    boozmn_path = args.booz.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    grid = GridSpec(n_theta=args.n_theta, n_zeta=args.n_zeta, n_xi=args.n_xi)

    scan = build_scan(wout_path=wout_path, boozmn_path=boozmn_path, grid=grid)
    _report_scan_warnings(scan)
    output = write_neopax_scan_hdf5(scan, output_path)
    print(f"wrote NEOPAX-style scan to: {output}")
    print(f"wout: {wout_path}")
    print(f"booz: {boozmn_path}")
    print(f"grid: n_theta={grid.n_theta}, n_zeta={grid.n_zeta}, n_xi={grid.n_xi}")
    print(f"rho points: {scan.rho.shape[0]}")
    print(f"nu_v points: {scan.nu_v.shape[0]}")
    print(f"Er_tilde points: {scan.Er_tilde.shape[0] if scan.Er_tilde is not None else 0}")
    print(f"D11 shape: {scan.D11.shape}")
    print(f"D31 shape: {scan.D31.shape if scan.D31 is not None else None}")


if __name__ == "__main__":
    main()
