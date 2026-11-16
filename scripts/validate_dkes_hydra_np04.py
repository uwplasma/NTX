#!/usr/bin/env python3
"""NTX against the IPP DKES coefficients of the HELIAS reactor hydra_Np04_20190108.

The DKES results (``DKES_rho_<rho>.h5``) and the Geiger Boozer file cannot be redistributed.
Point ``--dkes-dir`` (or ``NTX_DKES_HYDRA_DIR``) and ``--boozer`` (or ``NTX_HYDRA_BOOZER``) at them;
the committed record ``benchmarks/results/dkes_hydra_np04.json`` holds agreement statistics only.

Normalization (the mapping fixed by the DKX comparison, uwplasma/DKX#395): ``nu_hat = cmul`` (DKES
``CMUL = nu_D/v``, 1/m), ``epsi_hat = efield / r`` with ``r = sqrt(2 psi / B0)``
(``EStar = efield |G| / (B0 |iota| r)``), ``D33* = D33 / D33_spitzer`` and, with ``R = |G| / B0``,

    D11* = (8 / pi) B0^2 R |iota| D11,       D31* = 3 / (2 * 1.46) B0 |iota| sqrt(r / R) D31,

the Beidler et al. (2011) plateau and banana normalizations of NTX's psi_p-scaled coefficients.
The DKES ``_S`` datasets (SFINCS ``transportMatrix`` entries) are converted with the same closed
form DKX uses (``nuPrime = cmul |G + iota I| / (B0 nuD(1))``, ``nuD(1) = 0.836028``).

    python scripts/validate_dkes_hydra_np04.py --dkes-dir DIR --boozer FILE --out record.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

DELTA = 0.0045694  # SFINCS rho* of the reference normalization (1 keV protons, 1 T, 1 m)
NU_D_ONE = 0.836028  # nu_D(x = 1)
TRAPPED = 1.46  # 1 - f_c = 1.46 sqrt(eps_t)
FIRST_SURFACE_RHO = math.sqrt(0.015) + 1e-7  # rho = 0.12247 is the first file surface
GATES = {"cmul_min": 1e-3, "d11_rel": 0.25, "d33_rel": 0.01, "d31_abs": 0.003}
REGIMES = (("collisional", 0.1, 10.0), ("plateau", 1e-3, 3e-2), ("low_nu", 0.0, 3e-4))
GRID = (17, 41, 96)
EFIELDS = (0.0, 0.01, 0.03)  # the radial fields of the DKX comparison; the files hold more


def reference_rows(directory: Path | str) -> list[dict]:
    """The DKES points of ``DKES_rho_<rho>.h5``.

    ``cmul = nuprime / nuprimefac`` and ``|efield| = |Erstar / Estarfac|``.
    """
    import h5py

    rows = []
    for path in sorted(Path(directory).glob("DKES_rho_*.h5")):
        with h5py.File(path, "r") as f:
            cmul = np.asarray(f["nuprime"]) / float(f["nuprimefac"][()])
            efield = np.abs(np.asarray(f["Erstar"]) / float(f["Estarfac"][()]))
            data = {k: np.asarray(f[k]) for k in ("D11_S", "D13_S", "D31_S", "D33_S")}
        rho = path.stem.removeprefix("DKES_rho_")
        for i in range(len(cmul)):
            if not any(abs(efield[i] - e) <= 1e-9 + 1e-3 * e for e in EFIELDS):
                continue
            rows.append(
                {
                    "rho": rho,
                    "cmul": float(f"{cmul[i]:.6g}"),
                    "efield": float(f"{efield[i]:.6g}"),
                    **{k: float(v[i]) for k, v in data.items()},
                }
            )
    return rows


def surface_constants(surface, b2_mean: float) -> dict[str, float]:
    """DKX-style flux-surface constants of an NTX Boozer surface."""
    g, i_hat, iota, b0 = (
        float(surface.b_zeta),
        float(surface.b_theta),
        float(surface.iota),
        float(surface.b0),
    )
    return {
        "g": g,
        "I": i_hat,
        "iota": iota,
        "b0": b0,
        "r": float(surface.psi_p) / b0,
        "fsab2": b2_mean,
    }


def dkes_to_star(row: dict, c: dict[str, float]) -> dict[str, float]:
    """Beidler ``D*`` of a DKES row (``_S`` datasets), unit charge and mass, ``x0 = 1``."""
    b0, r = abs(c["b0"]), c["r"]
    g_abs, iota = abs(c["g"]), abs(c["iota"])
    g_plus = abs(c["g"] + c["iota"] * c["I"])
    r_major = g_abs / b0
    eps_t = r / r_major
    qfac = math.sqrt(math.pi) / 4.0  # sqrt(pi) / (4 w0 x0^2 exp(-x0^2)), w0 = e, x0 = 1
    dr_dpsi = 1.0 / (b0 * r)
    c11 = -(DELTA**2 / 4.0) * dr_dpsi**2 * g_abs**2 / (b0 * g_plus)
    c13 = 0.5 * DELTA * (g_abs / b0) * dr_dpsi
    c33 = g_plus / b0
    sign = -1.0 if c["iota"] < 0.0 else 1.0
    v_d = DELTA / (2.0 * r_major * b0)
    d11_p = (math.pi / 4.0) * v_d**2 * r_major / iota
    d31_b = (2.0 / 3.0) * (v_d * r_major / (iota * eps_t)) * TRAPPED * math.sqrt(eps_t)
    d33_ps = c["fsab2"] / (3.0 * row["cmul"] * b0 * b0)
    return {
        "D11_star": qfac * c11 * row["D11_S"] / d11_p,
        "D31_star": -sign * qfac * c13 * row["D31_S"] / d31_b,
        "D33_star": qfac * c33 * row["D33_S"] / d33_ps,
    }


def ntx_star(result, c: dict[str, float]) -> dict[str, float]:
    """Beidler ``D*`` of an NTX ``TransportResult`` (``psi_p``-scaled coefficients)."""
    b0, iota = abs(c["b0"]), abs(c["iota"])
    r_major = abs(c["g"]) / b0
    return {
        "D11_star": 8.0 / math.pi * b0**2 * r_major * iota * float(result.D11),
        "D31_star": 3.0
        / (2.0 * TRAPPED)
        * b0
        * iota
        * math.sqrt(c["r"] / r_major)
        * float(result.D31),
        "D33_star": float(result.D33) / float(result.D33_spitzer),
    }


def solve_columns(boozer: Path, rho: str, rows: list[dict], grid=GRID) -> list[dict]:
    """NTX ``D*`` at every DKES point of one radius, sharing the prepared system."""
    import jax

    jax.config.update("jax_enable_x64", True)
    import ntx

    surface = ntx.load_geiger_boozer_surface(boozer, rho=max(float(rho), FIRST_SURFACE_RHO))
    prepared = ntx.prepare_monoenergetic_system(surface, ntx.GridSpec(*grid))
    c = surface_constants(surface, float(prepared.geometry.b2_mean))
    out = []
    for row in rows:
        result = ntx.solve_prepared(
            prepared, ntx.MonoenergeticCase(nu_hat=row["cmul"], epsi_hat=row["efield"] / c["r"])
        )
        out.append({**row, "ntx": ntx_star(result, c), "dkes": dkes_to_star(row, c)})
    return out


def agreement(points: list[dict]) -> list[dict]:
    """Per (rho, E class, regime) statistics of NTX against DKES, rounded to 0.1% (D31*: 1e-3)."""
    out = []
    for rho in dict.fromkeys(p["rho"] for p in points):
        for e_class in ("zero", "finite"):
            for name, lo, hi in REGIMES:
                sel = [
                    p
                    for p in points
                    if p["rho"] == rho
                    and (p["efield"] == 0.0) == (e_class == "zero")
                    and lo <= p["cmul"] <= hi
                ]
                if not sel:
                    continue
                d11 = [abs(p["ntx"]["D11_star"] / p["dkes"]["D11_star"] - 1.0) for p in sel]
                d33 = [abs(p["ntx"]["D33_star"] / p["dkes"]["D33_star"] - 1.0) for p in sel]
                d31 = [abs(p["ntx"]["D31_star"] - p["dkes"]["D31_star"]) for p in sel]
                out.append(
                    {
                        "rho": rho,
                        "efield": e_class,
                        "regime": name,
                        "points": len(sel),
                        "d11_median_pct": round(100 * float(np.median(d11)), 1),
                        "d11_max_pct": round(100 * max(d11), 1),
                        "d33_max_pct": round(100 * max(d33), 1),
                        "d31_max_abs": None if e_class == "zero" else round(max(d31), 3),
                    }
                )
    return out


EXCLUDED = [
    {
        "rho": "0.875",
        "efield": "zero",
        "coefficient": "D11_star",
        "reason": (
            "the stored D11_S at efield = 0 is inconsistent with the .dk output of the same run "
            "(see the DKX record dkes_hydra_np04); reported to the file's owner"
        ),
    }
]
NOTES = [
    "grid 17 x 41 x 96 (n_theta x n_zeta per period x n_xi), all harmonics of the file at or "
    "below the grid Nyquist, the grid of the DKX comparison",
    "cmul <= 3e-4 is reported, not gated: the grid is not converged there "
    "(the DKX comparison finds the same)",
    "efield = 0, 0.01 and 0.03 only: the files carry more fields, DKX ran these three",
    "rho = 0.5 at efield = 0 has no DKES point in the files",
    "rho = 0.12247 is run at the first surface of the Boozer file plus 1e-7 in rho",
]


def build_record(stats: list[dict], grid) -> dict:
    return {
        "schema": "ntx.dkes_hydra_np04.v1",
        "grid": list(grid),
        "provenance": (
            "DKES, IPP, hydra_Np04_20190108; Boozer data J. Geiger, IPP; neither is "
            "redistributable, only agreement statistics are stored"
        ),
        "gates": GATES,
        "excluded": EXCLUDED,
        "notes": NOTES,
        "agreement": stats,
    }


def gate_errors(stats: list[dict], excluded: set[tuple[str, str]] = frozenset()) -> list[str]:
    """Gate violations of the gated regimes (``cmul >= 1e-3``)."""
    errors = []
    for s in stats:
        if s["regime"] == "low_nu":
            continue
        where = f"{s['rho']} {s['efield']} {s['regime']}"
        if (s["rho"], s["efield"]) not in excluded and s["d11_max_pct"] / 100 > GATES["d11_rel"]:
            errors.append(f"{where}: D11* differs by {s['d11_max_pct']}%")
        if s["d33_max_pct"] / 100 > GATES["d33_rel"]:
            errors.append(f"{where}: D33* differs by {s['d33_max_pct']}%")
        if s["d31_max_abs"] is not None and s["d31_max_abs"] > GATES["d31_abs"]:
            errors.append(f"{where}: D31* differs by {s['d31_max_abs']}")
    return errors


def compare(boozer: Path, dkes_dir: Path, rhos=None, grid=GRID) -> list[dict]:
    reference = reference_rows(dkes_dir)
    points: list[dict] = []
    for rho in dict.fromkeys(r["rho"] for r in reference):
        if rhos and rho not in rhos:
            continue
        points += solve_columns(boozer, rho, [r for r in reference if r["rho"] == rho], grid)
    return points


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dkes-dir", type=Path, default=os.environ.get("NTX_DKES_HYDRA_DIR"))
    parser.add_argument("--boozer", type=Path, default=os.environ.get("NTX_HYDRA_BOOZER"))
    parser.add_argument("--rho", nargs="*")
    parser.add_argument("--grid", type=int, nargs=3, default=GRID)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if not (args.dkes_dir and args.boozer):
        parser.error(
            "--dkes-dir and --boozer (or NTX_DKES_HYDRA_DIR and NTX_HYDRA_BOOZER) are required"
        )
    stats = agreement(compare(Path(args.boozer), Path(args.dkes_dir), args.rho, tuple(args.grid)))
    record = build_record(stats, args.grid)
    text = json.dumps(record, indent=1) + "\n"
    if args.out:
        args.out.write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
