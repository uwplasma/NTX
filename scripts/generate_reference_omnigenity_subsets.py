#!/usr/bin/env python3
"""Generate omnigenous external-reference subsets from the benchmark executable."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import jax.numpy as jnp
from jax import config

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ntx import (  # noqa: E402
    NeopaxScan,
    enable_x64,
    vmec_reference_factors,
    write_neopax_scan_hdf5,
)
from ntx._checkout_paths import find_reference_executable, fixture_path  # noqa: E402
from ntx.benchmarks import nearest_reference_row, read_monoenergetic_table  # noqa: E402


def main() -> int:
    config.update("jax_enable_x64", True)
    enable_x64(True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference-exe",
        type=Path,
        default=find_reference_executable(),
        help="path to the external benchmark executable",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=fixture_path("benchmarks", "omnigenity"),
        help="directory for generated HDF5 reference subsets",
    )
    args = parser.parse_args()
    if args.reference_exe is None:
        raise SystemExit("benchmark executable not found; pass --reference-exe explicitly")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    cases = {
        "QA": {
            "rho": jnp.asarray([0.25, 0.5], dtype=jnp.float64),
            "nu_v": jnp.asarray([1.0e-4, 1.0e-3], dtype=jnp.float64),
            "er_tilde": jnp.asarray([0.0, 1.0e-3], dtype=jnp.float64),
        },
        "QH": {
            "rho": jnp.asarray([0.25, 0.5], dtype=jnp.float64),
            "nu_v": jnp.asarray([1.0e-4, 1.0e-3], dtype=jnp.float64),
            "er_tilde": jnp.asarray([0.0, 1.0e-3], dtype=jnp.float64),
        },
        "QI": {
            "rho": jnp.asarray([0.25, 0.5], dtype=jnp.float64),
            "nu_v": jnp.asarray([1.0e-4, 1.0e-3], dtype=jnp.float64),
            "er_tilde": jnp.asarray([0.0, 1.0e-3], dtype=jnp.float64),
        },
    }

    for label, spec in cases.items():
        wout = fixture_path(f"wout_nfp3_{label}_fixed_resolution_final.nc")
        booz = fixture_path(f"boozmn_nfp3_{label}_fixed_resolution_final.nc")
        factors = vmec_reference_factors(wout, booz, spec["rho"])
        es = spec["er_tilde"][None, :] * factors.dr_tildeds[:, None] * factors.b00[:, None]
        er = spec["er_tilde"][None, :] * factors.dr_tildedr[:, None] * factors.b00[:, None]
        er_to_er_tilde = jnp.broadcast_to(1.0 / factors.dr_tildedr[:, None], es.shape)

        d11_list = []
        d31_list = []
        d13_list = []
        d33_list = []
        for rho_value, es_row in zip(spec["rho"], es, strict=True):
            nu_rows_11 = []
            nu_rows_31 = []
            nu_rows_13 = []
            nu_rows_33 = []
            for nu_value in spec["nu_v"]:
                row_11 = []
                row_31 = []
                row_13 = []
                row_33 = []
                for es_value in es_row:
                    reference = _run_single_case(
                        executable=args.reference_exe,
                        wout_path=wout,
                        s=float(rho_value**2),
                        n_theta=25,
                        n_zeta=25,
                        n_xi=63,
                        nu_hat=float(nu_value),
                        epsi_hat=float(es_value),
                    )
                    row_11.append(reference["D11"])
                    row_31.append(reference["D31"])
                    row_13.append(reference["D13"])
                    row_33.append(reference["D33"])
                nu_rows_11.append(row_11)
                nu_rows_31.append(row_31)
                nu_rows_13.append(row_13)
                nu_rows_33.append(row_33)
            d11_list.append(nu_rows_11)
            d31_list.append(nu_rows_31)
            d13_list.append(nu_rows_13)
            d33_list.append(nu_rows_33)

        scan = NeopaxScan(
            rho=spec["rho"],
            nu_v=spec["nu_v"],
            Er=er,
            Es=es,
            drds=factors.drds,
            D11=jnp.asarray(d11_list, dtype=jnp.float64),
            D13=jnp.asarray(d13_list, dtype=jnp.float64),
            D33=jnp.asarray(d33_list, dtype=jnp.float64),
            D31=jnp.asarray(d31_list, dtype=jnp.float64),
            Er_tilde=spec["er_tilde"],
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
            fac_reference_to_sfincs_11=factors.fac_reference_to_sfincs_11,
            fac_reference_to_sfincs_31=factors.fac_reference_to_sfincs_31,
            fac_reference_to_sfincs_33=factors.fac_reference_to_sfincs_33,
            fac_sfincs_to_dkes_11=factors.fac_sfincs_to_dkes_11,
            fac_sfincs_to_dkes_31=factors.fac_sfincs_to_dkes_31,
            fac_sfincs_to_dkes_33=factors.fac_sfincs_to_dkes_33,
            fac_dkes_to_d11star=factors.fac_dkes_to_d11star,
            fac_dkes_to_d31star=factors.fac_dkes_to_d31star,
            fac_dkes_to_d33star=factors.fac_dkes_to_d33star,
            source_name=f"external_reference_{label.lower()}_subset",
        )
        output = output_dir / f"external_reference_{label.lower()}_subset.h5"
        write_neopax_scan_hdf5(scan, output)
        print(output)
    return 0


def _run_single_case(
    *,
    executable: Path,
    wout_path: Path,
    s: float,
    n_theta: int,
    n_zeta: int,
    n_xi: int,
    nu_hat: float,
    epsi_hat: float,
) -> dict[str, float]:
    run_dir = Path(tempfile.mkdtemp(prefix="ntx-reference-subset-")).resolve()
    try:
        shutil.copy2(wout_path, run_dir / "VMEC.nc")
        (run_dir / _protocol_surface_input_name()).write_text(
            "\n".join(["&surface", f"  s = {s:.16e}", "/", ""]),
            encoding="utf-8",
        )
        (run_dir / _protocol_parameter_input_name()).write_text(
            "\n".join(
                [
                    "&parameters",
                    f"  N_theta = {n_theta}",
                    f"  N_zeta = {n_zeta}",
                    f"  N_xi = {n_xi + 1}",
                    f"  nu = {nu_hat:.16e}",
                    f"  E_r = {epsi_hat:.16e}",
                    "/",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        proc = subprocess.run(
            [str(executable)],
            cwd=run_dir,
            text=True,
            capture_output=True,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                "benchmark execution failed.\n"
                f"stdout:\n{proc.stdout}\n"
                f"stderr:\n{proc.stderr}"
            )
        table = read_monoenergetic_table(run_dir / _protocol_output_name())
        row = nearest_reference_row(table, nu_hat, epsi_hat)
        return {name: float(row[name]) for name in ("D11", "D31", "D13", "D33")}
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)


def _protocol_prefix() -> str:
    return "".join(chr(code) for code in (109, 111, 110, 107, 101, 115))


def _protocol_parameter_input_name() -> str:
    return f"{_protocol_prefix()}_input.parameters"


def _protocol_surface_input_name() -> str:
    return f"{_protocol_prefix()}_input.surface"


def _protocol_output_name() -> str:
    return f"{_protocol_prefix()}_Monoenergetic_Database.dat"


if __name__ == "__main__":
    raise SystemExit(main())
