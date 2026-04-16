#!/usr/bin/env python3
"""Audit fixed-field monoenergetic channels against SFINCS-JAX transport matrices."""
# ruff: noqa: E402

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ntx import GridSpec, MonoenergeticCase, load_vmec_surface, solve_monoenergetic
from ntx._checkout_paths import find_qs_zenodo_root, find_sfincs_jax_root, find_vmec_jax_root

OUTPUT_DIR = ROOT / "examples" / "outputs" / "fixed_field_transport_matrix_audit"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PREFIX = OUTPUT_DIR / "fixed_field_transport_matrix_audit"

RHO_VALUES = np.array([0.25, 0.50, 0.75], dtype=float)
NU_PRIME = 8.31565e-3
ESTAR = 0.0
NTX_GRID = GridSpec(n_theta=25, n_zeta=25, n_xi=63)
SFINCS_RESOLUTION = {
    "Ntheta": 25,
    "Nzeta": 39,
    "Nxi": 60,
    "Nx": 7,
    "solverTolerance": "1d-6",
}
RECOMPUTE = os.environ.get("NTX_FIXED_FIELD_AUDIT_RECOMPUTE", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}


@dataclass(frozen=True)
class FixedFieldCase:
    name: str
    label: str
    helicity_n: int
    wout_path: Path


def _fixed_field_cases() -> dict[str, FixedFieldCase]:
    vmec_root = find_vmec_jax_root()
    if vmec_root is None:
        raise RuntimeError("vmec_jax checkout is required for the fixed-field audit")
    data = vmec_root / "examples" / "data"
    cases = {
        "qa": FixedFieldCase(
            name="qa",
            label="QA fixed-field reference",
            helicity_n=0,
            wout_path=data / "wout_LandremanPaul2021_QA_reactorScale_lowres_reference.nc",
        ),
        "qh": FixedFieldCase(
            name="qh",
            label="QH fixed-field reference",
            helicity_n=-1,
            wout_path=data / "wout_LandremanPaul2021_QH_reactorScale_lowres_reference.nc",
        ),
    }
    missing = [case.wout_path for case in cases.values() if not case.wout_path.exists()]
    if missing:
        raise FileNotFoundError(f"missing fixed-field reference files: {missing}")
    return cases


def _sfincs_jax_pythonpath() -> list[str]:
    root = find_sfincs_jax_root()
    if root is None:
        raise RuntimeError("sfincs_jax checkout is required for the fixed-field audit")
    existing = [path for path in sys.path if path]
    return [str(root), *existing]


def _namelist_text(case: FixedFieldCase, rho: float) -> str:
    return f"""! Fixed-field transport-matrix audit case
&general
  RHSMode = 3
/

&geometryParameters
  geometryScheme = 5
  VMECRadialOption = 0
  inputRadialCoordinate = 3
  rN_wish = {rho:.8f}
  equilibriumFile = "{case.wout_path}"
/

&speciesParameters
/

&physicsParameters
  nuPrime = {NU_PRIME:.8e}
  EStar = {ESTAR:.8e}
  collisionOperator = 1
  includeXDotTerm = .false.
  includeElectricFieldTermInXiDot = .false.
  useDKESExBDrift = .true.
  includePhi1 = .false.
/

&resolutionParameters
  Ntheta = {SFINCS_RESOLUTION["Ntheta"]}
  Nzeta = {SFINCS_RESOLUTION["Nzeta"]}
  Nxi = {SFINCS_RESOLUTION["Nxi"]}
  Nx = {SFINCS_RESOLUTION["Nx"]}
  solverTolerance = {SFINCS_RESOLUTION["solverTolerance"]}
/

&otherNumericalParameters
/

&preconditionerOptions
/

&export_f
/
"""


def _prepare_sfincs_jax_case(case: FixedFieldCase, rho: float) -> tuple[Path, Path]:
    case_dir = OUTPUT_DIR / case.name / f"rho_{rho:.3f}"
    case_dir.mkdir(parents=True, exist_ok=True)
    input_path = case_dir / "input.namelist"
    matrix_path = case_dir / "transportMatrix.npy"
    input_path.write_text(_namelist_text(case, rho), encoding="utf-8")
    return input_path, matrix_path


def _run_sfincs_jax_transport_matrix(
    case: FixedFieldCase,
    rho: float,
    *,
    recompute: bool,
) -> np.ndarray:
    input_path, matrix_path = _prepare_sfincs_jax_case(case, rho)
    if recompute or not matrix_path.exists():
        env = dict(os.environ)
        env["PYTHONPATH"] = ":".join(_sfincs_jax_pythonpath())
        command = [
            sys.executable,
            "-c",
            (
                "from sfincs_jax.cli import main; "
                "raise SystemExit(main())"
            ),
            "transport-matrix-v3",
            "--input",
            str(input_path),
            "--out-matrix",
            str(matrix_path),
            "--solve-method",
            "auto",
            "--tol",
            "1e-10",
        ]
        subprocess.run(
            command,
            cwd=str(find_sfincs_jax_root()),
            env=env,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
    matrix = np.load(matrix_path)
    if matrix.shape != (2, 2):
        raise ValueError(f"expected a 2x2 transport matrix, got {matrix.shape}")
    return np.asarray(matrix, dtype=float)


def _compute_ntx_channels(case: FixedFieldCase, rho: float) -> dict[str, float]:
    surface = load_vmec_surface(
        case.wout_path,
        psi_n=float(rho**2),
        vmec_radial_option=0,
        vmec_nyquist_option=1,
        vmec_mode_convention="filtered_nyquist",
    )
    result = solve_monoenergetic(
        surface,
        NTX_GRID,
        MonoenergeticCase(nu_hat=float(NU_PRIME), epsi_hat=float(ESTAR)),
    )
    drds = float(surface.aminor_p * 0.5 / max(rho, 1.0e-8))
    return {
        "D11_raw": float(result.D11),
        "D31_raw": float(result.D31),
        "D13_raw": float(result.D13),
        "D33_raw": float(result.D33),
        "D13_neopax": float(-result.D13 * drds),
        "D31_plus_drds": float(result.D31 * drds),
        "D31_minus_drds": float(-result.D31 * drds),
        "D33_nu": float(result.D33 * NU_PRIME),
        "drds": drds,
    }


def _relative_error(a: float, b: float) -> float:
    return abs(a - b) / max(abs(b), 1.0e-16)


def _run_case(case: FixedFieldCase, *, recompute: bool) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for rho in RHO_VALUES:
        matrix = _run_sfincs_jax_transport_matrix(case, float(rho), recompute=recompute)
        ntx = _compute_ntx_channels(case, float(rho))
        rows.append(
            {
                "rho": float(rho),
                "sfincs_jax": {
                    "L11": float(matrix[0, 0]),
                    "L13": float(matrix[0, 1]),
                    "L31": float(matrix[1, 0]),
                    "L33": float(matrix[1, 1]),
                },
                "ntx": ntx,
                "candidate_relative_error": {
                    "D13_neopax_vs_L13": _relative_error(ntx["D13_neopax"], float(matrix[0, 1])),
                    "D31_plus_drds_vs_L31": _relative_error(
                        ntx["D31_plus_drds"], float(matrix[1, 0])
                    ),
                    "D31_minus_drds_vs_L31": _relative_error(
                        ntx["D31_minus_drds"], float(matrix[1, 0])
                    ),
                    "D33_nu_vs_L33": _relative_error(ntx["D33_nu"], float(matrix[1, 1])),
                },
                "onsager": {
                    "ntx_D31_plus_D13": float(ntx["D31_raw"] + ntx["D13_raw"]),
                    "sfincs_L31_minus_L13": float(matrix[1, 0] - matrix[0, 1]),
                },
            }
        )
    case_payload = asdict(case)
    case_payload["wout_path"] = str(case.wout_path)
    return {"case": case_payload, "rows": rows}


def _plot(summary: dict[str, Any]) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(11.2, 9.4), sharex=True, constrained_layout=True)
    channel_specs = (
        ("L13", "D13_neopax", r"$L_{13}$ vs $-D_{13}\,dr/ds$"),
        ("L31", "D31_minus_drds", r"$L_{31}$ vs $-D_{31}\,dr/ds$"),
        ("L33", "D33_nu", r"$L_{33}$ vs $\nu D_{33}$"),
    )
    colors = {"sfincs": "#d55e00", "ntx": "#1f77b4"}

    for col, case_key in enumerate(("qa", "qh")):
        rows = summary[case_key]["rows"]
        rho = np.array([row["rho"] for row in rows], dtype=float)
        for row_idx, (sfincs_key, ntx_key, title) in enumerate(channel_specs):
            ax = axes[row_idx, col]
            sfincs_vals = np.array([row["sfincs_jax"][sfincs_key] for row in rows], dtype=float)
            ntx_vals = np.array([row["ntx"][ntx_key] for row in rows], dtype=float)
            ax.plot(rho, sfincs_vals, "o-", lw=2.2, color=colors["sfincs"], label="SFINCS-JAX")
            ax.plot(rho, ntx_vals, "s--", lw=2.0, color=colors["ntx"], label="NTX candidate")
            ax.set_title(f"{summary[case_key]['case']['label']}: {title}")
            ax.grid(alpha=0.24, lw=0.6)
            if row_idx == 2:
                ax.set_xlabel(r"$\rho$")
            if col == 0:
                ax.set_ylabel("normalized coefficient")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        frameon=False,
    )
    fig.savefig(OUTPUT_PREFIX.with_suffix(".png"), dpi=250, bbox_inches="tight")
    fig.savefig(OUTPUT_PREFIX.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    cases = _fixed_field_cases()
    zenodo = find_qs_zenodo_root()
    summary = {
        "inputs": {
            "rho": RHO_VALUES.tolist(),
            "nuPrime": NU_PRIME,
            "EStar": ESTAR,
            "ntx_grid": {
                "n_theta": NTX_GRID.n_theta,
                "n_zeta": NTX_GRID.n_zeta,
                "n_xi": NTX_GRID.n_xi,
            },
            "sfincs_resolution": SFINCS_RESOLUTION,
            "zenodo_root": str(zenodo) if zenodo is not None else None,
        }
    }
    for key, case in cases.items():
        summary[key] = _run_case(case, recompute=RECOMPUTE)
    _plot(summary)
    OUTPUT_PREFIX.with_suffix(".json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
