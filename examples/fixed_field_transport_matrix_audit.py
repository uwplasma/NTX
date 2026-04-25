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

_rho_env = os.environ.get("NTX_FIXED_FIELD_AUDIT_RHO", "").strip()
if _rho_env:
    RHO_VALUES = np.array([float(value) for value in _rho_env.split(",")], dtype=float)
else:
    RHO_VALUES = np.array([0.25, 0.50, 0.75], dtype=float)
CASE_FILTER = tuple(
    value.strip().lower()
    for value in os.environ.get("NTX_FIXED_FIELD_AUDIT_CASES", "qa,qh").split(",")
    if value.strip()
)
NU_PRIME = 8.31565e-3
ESTAR = 0.0
NTX_GRID = GridSpec(
    n_theta=int(os.environ.get("NTX_FIXED_FIELD_AUDIT_NTX_NTHETA", "25")),
    n_zeta=int(os.environ.get("NTX_FIXED_FIELD_AUDIT_NTX_NZETA", "25")),
    n_xi=int(os.environ.get("NTX_FIXED_FIELD_AUDIT_NTX_NXI", "63")),
)
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
    source_family: str


@dataclass(frozen=True)
class MonoenergeticBridge:
    nu_n: float
    b0_over_bbar: float
    g_hat: float
    i_hat: float
    factor_31: float
    factor_33: float


def _fixed_field_cases() -> dict[str, FixedFieldCase]:
    zenodo_root = find_qs_zenodo_root()
    if zenodo_root is not None:
        data = zenodo_root / "codes" / "simsopt" / "tests" / "test_files"
        cases = {
            "qa": FixedFieldCase(
                name="qa",
                label="QA precise-QS fixed-field reference",
                helicity_n=0,
                wout_path=data / "wout_LandremanPaul2021_QA_reactorScale_lowres_reference.nc",
                source_family="zenodo_precise_qs",
            ),
            "qh": FixedFieldCase(
                name="qh",
                label="QH precise-QS fixed-field reference",
                helicity_n=-1,
                wout_path=data / "wout_LandremanPaul2021_QH_reactorScale_lowres_reference.nc",
                source_family="zenodo_precise_qs",
            ),
        }
        if all(case.wout_path.exists() for case in cases.values()):
            return cases

    vmec_root = find_vmec_jax_root()
    if vmec_root is None:
        raise RuntimeError(
            "fixed-field audit requires either the local precise-QS Zenodo archive "
            "or a vmec_jax checkout"
        )
    data = vmec_root / "examples" / "data"
    cases = {
        "qa": FixedFieldCase(
            name="qa",
            label="QA fixed-field reference",
            helicity_n=0,
            wout_path=data / "wout_LandremanPaul2021_QA_reactorScale_lowres_reference.nc",
            source_family="vmec_jax_examples",
        ),
        "qh": FixedFieldCase(
            name="qh",
            label="QH fixed-field reference",
            helicity_n=-1,
            wout_path=data / "wout_LandremanPaul2021_QH_reactorScale_lowres_reference.nc",
            source_family="vmec_jax_examples",
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


def _bridge_from_scalars(
    *,
    surface_b0: float,
    psi_a_hat: float,
    b0_over_bbar: float,
    g_hat: float,
    i_hat: float,
    iota: float,
    nu_prime: float,
) -> MonoenergeticBridge:
    denom = float(g_hat) + float(iota) * float(i_hat)
    if abs(denom) <= 1.0e-16:
        raise ZeroDivisionError("fixed-field monoenergetic bridge requires GHat + iota*IHat != 0")
    # Match the v3 RHSMode=3 monoenergetic overwrite exactly.
    nu_n = float(nu_prime) * float(b0_over_bbar) / denom
    # Landreman/H. Smith VMEC-s-coordinate bridge used in the benchmark scripts.
    factor_31 = 4.0 * float(surface_b0) * float(psi_a_hat) / (np.sqrt(np.pi) * float(g_hat))
    # The archived formula carries the opposite sign because its stored D33
    # convention differs from NTX's raw D33 sign. NTX raw D33 is positive on
    # the current fixed-field cases, so the bridge here is written in the NTX
    # sign convention.
    factor_33 = 2.0 * float(surface_b0) / (denom * np.sqrt(np.pi))
    return MonoenergeticBridge(
        nu_n=float(nu_n),
        b0_over_bbar=float(b0_over_bbar),
        g_hat=float(g_hat),
        i_hat=float(i_hat),
        factor_31=float(factor_31),
        factor_33=float(factor_33),
    )


def _sfincs_rhsmode3_bridge(
    input_path: Path,
    surface_b0: float,
    psi_a_hat: float,
) -> MonoenergeticBridge:
    root = find_sfincs_jax_root()
    if root is None:
        raise RuntimeError("sfincs_jax checkout is required for the fixed-field audit")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from sfincs_jax.namelist import read_sfincs_input
    from sfincs_jax.transport_matrix import _flux_functions_from_op
    from sfincs_jax.v3 import geometry_from_namelist, grids_from_namelist
    from sfincs_jax.v3_system import full_system_operator_from_namelist

    nml = read_sfincs_input(input_path)
    grids = grids_from_namelist(nml)
    geom = geometry_from_namelist(nml=nml, grids=grids)
    op = full_system_operator_from_namelist(nml=nml, grids=grids, geom=geom)
    b0_over_bbar, g_hat, i_hat = _flux_functions_from_op(op)
    return _bridge_from_scalars(
        surface_b0=surface_b0,
        psi_a_hat=psi_a_hat,
        b0_over_bbar=float(b0_over_bbar),
        g_hat=float(g_hat),
        i_hat=float(i_hat),
        iota=float(geom.iota),
        nu_prime=NU_PRIME,
    )


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


def _compute_ntx_channels(
    case: FixedFieldCase,
    rho: float,
    bridge: MonoenergeticBridge,
) -> dict[str, float]:
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
        MonoenergeticCase(nu_hat=float(bridge.nu_n), epsi_hat=float(ESTAR)),
    )
    return {
        "nu_n": float(bridge.nu_n),
        "surface_b0": float(surface.b0),
        "psi_a_hat": float(surface.psi_a_hat),
        "b0_over_bbar": float(bridge.b0_over_bbar),
        "g_hat": float(bridge.g_hat),
        "i_hat": float(bridge.i_hat),
        "D11_raw": float(result.D11),
        "D31_raw": float(result.D31),
        "D13_raw": float(result.D13),
        "D33_raw": float(result.D33),
        "D33_spitzer": float(result.D33_spitzer),
        "L13_bridge": float(-result.D13 * bridge.factor_31),
        "L31_bridge": float(result.D31 * bridge.factor_31),
        "L33_bridge": float(result.D33 * bridge.factor_33),
        "L33_spitzer_bridge": float(result.D33_spitzer * bridge.factor_33),
        "factor_31": float(bridge.factor_31),
        "factor_33": float(bridge.factor_33),
    }


def _relative_error(a: float, b: float) -> float:
    return abs(a - b) / max(abs(b), 1.0e-16)


def _run_case(case: FixedFieldCase, *, recompute: bool) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for rho in RHO_VALUES:
        input_path, _ = _prepare_sfincs_jax_case(case, float(rho))
        matrix = _run_sfincs_jax_transport_matrix(case, float(rho), recompute=recompute)
        surface = load_vmec_surface(
            case.wout_path,
            psi_n=float(rho**2),
            vmec_radial_option=0,
            vmec_nyquist_option=1,
            vmec_mode_convention="filtered_nyquist",
        )
        bridge = _sfincs_rhsmode3_bridge(input_path, float(surface.b0), float(surface.psi_a_hat))
        ntx = _compute_ntx_channels(case, float(rho), bridge)
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
                    "L13_bridge_vs_L13": _relative_error(ntx["L13_bridge"], float(matrix[0, 1])),
                    "L31_bridge_vs_L31": _relative_error(ntx["L31_bridge"], float(matrix[1, 0])),
                    "L33_bridge_vs_L33": _relative_error(ntx["L33_bridge"], float(matrix[1, 1])),
                    "L33_spitzer_bridge_vs_L33": _relative_error(
                        ntx["L33_spitzer_bridge"], float(matrix[1, 1])
                    ),
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
    case_keys = [key for key in ("qa", "qh") if key in summary]
    if not case_keys:
        raise ValueError("fixed-field transport-matrix audit plot requires at least one case")
    fig, axes = plt.subplots(
        3,
        len(case_keys),
        figsize=(5.6 * len(case_keys), 9.4),
        sharex=True,
        constrained_layout=True,
    )
    axes = np.asarray(axes, dtype=object).reshape(3, len(case_keys))
    channel_specs = (
        ("L13", "L13_bridge", r"$L_{13}$ vs archive-backed bridge"),
        ("L31", "L31_bridge", r"$L_{31}$ vs archive-backed bridge"),
        ("L33", "L33_bridge", r"$L_{33}$ vs archive-backed bridge"),
    )
    colors = {"sfincs": "#d55e00", "ntx": "#1f77b4", "spitzer": "#6c757d"}

    for col, case_key in enumerate(case_keys):
        rows = summary[case_key]["rows"]
        rho = np.array([row["rho"] for row in rows], dtype=float)
        for row_idx, (sfincs_key, ntx_key, title) in enumerate(channel_specs):
            ax = axes[row_idx, col]
            sfincs_vals = np.array([row["sfincs_jax"][sfincs_key] for row in rows], dtype=float)
            ntx_vals = np.array([row["ntx"][ntx_key] for row in rows], dtype=float)
            ax.plot(rho, sfincs_vals, "o-", lw=2.2, color=colors["sfincs"], label="SFINCS-JAX")
            ax.plot(rho, ntx_vals, "s--", lw=2.0, color=colors["ntx"], label="NTX bridge")
            if sfincs_key == "L33":
                spitzer_vals = np.array(
                    [row["ntx"]["L33_spitzer_bridge"] for row in rows],
                    dtype=float,
                )
                ax.plot(
                    rho,
                    spitzer_vals,
                    "^:",
                    lw=1.8,
                    color=colors["spitzer"],
                    label="NTX Spitzer bridge",
                )
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
        if CASE_FILTER and key not in CASE_FILTER:
            continue
        summary[key] = _run_case(case, recompute=RECOMPUTE)
    _plot(summary)
    OUTPUT_PREFIX.with_suffix(".json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
