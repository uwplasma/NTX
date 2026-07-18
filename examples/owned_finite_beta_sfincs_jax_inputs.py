#!/usr/bin/env python3
"""Generate owned SFINCS-JAX inputs on the same finite-beta grids as NTX.

This script does not use archived transport outputs.  It writes SFINCS-JAX
``RHSMode=3`` monoenergetic input decks that point at the same VMEC ``wout``
files, radial grid, collisionality grid, electric-field grid, and resolution
contract used by ``owned_geometry_neopax_dataset.py``.  Passing
``--run-sfincs-jax`` then runs those decks through the local SFINCS-JAX checkout
and stores HDF5 outputs beside the generated inputs.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from examples.owned_finite_beta_bootstrap_comparison import (  # noqa: E402
    ProfileContract,
    _profile_values,
)
from examples.owned_geometry_neopax_dataset import (  # noqa: E402
    DEFAULT_ES,
    DEFAULT_NU_V,
    OwnedJaxGeometryCase,
    discover_owned_case_specs,
)
from ntx import (  # noqa: E402
    GridSpec,
    MonoenergeticCase,
    solve_monoenergetic,
    surface_from_vmex_vmec_wout_file,
)

OUTPUT_PREFIX = ROOT / "docs" / "_static" / "owned_finite_beta_sfincs_jax_inputs"
WORKDIR = ROOT / "examples" / "outputs" / "owned_finite_beta_sfincs_jax_inputs"
SFINCS_JAX_ROOT = Path(
    os.environ.get("NTX_SFINCS_JAX_ROOT", "/Users/rogeriojorge/local/tests/sfincs_jax")
)
DEFAULT_GRID = GridSpec(25, 31, 32)
DEFAULT_SFINCS_RHO = (1.0 / 7.0, 0.30, 0.50)
DEFAULT_RHS2_NL = 3
DEFAULT_RHS2_NX = 3
RHSMODE2_SPECIES = {
    "electron": {"Zs": -1.0, "mHats": 1.0 / 1836.15267343},
    "ion": {"Zs": 1.0, "mHats": 2.0},
}


@dataclass(frozen=True)
class SfincsDeck:
    case_id: str
    case_label: str
    family: str
    rhs_mode: int
    species: str | None
    n_hat: float | None
    t_hat: float | None
    rho: float
    s: float
    nu_prime: float
    e_star: float
    input_path: Path
    output_path: Path
    wout_path: Path
    status: str
    seconds: float | None = None
    error: str | None = None
    transport_summary: dict[str, object] | None = None

    def as_payload(self) -> dict[str, object]:
        payload = asdict(self)
        for key in ("input_path", "output_path", "wout_path"):
            payload[key] = str(payload[key])
        return payload


def _safe_float_label(value: float) -> str:
    return f"{value:.6g}".replace("-", "m").replace("+", "").replace(".", "p")


def _rhsmode2_hat_values(
    rho: float,
    *,
    use_profile_contract: bool,
) -> tuple[float, float]:
    if not use_profile_contract:
        return 1.0, 1.0
    profiles = _profile_values(
        np.asarray([float(rho)], dtype=float),
        ProfileContract(),
        a_b=1.0,
    )
    return (
        float(np.asarray(profiles["density"], dtype=float)[0] / 1.0e20),
        float(np.asarray(profiles["temperature"], dtype=float)[0] / 1.0e3),
    )


def _select_cases(
    case_specs: tuple[OwnedJaxGeometryCase, ...] | None,
    case_ids: tuple[str, ...],
    case_limit: int | None,
) -> tuple[OwnedJaxGeometryCase, ...]:
    cases = list(case_specs if case_specs is not None else discover_owned_case_specs())
    cases = [case for case in cases if "finite beta" in case.family.lower()]
    if case_ids:
        requested = set(case_ids)
        cases = [case for case in cases if case.id in requested]
    if case_limit is not None and case_limit > 0:
        cases = cases[:case_limit]
    return tuple(cases)


def _sfincs_input_text(
    *,
    rhs_mode: int,
    rhsmode2_species: str,
    rhsmode2_n_hat: float,
    rhsmode2_t_hat: float,
    wout_path: Path,
    rho: float,
    nu_prime: float,
    e_star: float,
    grid: GridSpec,
    rhs2_nl: int,
    rhs2_nx: int,
    solver_tolerance: float,
    min_bmn_to_load: float,
) -> str:
    rhs_mode = int(rhs_mode)
    if rhs_mode == 3:
        species_parameters = ""
        physics_frequency = f"  nuPrime = {nu_prime:.17g}\n  EStar = {e_star:.17g}"
        resolution_extra = "  Nx = 1"
        other_numerics = ""
    elif rhs_mode == 2:
        if rhsmode2_species not in RHSMODE2_SPECIES:
            raise ValueError(f"rhsmode2_species must be one of {sorted(RHSMODE2_SPECIES)}")
        species_values = RHSMODE2_SPECIES[rhsmode2_species]
        species_parameters = (
            f"  Zs = {species_values['Zs']:.17g}\n"
            f"  mHats = {species_values['mHats']:.17g}\n"
            f"  nHats = {float(rhsmode2_n_hat):.17g}\n"
            f"  THats = {float(rhsmode2_t_hat):.17g}"
        )
        physics_frequency = f"  nu_n = {nu_prime:.17g}\n  Er = {e_star:.17g}"
        resolution_extra = f"  NL = {int(rhs2_nl)}\n  Nx = {int(rhs2_nx)}"
        other_numerics = "  Nxi_for_x_option = 0"
    else:
        raise ValueError("rhs_mode must be 2 or 3")
    return f"""! Owned finite-beta SFINCS-JAX input generated by NTX.
! The geometry/profile grid is paired with examples/owned_geometry_neopax_dataset.py.

&general
  RHSMode = {rhs_mode}
/

&geometryParameters
  geometryScheme = 5
  equilibriumFile = "{wout_path}"
  inputRadialCoordinate = 3
  rN_wish = {rho:.17g}
  VMECRadialOption = 0
  min_Bmn_to_load = {min_bmn_to_load:.17g}
/

&speciesParameters
{species_parameters}
/

&physicsParameters
{physics_frequency}
  collisionOperator = 1
  includeXDotTerm = .false.
  includeElectricFieldTermInXiDot = .false.
  useDKESExBDrift = .true.
  includePhi1 = .false.
/

&resolutionParameters
  Ntheta = {int(grid.n_theta)}
  Nzeta = {int(grid.n_zeta)}
  Nxi = {int(grid.n_xi)}
{resolution_extra}
  solverTolerance = {solver_tolerance:.17g}
/

&otherNumericalParameters
{other_numerics}
/

&preconditionerOptions
/

&export_f
/
"""


def _run_sfincs_jax(
    input_path: Path,
    output_path: Path,
    *,
    timeout_s: int,
) -> tuple[str, float | None, str | None]:
    env = os.environ.copy()
    env.setdefault("SFINCS_JAX_GMRES_DISTRIBUTED", "0")
    env.setdefault("SFINCS_JAX_MATVEC_SHARD_AXIS", "off")
    if SFINCS_JAX_ROOT.exists():
        env["PYTHONPATH"] = f"{SFINCS_JAX_ROOT}{os.pathsep}{env.get('PYTHONPATH', '')}"
    command = [
        sys.executable,
        "-m",
        "sfincs_jax",
        "write-output",
        "--input",
        str(input_path),
        "--out",
        str(output_path),
        "--compute-transport-matrix",
        "--quiet",
    ]
    start = time.perf_counter()
    try:
        subprocess.run(
            command,
            check=True,
            cwd=input_path.parent,
            env=env,
            timeout=timeout_s,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:  # pragma: no cover - optional external runtime.
        details = (exc.stderr or exc.stdout or str(exc)).strip()
        if len(details) > 2000:
            details = details[:2000] + "..."
        return (
            "failed",
            time.perf_counter() - start,
            f"CalledProcessError({exc.returncode}): {details}",
        )
    except subprocess.TimeoutExpired as exc:  # pragma: no cover - optional external runtime.
        return "failed", time.perf_counter() - start, f"TimeoutExpired: {exc}"
    except Exception as exc:  # pragma: no cover - optional external runtime.
        return "failed", time.perf_counter() - start, f"{type(exc).__name__}: {exc}"
    return "complete", time.perf_counter() - start, None


def _summarize_sfincs_output(
    output_path: Path,
    *,
    nu_prime: float,
) -> dict[str, object] | None:
    if not output_path.exists():
        return None
    try:
        import h5py
    except ModuleNotFoundError:  # pragma: no cover - h5py is optional outside examples.
        return {
            "status": "unreadable",
            "reason": "h5py is required to inspect SFINCS-JAX HDF5 outputs",
        }
    with h5py.File(output_path, "r") as handle:
        if "transportMatrix" not in handle:
            return {"status": "missing_transportMatrix"}
        matrix = np.asarray(handle["transportMatrix"], dtype=float)
        scalars: dict[str, float | int] = {}
        for name in (
            "B0OverBBar",
            "Delta",
            "GHat",
            "IHat",
            "alpha",
            "iota",
            "psiAHat",
            "aHat",
            "rN",
            "nu_n",
            "Ntheta",
            "Nzeta",
            "Nxi",
            "Nx",
        ):
            if name in handle:
                value = np.asarray(handle[name])
                if value.shape == ():
                    item = value.item()
                    scalars[name] = int(item) if name.startswith("N") else float(item)
        species_arrays: dict[str, list[float]] = {}
        for name in ("x", "Zs", "mHats", "nHats", "THats"):
            if name in handle:
                species_arrays[name] = np.asarray(handle[name], dtype=float).reshape(-1).tolist()
        nu_n = float(scalars["nu_n"]) if "nu_n" in scalars else float("nan")
        ratio = nu_n / float(nu_prime) if float(nu_prime) != 0.0 else float("nan")
        pas_nu_d = _sfincs_pas_deflection_frequency_first_species(species_arrays)
        return {
            "status": "complete",
            "transportMatrix": matrix.tolist(),
            "transportMatrix_shape": list(matrix.shape),
            "transportMatrix_abs_max": float(np.nanmax(np.abs(matrix))),
            "sfincs_runtime_nu_n": nu_n,
            "sfincs_runtime_nu_n_over_input_nuPrime": float(ratio),
            "sfincs_pas_nu_d_hat_first_species": pas_nu_d,
            "scalars": scalars,
            "species_arrays": species_arrays,
        }


def _sfincs_pas_deflection_frequency_first_species(
    arrays: dict[str, list[float]],
) -> float:
    """Return SFINCS' PAS deflection-frequency multiplier for RHSMode=3.

    NTX's monoenergetic PAS block uses `nu_hat * l(l+1)/2`. SFINCS RHSMode=3
    uses `nu_n * nuDHat(x=1) * l(l+1)/2`. For a same-operator comparison,
    the NTX solve must therefore use `nu_hat = nu_n * nuDHat`.
    """

    x = np.asarray(arrays.get("x", [1.0]), dtype=float).reshape(-1)
    z_s = np.asarray(arrays.get("Zs", [1.0]), dtype=float).reshape(-1)
    m_hats = np.asarray(arrays.get("mHats", [1.0]), dtype=float).reshape(-1)
    n_hats = np.asarray(arrays.get("nHats", [1.0]), dtype=float).reshape(-1)
    t_hats = np.asarray(arrays.get("THats", [1.0]), dtype=float).reshape(-1)
    if x.size == 0 or z_s.size == 0:
        return 1.0

    species_count = min(z_s.size, m_hats.size, n_hats.size, t_hats.size)
    if species_count == 0:
        return 1.0
    z_s = z_s[:species_count]
    m_hats = m_hats[:species_count]
    n_hats = n_hats[:species_count]
    t_hats = t_hats[:species_count]
    x0 = float(x[0])
    if not np.isfinite(x0) or x0 == 0.0:
        return 1.0

    sqrt_pi = math.sqrt(math.pi)
    a = 0
    total = 0.0
    for b in range(species_count):
        species_factor = math.sqrt((t_hats[a] * m_hats[b]) / (t_hats[b] * m_hats[a]))
        xb = x0 * species_factor
        if abs(xb) < 1.0e-5:
            psi = ((2.0 / 3.0) * xb - (2.0 / 5.0) * xb**3 + (1.0 / 7.0) * xb**5) / sqrt_pi
        else:
            psi = (math.erf(xb) - (2.0 / sqrt_pi) * xb * math.exp(-(xb * xb))) / (
                2.0 * xb * xb
            )
        total += z_s[b] * z_s[b] * n_hats[b] * (math.erf(xb) - psi) / (x0**3)
    t32m = t_hats[a] * math.sqrt(t_hats[a] * m_hats[a])
    nu_d = (3.0 * sqrt_pi / 4.0) * z_s[a] * z_s[a] * total / t32m
    if not np.isfinite(nu_d) or nu_d <= 0.0:
        return 1.0
    return float(nu_d)


def _relative_difference(reference: float, candidate: float) -> float:
    return abs(float(candidate) - float(reference)) / max(abs(float(reference)), 1.0e-30)


def _ntx_same_grid_transport_summary(
    *,
    case: OwnedJaxGeometryCase,
    rho: float,
    e_star: float,
    grid: GridSpec,
    min_bmn_to_load: float,
    transport_summary: dict[str, object],
) -> dict[str, object]:
    """Compare one completed SFINCS-JAX transport matrix to the NTX VMEC path."""

    matrix = np.asarray(transport_summary["transportMatrix"], dtype=float)
    scalars = dict(transport_summary.get("scalars", {}))
    required = ("GHat", "IHat", "iota", "psiAHat", "nu_n")
    missing = [name for name in required if name not in scalars]
    if missing:
        return {
            "status": "skipped",
            "reason": f"SFINCS-JAX output is missing required scalars: {missing}",
        }
    surface = surface_from_vmex_vmec_wout_file(
        case.wout_path,
        s=float(rho) ** 2,
        min_bmn_to_load=min_bmn_to_load,
    )
    result = solve_monoenergetic(
        surface,
        grid,
        MonoenergeticCase(
            nu_hat=float(scalars["nu_n"])
            * float(transport_summary.get("sfincs_pas_nu_d_hat_first_species", 1.0)),
            epsi_hat=float(e_star),
        ),
    )
    denom = float(scalars["GHat"]) + float(scalars["iota"]) * float(scalars["IHat"])
    b0_over_bbar = float(scalars.get("B0OverBBar", float(surface.b0)))
    pas_nu_d = float(transport_summary.get("sfincs_pas_nu_d_hat_first_species", 1.0))
    factor_31 = (
        4.0
        * b0_over_bbar
        * float(scalars["psiAHat"])
        / (np.sqrt(np.pi) * float(scalars["GHat"]))
    )
    factor_33 = pas_nu_d * 4.0 * b0_over_bbar / (3.0 * denom)
    ntx = {
        "D11_raw": float(result.D11),
        "D13_raw": float(result.D13),
        "D31_raw": float(result.D31),
        "D33_raw": float(result.D33),
        "D33_spitzer": float(result.D33_spitzer),
        "L13_bridge": float(-result.D13 * factor_31),
        "L31_bridge": float(result.D31 * factor_31),
        "L33_bridge": float(result.D33 * factor_33),
        "L33_spitzer_bridge": float(result.D33_spitzer * factor_33),
        "factor_31": float(factor_31),
        "factor_33": float(factor_33),
        "surface_b0": float(surface.b0),
        "sfincs_B0OverBBar": b0_over_bbar,
        "surface_psi_a_hat": float(surface.psi_a_hat),
        "sfincs_pas_nu_d_hat_first_species": pas_nu_d,
        "effective_ntx_nu_hat": float(scalars["nu_n"]) * pas_nu_d,
    }
    return {
        "status": "complete",
        "geometry_path": "vmex wout harmonics -> NTX VmecSurface",
        "comparison_scope": (
            "Coefficient-level finite-beta diagnostic only; bootstrap-current "
            "parity requires profile closure and production-resolution scans."
        ),
        "ntx": ntx,
        "relative_difference": {
            "L13_bridge_vs_sfincs": _relative_difference(matrix[0, 1], ntx["L13_bridge"]),
            "L31_bridge_vs_sfincs": _relative_difference(matrix[1, 0], ntx["L31_bridge"]),
            "L33_bridge_vs_sfincs": _relative_difference(matrix[1, 1], ntx["L33_bridge"]),
            "L33_spitzer_bridge_vs_sfincs": _relative_difference(
                matrix[1, 1],
                ntx["L33_spitzer_bridge"],
            ),
        },
    }


def build_payload(
    *,
    case_specs: tuple[OwnedJaxGeometryCase, ...] | None = None,
    case_ids: tuple[str, ...] = (),
    case_limit: int | None = 2,
    rho: tuple[float, ...] = DEFAULT_SFINCS_RHO,
    nu_v: tuple[float, ...] = DEFAULT_NU_V,
    es_values: tuple[float, ...] = DEFAULT_ES,
    grid: GridSpec = DEFAULT_GRID,
    output_dir: Path = WORKDIR,
    run_sfincs_jax: bool = False,
    rhs_mode: int = 3,
    rhsmode2_species: str = "electron",
    rhsmode2_use_profile_contract: bool = False,
    rhs2_nl: int = DEFAULT_RHS2_NL,
    rhs2_nx: int = DEFAULT_RHS2_NX,
    timeout_s: int = 300,
    solver_tolerance: float = 1.0e-6,
    min_bmn_to_load: float = 1.0e-5,
) -> dict[str, object]:
    """Write SFINCS-JAX input decks and optionally execute them."""

    rhs_mode = int(rhs_mode)
    if rhs_mode not in {2, 3}:
        raise ValueError("rhs_mode must be 2 or 3")
    if rhs_mode == 2 and rhsmode2_species not in RHSMODE2_SPECIES:
        raise ValueError(f"rhsmode2_species must be one of {sorted(RHSMODE2_SPECIES)}")
    cases = _select_cases(case_specs, case_ids, case_limit)
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    decks: list[SfincsDeck] = []
    for case in cases:
        for rho_value in rho:
            for nu_value in nu_v:
                for es_value in es_values:
                    rhsmode2_n_hat, rhsmode2_t_hat = _rhsmode2_hat_values(
                        float(rho_value),
                        use_profile_contract=bool(rhsmode2_use_profile_contract),
                    )
                    deck_base = output_dir / case.id
                    if rhs_mode == 2:
                        profile_label = "profile" if rhsmode2_use_profile_contract else "unit"
                        deck_base = deck_base / f"rhsMode_2_{rhsmode2_species}_{profile_label}"
                    deck_dir = (
                        deck_base
                        / f"rho_{_safe_float_label(float(rho_value))}"
                        / f"nuPrime_{_safe_float_label(float(nu_value))}"
                        / f"EStar_{_safe_float_label(float(es_value))}"
                    )
                    deck_dir.mkdir(parents=True, exist_ok=True)
                    input_path = deck_dir / "input.namelist"
                    output_path = deck_dir / "sfincs_jax_output.h5"
                    input_path.write_text(
                        _sfincs_input_text(
                            rhs_mode=rhs_mode,
                            rhsmode2_species=rhsmode2_species,
                            rhsmode2_n_hat=rhsmode2_n_hat,
                            rhsmode2_t_hat=rhsmode2_t_hat,
                            wout_path=case.wout_path,
                            rho=float(rho_value),
                            nu_prime=float(nu_value),
                            e_star=float(es_value),
                            grid=grid,
                            rhs2_nl=rhs2_nl,
                            rhs2_nx=rhs2_nx,
                            solver_tolerance=solver_tolerance,
                            min_bmn_to_load=min_bmn_to_load,
                        )
                    )
                    status = "input_written"
                    seconds = None
                    error = None
                    if run_sfincs_jax:
                        status, seconds, error = _run_sfincs_jax(
                            input_path,
                            output_path,
                            timeout_s=timeout_s,
                        )
                    elif output_path.exists():
                        status = "output_found"
                    transport_summary = _summarize_sfincs_output(
                        output_path,
                        nu_prime=float(nu_value),
                    )
                    if (
                        rhs_mode == 3
                        and transport_summary is not None
                        and transport_summary.get("status") == "complete"
                    ):
                        try:
                            transport_summary["ntx_same_grid"] = (
                                _ntx_same_grid_transport_summary(
                                    case=case,
                                    rho=float(rho_value),
                                    e_star=float(es_value),
                                    grid=grid,
                                    min_bmn_to_load=min_bmn_to_load,
                                    transport_summary=transport_summary,
                                )
                            )
                        except Exception as exc:  # pragma: no cover - optional stack runtime.
                            transport_summary["ntx_same_grid"] = {
                                "status": "failed",
                                "reason": f"{type(exc).__name__}: {exc}",
                            }
                    decks.append(
                        SfincsDeck(
                            case_id=case.id,
                            case_label=case.label,
                            family=case.family,
                            rhs_mode=rhs_mode,
                            species=(rhsmode2_species if rhs_mode == 2 else None),
                            n_hat=(rhsmode2_n_hat if rhs_mode == 2 else None),
                            t_hat=(rhsmode2_t_hat if rhs_mode == 2 else None),
                            rho=float(rho_value),
                            s=float(rho_value) ** 2,
                            nu_prime=float(nu_value),
                            e_star=float(es_value),
                            input_path=input_path,
                            output_path=output_path,
                            wout_path=case.wout_path,
                            status=status,
                            seconds=seconds,
                            error=error,
                            transport_summary=transport_summary,
                        )
                    )

    deck_payloads = [deck.as_payload() for deck in decks]
    completed_transport = [
        deck.transport_summary
        for deck in decks
        if deck.transport_summary is not None
        and deck.transport_summary.get("status") == "complete"
    ]
    nu_ratios = [
        float(summary["sfincs_runtime_nu_n_over_input_nuPrime"])
        for summary in completed_transport
        if np.isfinite(float(summary["sfincs_runtime_nu_n_over_input_nuPrime"]))
    ]
    ntx_channel_errors = [
        float(value)
        for summary in completed_transport
        if isinstance(summary.get("ntx_same_grid"), dict)
        and summary["ntx_same_grid"].get("status") == "complete"
        for key, value in summary["ntx_same_grid"]["relative_difference"].items()
        if not str(key).startswith("L33_spitzer")
        if np.isfinite(float(value))
    ]
    ntx_spitzer_channel_errors = [
        float(value)
        for summary in completed_transport
        if isinstance(summary.get("ntx_same_grid"), dict)
        and summary["ntx_same_grid"].get("status") == "complete"
        for key, value in summary["ntx_same_grid"]["relative_difference"].items()
        if str(key).startswith("L33_spitzer")
        if np.isfinite(float(value))
    ]
    grid_label = f"{int(grid.n_theta)} x {int(grid.n_zeta)} x {int(grid.n_xi)}"
    rhs_scope = (
        "RHSMode=3 monoenergetic"
        if rhs_mode == 3
        else f"RHSMode=2 energy-integrated row-3 ({rhsmode2_species})"
    )
    return {
        "benchmark": "owned_finite_beta_sfincs_jax_inputs",
        "classification": "owned finite-beta SFINCS-JAX generation contract",
        "claim_scope": (
            f"Generates SFINCS-JAX {rhs_scope} input decks on the same finite-beta "
            "VMEC wout, rho, collisionality, electric-field, and resolution grids "
            "used by the owned NTX+NEOPAX scan lane. Completed outputs are ingested "
            "with the reported nu_n normalization. RHSMode=3 outputs are compared "
            "against NTX on the same geometry and grid; RHSMode=2 outputs are kept "
            "as profile-source current-row diagnostics. This payload uses a "
            f"{grid_label} coefficient grid. The default committed artifact is a "
            "smoke-resolution ladder, while separately named output prefixes can "
            "record production stress probes. Neither form is a finite-beta "
            "bootstrap-current parity claim until profile-current closure "
            "diagnostics are complete on the same contract."
        ),
        "normalization_contract": {
            "rho_to_s": "s=rho^2",
            "ntx_nu_v_to_sfincs_nuPrime": (
                "SFINCS-JAX reports runtime nu_n = nuPrime*B0OverBBar/(GHat+iota*IHat); "
                "owned parity comparisons must either solve NTX at that reported nu_n "
                "or generate nuPrime from the target NTX nu_v using the same bridge"
            ),
            "radial_interpolation": (
                "VMECRadialOption=0 is used so SFINCS-JAX evaluates the requested "
                "rN_wish directly instead of snapping to a nearest VMEC half-grid surface."
            ),
            "rhs3_flow_row_bridge": (
                "NTX D33 maps to the RHSMode=3 SFINCS flow row with "
                "nuDHat*4*B0OverBBar/(3*(GHat+iota*IHat)); this is the row-2 "
                "flow normalization in the SFINCS transport-matrix diagnostic "
                "when collisionOperator=1."
            ),
            "pas_collision_frequency_bridge": (
                "NTX uses nu_hat as the monoenergetic pitch-angle scattering "
                "frequency, while SFINCS-JAX RHSMode=3 uses nu_n*nuDHat(x=1). "
                "The same-grid NTX solve therefore uses nu_hat=nu_n*nuDHat."
            ),
            "ntx_es_to_sfincs_EStar": (
                "identity for this monoenergetic generation audit; electric-field "
                "normalization must stay explicit in any promoted comparison"
            ),
            "rhsmode2_frequency_semantics": (
                "RHSMode=2 uses nu_n directly, not the RHSMode=3 nuPrime overwrite. "
                "The generator stores the input axis under nuPrime for backward "
                "JSON compatibility but writes nu_n to RHSMode=2 decks."
            ),
            "rhsmode2_profile_contract": (
                "When rhsmode2_use_profile_contract is true, nHats=n/1e20 and "
                "THats=T/1keV are evaluated from the same analytic finite-beta "
                "profile contract used by the NTX+NEOPAX bootstrap-current audit."
            ),
        },
        "inputs": {
            "rhs_mode": int(rhs_mode),
            "rhsmode2_species": rhsmode2_species if rhs_mode == 2 else None,
            "rhsmode2_use_profile_contract": (
                bool(rhsmode2_use_profile_contract) if rhs_mode == 2 else None
            ),
            "rho": [float(value) for value in rho],
            "s": [float(value) ** 2 for value in rho],
            "nuPrime": [float(value) for value in nu_v],
            "EStar": [float(value) for value in es_values],
            "grid": {
                "Ntheta": int(grid.n_theta),
                "Nzeta": int(grid.n_zeta),
                "Nxi": int(grid.n_xi),
                "NL": int(rhs2_nl) if rhs_mode == 2 else None,
                "Nx": int(rhs2_nx) if rhs_mode == 2 else 1,
            },
            "solverTolerance": float(solver_tolerance),
            "min_Bmn_to_load": float(min_bmn_to_load),
        },
        "decks": deck_payloads,
        "summary_metrics": {
            "case_count": len({deck.case_id for deck in decks}),
            "deck_count": len(decks),
            "completed_run_count": sum(deck.status == "complete" for deck in decks),
            "failed_run_count": sum(deck.status == "failed" for deck in decks),
            "input_written_count": sum(deck.status == "input_written" for deck in decks),
            "output_found_count": sum(deck.status == "output_found" for deck in decks),
            "completed_transport_matrix_count": len(completed_transport),
            "sfincs_runtime_nu_n_over_input_nuPrime_min": (
                float(np.nanmin(nu_ratios)) if nu_ratios else None
            ),
            "sfincs_runtime_nu_n_over_input_nuPrime_max": (
                float(np.nanmax(nu_ratios)) if nu_ratios else None
            ),
            "completed_ntx_same_grid_comparison_count": sum(
                isinstance(summary.get("ntx_same_grid"), dict)
                and summary["ntx_same_grid"].get("status") == "complete"
                for summary in completed_transport
            ),
            "max_ntx_same_grid_transport_relative_difference": (
                float(np.nanmax(ntx_channel_errors)) if ntx_channel_errors else None
            ),
            "max_ntx_same_grid_transport_spitzer_audit_relative_difference": (
                float(np.nanmax(ntx_spitzer_channel_errors))
                if ntx_spitzer_channel_errors
                else None
            ),
            "rhs_mode": int(rhs_mode),
            "rhsmode2_species": rhsmode2_species if rhs_mode == 2 else None,
        },
        "run_sfincs_jax": bool(run_sfincs_jax),
        "output_dir": str(output_dir),
        "figure_png": str(OUTPUT_PREFIX.with_suffix(".png").relative_to(ROOT)),
        "figure_pdf": str(OUTPUT_PREFIX.with_suffix(".pdf").relative_to(ROOT)),
        "open_work": [
            (
                "expand the current completed stress-radius smoke-resolution "
                "SFINCS-JAX transport-matrix ladder to production radial and "
                "collisionality resolution using the reported nu_n bridge"
            ),
            (
                "connect production SFINCS-JAX profile-current closure diagnostics "
                "to the same finite-beta profile contract"
            ),
            (
                "promote only after the same geometry, profile, normalization, "
                "and interpolation JSON sidecar is complete"
            ),
        ],
    }


def write_payload(payload: dict[str, object], output_prefix: Path = OUTPUT_PREFIX) -> None:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(payload)
    for key, suffix in (("figure_png", ".png"), ("figure_pdf", ".pdf")):
        path = output_prefix.with_suffix(suffix)
        try:
            payload[key] = str(path.relative_to(ROOT))
        except ValueError:
            payload[key] = str(path)
    output_prefix.with_suffix(".json").write_text(json.dumps(payload, indent=2) + "\n")


def build_figure(payload: dict[str, object], output_prefix: Path = OUTPUT_PREFIX) -> None:
    decks = payload["decks"]
    if not decks:
        raise ValueError("no SFINCS-JAX decks to plot")
    plt.style.use("default")
    fig, (ax_grid, ax_status, ax_matrix) = plt.subplots(1, 3, figsize=(14.4, 4.6))

    by_case: dict[str, list[dict[str, object]]] = {}
    for deck in decks:
        by_case.setdefault(str(deck["case_label"]), []).append(deck)
    for label, rows in by_case.items():
        rho_values = sorted({float(row["rho"]) for row in rows})
        nu_values = sorted({float(row["nu_prime"]) for row in rows})
        ax_grid.plot(rho_values, [len(nu_values)] * len(rho_values), marker="o", label=label)
    ax_grid.set_xlabel(r"$\rho$")
    ax_grid.set_ylabel("collisionality points per radius")
    ax_grid.set_title("(a) Same-grid SFINCS-JAX inputs")
    ax_grid.legend(loc="best", fontsize=8)
    ax_grid.grid(alpha=0.25)

    statuses = ["input_written", "output_found", "complete", "failed"]
    counts = [sum(str(deck["status"]) == status for deck in decks) for status in statuses]
    ax_status.bar(statuses, counts, color=["#0072b2", "#56b4e9", "#009e73", "#d55e00"])
    ax_status.set_ylabel("deck count")
    ax_status.set_title("(b) Optional SFINCS-JAX run status")
    ax_status.tick_params(axis="x", rotation=25)
    ax_status.grid(axis="y", alpha=0.25)

    completed = [
        deck
        for deck in decks
        if isinstance(deck.get("transport_summary"), dict)
        and deck["transport_summary"].get("status") == "complete"
    ]
    if completed:
        first = completed[0]
        matrix = np.asarray(first["transport_summary"]["transportMatrix"], dtype=float)
        image = ax_matrix.imshow(matrix, cmap="coolwarm", aspect="equal")
        for (i, j), value in np.ndenumerate(matrix):
            ax_matrix.text(j, i, f"{value:.2e}", ha="center", va="center", fontsize=8)
        ntx_summary = first["transport_summary"].get("ntx_same_grid", {})
        if isinstance(ntx_summary, dict) and ntx_summary.get("status") == "complete":
            max_rel = max(
                float(value)
                for key, value in ntx_summary["relative_difference"].items()
                if not str(key).startswith("L33_spitzer")
                if np.isfinite(float(value))
            )
            ax_matrix.text(
                0.5,
                -0.22,
                f"NTX same-grid max rel. diff. = {max_rel:.2e}",
                ha="center",
                va="top",
                transform=ax_matrix.transAxes,
                fontsize=8.5,
            )
        ax_matrix.set_xticks(range(matrix.shape[1]))
        ax_matrix.set_yticks(range(matrix.shape[0]))
        ax_matrix.set_title("(c) First completed transport matrix")
        ax_matrix.set_xlabel("force column")
        ax_matrix.set_ylabel("flux row")
        fig.colorbar(image, ax=ax_matrix, fraction=0.046, pad=0.04)
    else:
        ax_matrix.text(
            0.5,
            0.5,
            "No completed HDF5 outputs yet",
            ha="center",
            va="center",
            transform=ax_matrix.transAxes,
        )
        ax_matrix.set_title("(c) Output ingestion")
        ax_matrix.set_axis_off()

    fig.suptitle("Owned finite-beta SFINCS-JAX generation contract", fontsize=13)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_prefix.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        help="finite-beta case id to include",
    )
    parser.add_argument(
        "--case-limit",
        type=int,
        default=2,
        help="limit discovered finite-beta cases",
    )
    parser.add_argument("--rho", nargs="+", type=float, default=list(DEFAULT_SFINCS_RHO))
    parser.add_argument("--nu-v", nargs="+", type=float, default=list(DEFAULT_NU_V))
    parser.add_argument("--es", nargs="+", type=float, default=list(DEFAULT_ES))
    parser.add_argument("--n-theta", type=int, default=DEFAULT_GRID.n_theta)
    parser.add_argument("--n-zeta", type=int, default=DEFAULT_GRID.n_zeta)
    parser.add_argument("--n-xi", type=int, default=DEFAULT_GRID.n_xi)
    parser.add_argument("--rhs-mode", type=int, choices=(2, 3), default=3)
    parser.add_argument(
        "--rhsmode2-species",
        choices=tuple(RHSMODE2_SPECIES),
        default="electron",
        help="single species used when --rhs-mode=2",
    )
    parser.add_argument(
        "--rhsmode2-use-profile-contract",
        action="store_true",
        help="write nHats/THats from the owned finite-beta bootstrap profile contract",
    )
    parser.add_argument("--rhs2-nl", type=int, default=DEFAULT_RHS2_NL)
    parser.add_argument("--rhs2-nx", type=int, default=DEFAULT_RHS2_NX)
    parser.add_argument("--solver-tolerance", type=float, default=1.0e-6)
    parser.add_argument("--min-bmn-to-load", type=float, default=1.0e-5)
    parser.add_argument("--run-sfincs-jax", action="store_true", help="execute generated decks")
    parser.add_argument("--timeout-s", type=int, default=300)
    parser.add_argument("--output-prefix", type=Path, default=OUTPUT_PREFIX)
    parser.add_argument("--output-dir", type=Path, default=WORKDIR)
    args = parser.parse_args()

    payload = build_payload(
        case_ids=tuple(args.case),
        case_limit=args.case_limit,
        rho=tuple(args.rho),
        nu_v=tuple(args.nu_v),
        es_values=tuple(args.es),
        grid=GridSpec(args.n_theta, args.n_zeta, args.n_xi),
        output_dir=args.output_dir,
        run_sfincs_jax=bool(args.run_sfincs_jax),
        rhs_mode=int(args.rhs_mode),
        rhsmode2_species=str(args.rhsmode2_species),
        rhsmode2_use_profile_contract=bool(args.rhsmode2_use_profile_contract),
        rhs2_nl=int(args.rhs2_nl),
        rhs2_nx=int(args.rhs2_nx),
        timeout_s=int(args.timeout_s),
        solver_tolerance=float(args.solver_tolerance),
        min_bmn_to_load=float(args.min_bmn_to_load),
    )
    write_payload(payload, args.output_prefix)
    build_figure(payload, args.output_prefix)
    print(
        f"wrote {args.output_prefix.with_suffix('.json')}, "
        f"{args.output_prefix.with_suffix('.png')}, and "
        f"{args.output_prefix.with_suffix('.pdf')}"
    )


if __name__ == "__main__":
    main()
