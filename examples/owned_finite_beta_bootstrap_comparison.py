#!/usr/bin/env python3
"""Compare finite-beta Redl and NTX+NEOPAX bootstrap-current paths.

This example keeps all inputs owned: the same finite-beta VMEC wout is used to
build the Redl geometry factors, the NTX monoenergetic scan, the NEOPAX field,
and the analytic density/temperature profiles.  It is a reduced-grid stress
audit for normalization and interpolation provenance, not an independent-code
parity claim.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from netCDF4 import Dataset  # noqa: E402
from scipy.constants import elementary_charge  # noqa: E402

from examples.owned_geometry_neopax_dataset import (  # noqa: E402
    OwnedJaxGeometryCase,
    _build_scan_for_path,
    _case_psi_p_for_boozer,
    _drds_from_minor_radius,
    discover_owned_case_specs,
)
from ntx import GridSpec, to_neopax_monoenergetic, write_neopax_scan_hdf5  # noqa: E402

OUTPUT_PREFIX = ROOT / "docs" / "_static" / "owned_finite_beta_bootstrap_comparison"
WORKDIR = ROOT / "examples" / "outputs" / "owned_finite_beta_bootstrap_comparison"
DEFAULT_CASE = "finite_beta_qa_pressure_current"
DEFAULT_SCAN_RHO = tuple(np.linspace(0.08, 0.96, 9))
DEFAULT_NU_V = tuple(np.logspace(-6.0, 2.0, 5))
DEFAULT_ES = (0.0, 1.0e-5, 1.0e-4)
DEFAULT_NTX_GRID = GridSpec(25, 31, 24)
DEFAULT_FIELD_RADIAL_POINTS = 15
DEFAULT_NEOPAX_X = 10
DEFAULT_MOMENTUM_ORDERS = (2, 4, 6, 8, 10, 12)
DEFAULT_N_ORDER = 12
DEFAULT_MBOZ = 5
DEFAULT_NBOZ = 5
DEFAULT_REDL_NTHETA = 256
EPS = 1.0e-30


@dataclass(frozen=True)
class ProfileContract:
    density_core_m3: float = 4.0e20
    density_edge_m3: float = 0.4e20
    temperature_core_ev: float = 12.0e3
    temperature_edge_ev: float = 0.5e3
    density_power: int = 5
    temperature_power: int = 1
    zeff: float = 1.0

    def as_payload(self) -> dict[str, float | int]:
        return asdict(self)


def _require_external_stacks() -> tuple[Any, Any, Any, Any, Any]:
    """Import optional local stacks only when the full audit is executed."""

    for path in (
        Path("/Users/rogeriojorge/local/booz_xform_jax/src"),
        Path("/Users/rogeriojorge/local/tests/NEOPAX"),
        Path("/Users/rogeriojorge/local/simsopt/src"),
    ):
        if path.exists() and str(path) not in sys.path:
            sys.path.insert(0, str(path))
    import NEOPAX
    from booz_xform_jax import Booz_xform
    from simsopt.mhd.bootstrap import compute_trapped_fraction, j_dot_B_Redl
    from simsopt.mhd.profiles import ProfileSpline

    return Booz_xform, compute_trapped_fraction, j_dot_B_Redl, ProfileSpline, NEOPAX


def _case_by_id(case_id: str) -> OwnedJaxGeometryCase:
    cases = {case.id: case for case in discover_owned_case_specs()}
    if case_id not in cases:
        raise ValueError(f"owned finite-beta case {case_id!r} was not found")
    return cases[case_id]


def _profile_values(
    rho: np.ndarray,
    contract: ProfileContract,
    *,
    a_b: float,
) -> dict[str, np.ndarray]:
    """Evaluate analytic profile values and radial derivatives."""

    rho_arr = np.asarray(rho, dtype=float)
    s = rho_arr**2
    density = contract.density_edge_m3 + (
        contract.density_core_m3 - contract.density_edge_m3
    ) * (1.0 - s**contract.density_power)
    temperature = contract.temperature_edge_ev + (
        contract.temperature_core_ev - contract.temperature_edge_ev
    ) * (1.0 - s**contract.temperature_power)
    d_density_ds = -(
        contract.density_core_m3 - contract.density_edge_m3
    ) * contract.density_power * s ** (contract.density_power - 1)
    d_temperature_ds = -(
        contract.temperature_core_ev - contract.temperature_edge_ev
    ) * contract.temperature_power * s ** (contract.temperature_power - 1)
    ds_dr = np.where(rho_arr > 0.0, 2.0 * rho_arr / max(float(a_b), EPS), 0.0)
    return {
        "rho": rho_arr,
        "s": s,
        "density": density,
        "temperature": temperature,
        "d_density_ds": d_density_ds,
        "d_temperature_ds": d_temperature_ds,
        "d_density_dr": d_density_ds * ds_dr,
        "d_temperature_dr": d_temperature_ds * ds_dr,
    }


def _profile_splines(contract: ProfileContract, ProfileSpline: Any) -> tuple[Any, Any, Any, float]:
    s_grid = np.linspace(0.0, 1.0, 201)
    values = _profile_values(np.sqrt(s_grid), contract, a_b=1.0)
    density = ProfileSpline(s_grid, values["density"], degree=3)
    temperature = ProfileSpline(s_grid, values["temperature"], degree=3)
    return density, temperature, temperature, float(contract.zeff)


def _run_boozer(case: OwnedJaxGeometryCase, *, mboz: int, nboz: int):
    Booz_xform, *_ = _require_external_stacks()
    bx = Booz_xform()
    bx.verbose = 0
    bx.read_wout(str(case.wout_path))
    bx.mboz = int(mboz)
    bx.nboz = int(nboz)
    bx.run(jit=False)
    return bx


def _write_boozmn(case: OwnedJaxGeometryCase, output_dir: Path, *, mboz: int, nboz: int) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"boozmn_{case.id}_m{mboz}_n{nboz}.nc"
    if path.exists():
        return path
    bx = _run_boozer(case, mboz=mboz, nboz=nboz)
    bx.write_boozmn(str(path))
    return path


def _redl_geometry_and_current(
    case: OwnedJaxGeometryCase,
    *,
    rho: np.ndarray,
    contract: ProfileContract,
    mboz: int,
    nboz: int,
    redl_ntheta: int,
    helicity_n: int,
) -> dict[str, np.ndarray | float | int]:
    _, compute_trapped_fraction, j_dot_B_Redl, ProfileSpline, _ = _require_external_stacks()
    bx = _run_boozer(case, mboz=mboz, nboz=nboz)
    ne, te, ti, zeff = _profile_splines(contract, ProfileSpline)
    s_values = np.asarray(rho, dtype=float) ** 2
    s_b = np.asarray(bx.s_b, dtype=float)
    bmnc_b = np.asarray(bx.bmnc_b, dtype=float)
    gmnc_b = np.asarray(bx.gmnc_b, dtype=float)
    xm_b = np.asarray(bx.xm_b, dtype=int)
    xn_b = np.asarray(bx.xn_b, dtype=int)
    nfp = int(np.asarray(bx.nfp).reshape(()))
    keep = xm_b * int(helicity_n) * nfp == xn_b
    theta = np.linspace(0.0, 2.0 * np.pi, int(redl_ntheta), endpoint=False)
    bmnc = np.vstack([np.interp(s_values, s_b, row) for row in bmnc_b[keep]])
    gmnc = np.vstack([np.interp(s_values, s_b, row) for row in gmnc_b[keep]])
    mod_b = np.zeros((theta.size, s_values.size))
    sqrtg = np.zeros_like(mod_b)
    for m_value, bcoef, gcoef in zip(xm_b[keep], bmnc, gmnc, strict=True):
        phase = np.cos(float(m_value) * theta)[:, None]
        mod_b += phase * bcoef[None, :]
        sqrtg += phase * gcoef[None, :]
    _, _, epsilon, fsa_b2, fsa_1_over_b, trapped_fraction = compute_trapped_fraction(
        mod_b,
        sqrtg,
    )
    boozer_g = np.interp(s_values, s_b, np.asarray(bx.Boozer_G_all, dtype=float))
    boozer_i = np.interp(s_values, s_b, np.asarray(bx.Boozer_I_all, dtype=float))
    iota = np.interp(s_values, s_b, np.asarray(bx.iota, dtype=float))
    with Dataset(case.wout_path, "r") as handle:
        psi_edge = -float(np.asarray(handle.variables["phi"][:], dtype=float)[-1]) / (
            2.0 * np.pi
        )
    current, details = j_dot_B_Redl(
        ne,
        te,
        ti,
        zeff,
        int(helicity_n),
        s=s_values,
        G=boozer_g,
        R=(boozer_g + iota * boozer_i) * fsa_1_over_b,
        iota=iota,
        epsilon=epsilon,
        f_t=trapped_fraction,
        psi_edge=psi_edge,
        nfp=nfp,
    )
    current_arr = np.asarray(current, dtype=float)
    root_fsa_b2 = np.sqrt(np.maximum(np.asarray(fsa_b2, dtype=float), EPS))
    density_term = np.asarray(details.dnds_term, dtype=float)
    electron_temperature_term = np.asarray(details.dTeds_term, dtype=float)
    ion_temperature_term = np.asarray(details.dTids_term, dtype=float)
    temperature_term = electron_temperature_term + ion_temperature_term
    return {
        "rho": np.asarray(rho, dtype=float),
        "s": s_values,
        "jdotb": current_arr,
        "current_over_root_fsab2": current_arr / root_fsa_b2,
        "density_gradient_term": density_term,
        "electron_temperature_gradient_term": electron_temperature_term,
        "ion_temperature_gradient_term": ion_temperature_term,
        "temperature_gradient_term": temperature_term,
        "density_gradient_term_over_root_fsab2": density_term / root_fsa_b2,
        "electron_temperature_gradient_term_over_root_fsab2": (
            electron_temperature_term / root_fsa_b2
        ),
        "ion_temperature_gradient_term_over_root_fsab2": (
            ion_temperature_term / root_fsa_b2
        ),
        "temperature_gradient_term_over_root_fsab2": temperature_term / root_fsa_b2,
        "epsilon": np.asarray(epsilon, dtype=float),
        "trapped_fraction": np.asarray(trapped_fraction, dtype=float),
        "L31": np.asarray(details.L31, dtype=float),
        "L32": np.asarray(details.L32, dtype=float),
        "alpha": np.asarray(details.alpha, dtype=float),
        "nu_e_star": np.asarray(details.nu_e_star, dtype=float),
        "nu_i_star": np.asarray(details.nu_i_star, dtype=float),
        "root_fsab2": root_fsa_b2,
        "psi_edge": float(psi_edge),
        "nfp": int(nfp),
        "helicity_n": int(helicity_n),
    }


def _neopax_current_from_upar(species: Any, upar: np.ndarray, *, species_axis: int) -> np.ndarray:
    """Convert NEOPAX Upar to raw parallel-flow current."""

    charge_qp = np.asarray(species.charge_qp, dtype=float)
    upar_arr = np.asarray(upar, dtype=float)
    if species_axis == 0:
        return np.sum(charge_qp[:, None] * elementary_charge * upar_arr, axis=0)
    if species_axis == 1:
        return np.sum(charge_qp[None, :] * elementary_charge * upar_arr, axis=1)
    raise ValueError("species_axis must be 0 or 1")


def _redl_observable_from_neopax_current(
    raw_parallel_current: np.ndarray,
    b0_over_bbar: np.ndarray,
) -> np.ndarray:
    """Map NEOPAX's raw parallel-flow moment to the Redl current observable."""

    return -np.asarray(b0_over_bbar, dtype=float) * np.asarray(
        raw_parallel_current,
        dtype=float,
    )


def _build_species(NEOPAX: Any, field: Any, contract: ProfileContract):
    rho = np.asarray(field.rho_grid, dtype=float)
    profiles = _profile_values(rho, contract, a_b=float(field.a_b))
    density = profiles["density"]
    temperature = profiles["temperature"]
    density_edge = float(contract.density_edge_m3)
    temperature_edge = float(contract.temperature_edge_ev)
    return NEOPAX.Species(
        2,
        int(field.n_r),
        np.asarray([0, 1], dtype=int),
        np.asarray([1.0 / 1836.15267343, 2.0], dtype=float),
        np.asarray([-1.0, 1.0], dtype=float),
        np.stack([temperature, temperature]),
        np.stack([density, density]),
        np.zeros_like(rho),
        field.r_grid,
        field.r_grid_half,
        field.dr,
        field.Vprime_half,
        field.overVprime,
        np.asarray([density_edge, density_edge], dtype=float),
        np.asarray([temperature_edge, temperature_edge], dtype=float),
        dTdr_override=np.stack(
            [profiles["d_temperature_dr"], profiles["d_temperature_dr"]]
        ),
        dndr_override=np.stack([profiles["d_density_dr"], profiles["d_density_dr"]]),
    )


def _adaptive_nu_values(
    NEOPAX: Any,
    species: Any,
    field: Any,
    *,
    neopax_x: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Build the physical `nu/v` support sampled by the NEOPAX convolution."""

    from NEOPAX._species import collisionality

    grid = NEOPAX.Grid.create_standard(int(field.n_r), int(neopax_x), 2, n_order=2)
    support: dict[str, Any] = {}
    positive_values: list[np.ndarray] = []
    for species_index, label in enumerate(("electron", "ion")):
        samples: list[np.ndarray] = []
        for radial_index, thermal_speed in enumerate(
            np.asarray(species.v_thermal[species_index], dtype=float)
        ):
            velocity = np.asarray(grid.v_norm, dtype=float) * float(thermal_speed)
            nu_over_v = np.asarray(
                collisionality(species_index, species, velocity, radial_index) / velocity,
                dtype=float,
            )
            finite_positive = nu_over_v[np.isfinite(nu_over_v) & (nu_over_v > 0.0)]
            if finite_positive.size:
                samples.append(finite_positive)
                positive_values.append(finite_positive)
        if samples:
            merged = np.concatenate(samples)
            support[label] = {
                "min": float(np.min(merged)),
                "max": float(np.max(merged)),
                "q10": float(np.quantile(merged, 0.10)),
                "median": float(np.quantile(merged, 0.50)),
                "q90": float(np.quantile(merged, 0.90)),
            }
        else:
            support[label] = {"min": None, "max": None}
    if not positive_values:
        raise ValueError("could not build positive finite collisionality support")
    merged_all = np.concatenate(positive_values)
    nu_min = max(float(np.min(merged_all)) / 3.0, 1.0e-8)
    nu_max = max(float(np.max(merged_all)) * 3.0, nu_min * 10.0)
    values = np.logspace(np.log10(nu_min), np.log10(nu_max), 17)
    support["axis"] = {
        "min": float(values[0]),
        "max": float(values[-1]),
        "count": int(values.size),
        "policy": "logspace from min(nu/v)/3 to max(nu/v)*3 over the sampled convolution support",
    }
    return values, support


def _evaluate_neopax_currents(
    NEOPAX: Any,
    *,
    species: Any,
    field: Any,
    database: Any,
    neopax_x: int,
    n_order: int,
) -> dict[str, Any]:
    """Evaluate no-momentum and total NEOPAX current observables for one order."""

    neopax_grid = NEOPAX.Grid.create_standard(
        int(field.n_r),
        int(neopax_x),
        2,
        n_order=int(n_order),
    )
    lij_nomom, _, _, upar_nomom = NEOPAX.get_Neoclassical_Fluxes(
        species,
        neopax_grid,
        field,
        database,
    )
    current_nomom_raw_species = np.asarray(
        species.charge_qp[:, None] * elementary_charge * np.asarray(upar_nomom, dtype=float),
        dtype=float,
    )
    current_nomom_raw = np.sum(current_nomom_raw_species, axis=0)
    _, _, upar_total, _, _ = NEOPAX.get_Neoclassical_Fluxes_With_Momentum_Correction(
        species,
        neopax_grid,
        field,
        database,
    )
    upar_total_arr = np.asarray(upar_total, dtype=float)
    current_total_raw_species = np.asarray(
        species.charge_qp[None, :] * elementary_charge * upar_total_arr,
        dtype=float,
    ).T
    current_total_raw = np.sum(current_total_raw_species, axis=0)
    b0_over_bbar = np.asarray(np.abs(field.B0), dtype=float)
    current_nomom = _redl_observable_from_neopax_current(
        current_nomom_raw,
        b0_over_bbar,
    )
    current_total = _redl_observable_from_neopax_current(
        current_total_raw,
        b0_over_bbar,
    )
    current_nomom_species = np.asarray(
        [
            _redl_observable_from_neopax_current(row, b0_over_bbar)
            for row in current_nomom_raw_species
        ],
        dtype=float,
    )
    current_total_species = np.asarray(
        [
            _redl_observable_from_neopax_current(row, b0_over_bbar)
            for row in current_total_raw_species
        ],
        dtype=float,
    )
    root_fsab2 = np.asarray(np.abs(field.B0) * np.sqrt(np.abs(field.Bsqav)), dtype=float)
    return {
        "n_order": int(n_order),
        "neopax_x": int(neopax_x),
        "lij_nomom": lij_nomom,
        "current_nomom_raw_parallel_species": current_nomom_raw_species,
        "current_total_raw_parallel_species": current_total_raw_species,
        "current_nomom_raw_parallel": current_nomom_raw,
        "current_total_raw_parallel": current_total_raw,
        "current_nomom_species": current_nomom_species,
        "current_total_species": current_total_species,
        "current_nomom": np.asarray(current_nomom, dtype=float),
        "current_total": np.asarray(current_total, dtype=float),
        "current_correction": np.asarray(current_total - current_nomom, dtype=float),
        "current_nomom_over_root_fsab2": np.asarray(current_nomom, dtype=float)
        / np.maximum(root_fsab2, EPS),
        "current_total_over_root_fsab2": np.asarray(current_total, dtype=float)
        / np.maximum(root_fsab2, EPS),
        "current_correction_over_root_fsab2": np.asarray(current_total - current_nomom, dtype=float)
        / np.maximum(root_fsab2, EPS),
        "root_fsab2": root_fsab2,
        "redl_observable_bridge": (-b0_over_bbar).tolist(),
    }


def _ntx_neopax_current(
    case: OwnedJaxGeometryCase,
    *,
    scan_rho: np.ndarray,
    nu_v: np.ndarray | None,
    es_values: np.ndarray,
    contract: ProfileContract,
    output_dir: Path,
    ntx_grid: GridSpec,
    field_radial_points: int,
    neopax_x: int,
    n_order: int,
    d33_mode: str,
    mboz: int,
    nboz: int,
    min_bmn_to_load: float,
    write_hdf5: bool,
    adaptive_nu: bool,
    momentum_orders: tuple[int, ...],
) -> dict[str, Any]:
    *_, NEOPAX = _require_external_stacks()
    boozmn_path = _write_boozmn(case, output_dir, mboz=mboz, nboz=nboz)
    field = NEOPAX.Field.read_vmec_booz(
        int(field_radial_points),
        str(case.wout_path),
        str(boozmn_path),
    )
    species = _build_species(NEOPAX, field, contract)
    nu_support = None
    if adaptive_nu or nu_v is None or np.asarray(nu_v).size == 0:
        nu_v, nu_support = _adaptive_nu_values(
            NEOPAX,
            species,
            field,
            neopax_x=neopax_x,
        )
    nu_v = np.asarray(nu_v, dtype=float)
    start = time.perf_counter()
    drds = _drds_from_minor_radius(scan_rho, float(field.a_b))
    scan = _build_scan_for_path(
        case,
        rho=scan_rho,
        nu_v=nu_v,
        es_values=es_values,
        drds=drds,
        grid=ntx_grid,
        path_key="booz_xform_jax",
        mboz=mboz,
        nboz=nboz,
        min_bmn_to_load=min_bmn_to_load,
    )
    scan_seconds = time.perf_counter() - start
    hdf5_payload: dict[str, object] | None = None
    if write_hdf5:
        path = output_dir / f"{case.id}_bootstrap_scan.h5"
        write_neopax_scan_hdf5(scan, path)
        hdf5_payload = {"path": str(path), "status": "written"}
    database = to_neopax_monoenergetic(scan, a_b=float(field.a_b), d33_mode=d33_mode)
    closure = _evaluate_neopax_currents(
        NEOPAX,
        species=species,
        field=field,
        database=database,
        neopax_x=neopax_x,
        n_order=n_order,
    )
    order_scan: dict[str, Any] = {}
    for order in momentum_orders:
        if int(order) == int(n_order):
            order_closure = closure
        else:
            order_closure = _evaluate_neopax_currents(
                NEOPAX,
                species=species,
                field=field,
                database=database,
                neopax_x=neopax_x,
                n_order=int(order),
            )
        order_scan[str(int(order))] = {
            "n_order": int(order),
            "current_total_over_root_fsab2": np.asarray(
                order_closure["current_total_over_root_fsab2"],
                dtype=float,
            ),
            "current_nomom_over_root_fsab2": np.asarray(
                order_closure["current_nomom_over_root_fsab2"],
                dtype=float,
            ),
        }
    root_fsab2 = np.asarray(closure["root_fsab2"], dtype=float)
    rho_field = np.asarray(field.rho_grid, dtype=float)
    profiles = _profile_values(rho_field, contract, a_b=float(field.a_b))
    lij_nomom = closure["lij_nomom"]
    return {
        "rho": rho_field,
        "current_nomom_raw_parallel_species": np.asarray(
            closure["current_nomom_raw_parallel_species"],
            dtype=float,
        ),
        "current_total_raw_parallel_species": np.asarray(
            closure["current_total_raw_parallel_species"],
            dtype=float,
        ),
        "current_nomom_raw_parallel": np.asarray(
            closure["current_nomom_raw_parallel"],
            dtype=float,
        ),
        "current_total_raw_parallel": np.asarray(
            closure["current_total_raw_parallel"],
            dtype=float,
        ),
        "current_nomom_species": np.asarray(closure["current_nomom_species"], dtype=float),
        "current_total_species": np.asarray(closure["current_total_species"], dtype=float),
        "current_nomom": np.asarray(closure["current_nomom"], dtype=float),
        "current_total": np.asarray(closure["current_total"], dtype=float),
        "current_correction": np.asarray(closure["current_correction"], dtype=float),
        "current_nomom_over_root_fsab2": np.asarray(
            closure["current_nomom_over_root_fsab2"],
            dtype=float,
        ),
        "current_total_over_root_fsab2": np.asarray(
            closure["current_total_over_root_fsab2"],
            dtype=float,
        ),
        "current_correction_over_root_fsab2": np.asarray(
            closure["current_correction_over_root_fsab2"],
            dtype=float,
        ),
        "root_fsab2": root_fsab2,
        "density": profiles["density"],
        "temperature": profiles["temperature"],
        "A1_electron": np.asarray(species.A1[0], dtype=float),
        "A2_electron": np.asarray(species.A2[0], dtype=float),
        "L31_electron": np.asarray(lij_nomom[0, :, 2, 0], dtype=float),
        "L32_electron": np.asarray(lij_nomom[0, :, 2, 1], dtype=float),
        "drds": drds,
        "nu_v": nu_v,
        "nu_support": nu_support,
        "momentum_order_scan": order_scan,
        "redl_observable_bridge": closure["redl_observable_bridge"],
        "scan_seconds": float(scan_seconds),
        "field_a_b": float(field.a_b),
        "booz_xform_psi_p": float(_case_psi_p_for_boozer(case)),
        "boozmn_path": str(boozmn_path),
        "hdf5": hdf5_payload,
    }


def _interp(x: np.ndarray, y: np.ndarray, x_new: np.ndarray) -> np.ndarray:
    return np.interp(
        np.asarray(x_new, dtype=float),
        np.asarray(x, dtype=float),
        np.asarray(y, dtype=float),
    )


def _relative_error(reference: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    return np.abs(candidate - reference) / np.maximum(np.abs(reference), EPS)


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {key: _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def build_payload(
    *,
    case_id: str = DEFAULT_CASE,
    scan_rho: tuple[float, ...] = DEFAULT_SCAN_RHO,
    nu_v: tuple[float, ...] = DEFAULT_NU_V,
    es_values: tuple[float, ...] = DEFAULT_ES,
    contract: ProfileContract | None = None,
    output_dir: Path = WORKDIR,
    ntx_grid: GridSpec = DEFAULT_NTX_GRID,
    field_radial_points: int = DEFAULT_FIELD_RADIAL_POINTS,
    neopax_x: int = DEFAULT_NEOPAX_X,
    n_order: int = DEFAULT_N_ORDER,
    d33_mode: str = "spitzer",
    mboz: int = DEFAULT_MBOZ,
    nboz: int = DEFAULT_NBOZ,
    redl_ntheta: int = DEFAULT_REDL_NTHETA,
    helicity_n: int = 0,
    min_bmn_to_load: float = 1.0e-5,
    write_hdf5: bool = True,
    adaptive_nu: bool = True,
    momentum_orders: tuple[int, ...] = DEFAULT_MOMENTUM_ORDERS,
) -> dict[str, Any]:
    if contract is None:
        contract = ProfileContract()
    case = _case_by_id(case_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    scan_rho_arr = np.asarray(scan_rho, dtype=float)
    nu_arr = np.asarray(nu_v, dtype=float)
    es_arr = np.asarray(es_values, dtype=float)
    ntx = _ntx_neopax_current(
        case,
        scan_rho=scan_rho_arr,
        nu_v=nu_arr,
        es_values=es_arr,
        contract=contract,
        output_dir=output_dir,
        ntx_grid=ntx_grid,
        field_radial_points=field_radial_points,
        neopax_x=neopax_x,
        n_order=n_order,
        d33_mode=d33_mode,
        mboz=mboz,
        nboz=nboz,
        min_bmn_to_load=min_bmn_to_load,
        write_hdf5=write_hdf5,
        adaptive_nu=adaptive_nu,
        momentum_orders=tuple(momentum_orders),
    )
    nu_arr = np.asarray(ntx["nu_v"], dtype=float)
    rho_compare = np.asarray(ntx["rho"], dtype=float)[1:-1]
    redl = _redl_geometry_and_current(
        case,
        rho=rho_compare,
        contract=contract,
        mboz=mboz,
        nboz=nboz,
        redl_ntheta=redl_ntheta,
        helicity_n=helicity_n,
    )
    redl_current = np.asarray(redl["current_over_root_fsab2"], dtype=float)
    ntx_nomom = _interp(
        np.asarray(ntx["rho"], dtype=float),
        np.asarray(ntx["current_nomom_over_root_fsab2"], dtype=float),
        rho_compare,
    )
    ntx_total = _interp(
        np.asarray(ntx["rho"], dtype=float),
        np.asarray(ntx["current_total_over_root_fsab2"], dtype=float),
        rho_compare,
    )
    summary_metrics = {
        "max_relative_error_total_vs_redl_interior": float(
            np.nanmax(_relative_error(redl_current, ntx_total))
        ),
        "max_relative_error_nomom_vs_redl_interior": float(
            np.nanmax(_relative_error(redl_current, ntx_nomom))
        ),
        "rms_relative_error_total_vs_redl_interior": float(
            np.sqrt(np.nanmean(_relative_error(redl_current, ntx_total) ** 2))
        ),
        "sign_agreement_fraction_total": float(
            np.mean(np.sign(redl_current) == np.sign(ntx_total))
        ),
    }
    order_scan: dict[str, Any] = {}
    for key, scan_entry in dict(ntx.get("momentum_order_scan", {})).items():
        order_total = _interp(
            np.asarray(ntx["rho"], dtype=float),
            np.asarray(scan_entry["current_total_over_root_fsab2"], dtype=float),
            rho_compare,
        )
        order_rel = _relative_error(redl_current, order_total)
        order_scan[str(key)] = {
            "n_order": int(scan_entry["n_order"]),
            "ntx_neopax_total_over_root_fsab2": order_total,
            "relative_error_total_vs_redl": order_rel,
            "max_relative_error_total_vs_redl": float(np.nanmax(order_rel)),
            "rms_relative_error_total_vs_redl": float(np.sqrt(np.nanmean(order_rel**2))),
            "sign_agreement_fraction_total": float(
                np.mean(np.sign(redl_current) == np.sign(order_total))
            ),
        }
    payload = {
        "benchmark": "owned_finite_beta_bootstrap_comparison",
        "classification": "owned finite-beta bootstrap-current stress audit",
        "claim_scope": (
            "Runs Redl and NTX+NEOPAX on the same finite-beta VMEC wout, Boozer "
            "transform, analytic profiles, radial grid, and current normalization. "
            "The Boozer path uses the physical VMEC edge-flux scale, the NTX scan "
            "covers the physical nu/v support sampled by the profile convolution, "
            "and the closure sidecar records Sonine-order convergence with an "
            "explicit D33_spitzer branch. This is a production-resolution "
            "reduced-closure stress audit "
            "and should not be promoted as SFINCS parity until same-grid SFINCS-JAX "
            "profile-current closure diagnostics are complete."
        ),
        "case": case.as_payload(),
        "profile_contract": contract.as_payload(),
        "normalization_contract": {
            "redl_observable": "<J dot B> / sqrt(<B^2>)",
            "neopax_observable": (
                "-(B0/Bbar) * elementary_charge * sum_s Z_s Upar_s / sqrt(<B^2>)"
            ),
            "charge_conversion": (
                "Upar is multiplied by one elementary-charge factor and the "
                "dimensionless species charge Z_s, matching the NEOPAX examples."
            ),
            "parallel_flow_bridge": (
                "The raw NEOPAX parallel-flow current is multiplied by -B0/Bbar "
                "before comparison with the Redl current observable, matching "
                "the fixed-field SFINCS/Redl validation convention."
            ),
            "booz_xform_psi_p": (
                "The Boozer-coordinate NTX path passes abs(phi_edge)/(2*pi) "
                "from the matching VMEC wout as psi_p instead of using the "
                "low-level unit-flux default."
            ),
        },
        "inputs": {
            "scan_rho": scan_rho_arr.tolist(),
            "nu_v": nu_arr.tolist(),
            "nu_v_policy": (
                "adaptive physical nu/v support from the NEOPAX velocity convolution"
                if adaptive_nu
                else "user-supplied nu/v support"
            ),
            "nu_support": ntx["nu_support"],
            "Es": es_arr.tolist(),
            "drds": _drds_from_minor_radius(scan_rho_arr, float(ntx["field_a_b"])).tolist(),
            "drds_definition": "dr/ds = a_b/(2*rho), with s=rho^2 and r=a_b*rho",
            "ntx_grid": {
                "n_theta": int(ntx_grid.n_theta),
                "n_zeta": int(ntx_grid.n_zeta),
                "n_xi": int(ntx_grid.n_xi),
            },
            "field_radial_points": int(field_radial_points),
            "neopax_x": int(neopax_x),
            "n_order": int(n_order),
            "d33_mode": d33_mode,
            "momentum_orders": [int(order) for order in momentum_orders],
            "mboz": int(mboz),
            "nboz": int(nboz),
            "redl_ntheta": int(redl_ntheta),
            "helicity_n": int(helicity_n),
            "min_bmn_to_load": float(min_bmn_to_load),
        },
        "ntx_neopax": ntx,
        "redl": redl,
        "comparison": {
            "rho": rho_compare,
            "redl_current_over_root_fsab2": redl_current,
            "ntx_neopax_nomom_over_root_fsab2": ntx_nomom,
            "ntx_neopax_total_over_root_fsab2": ntx_total,
            "relative_error_nomom_vs_redl": _relative_error(redl_current, ntx_nomom),
            "relative_error_total_vs_redl": _relative_error(redl_current, ntx_total),
            "momentum_order_scan": order_scan,
        },
        "summary_metrics": summary_metrics,
        "open_work": [
            (
                "connect completed same-grid SFINCS-JAX coefficient ladders to this "
                "finite-beta profile contract"
            ),
            (
                "close the inner-radius reduced-closure gap using the same physical "
                "profile, normalization, and interpolation contract before promoting parity"
            ),
            (
                "extend the production-resolution QA ladder to QH/QI and W7-X-owned "
                "families before promoting broad finite-beta current-profile claims"
            ),
            (
                "add downstream general-vs-legacy interpolation comparison once NEOPAX "
                "exposes a stable mode selector"
            ),
        ],
        "figure_png": str(OUTPUT_PREFIX.with_suffix(".png").relative_to(ROOT)),
        "figure_pdf": str(OUTPUT_PREFIX.with_suffix(".pdf").relative_to(ROOT)),
    }
    return _to_jsonable(payload)


def write_payload(payload: dict[str, Any], output_prefix: Path = OUTPUT_PREFIX) -> None:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    output_prefix.with_suffix(".json").write_text(json.dumps(payload, indent=2) + "\n")


def build_figure(payload: dict[str, Any], output_prefix: Path = OUTPUT_PREFIX) -> None:
    rho_field = np.asarray(payload["ntx_neopax"]["rho"], dtype=float)
    density = np.asarray(payload["ntx_neopax"]["density"], dtype=float)
    temperature = np.asarray(payload["ntx_neopax"]["temperature"], dtype=float)
    rho = np.asarray(payload["comparison"]["rho"], dtype=float)
    redl = np.asarray(payload["comparison"]["redl_current_over_root_fsab2"], dtype=float) / 1.0e6
    nomom = (
        np.asarray(payload["comparison"]["ntx_neopax_nomom_over_root_fsab2"], dtype=float)
        / 1.0e6
    )
    total = (
        np.asarray(payload["comparison"]["ntx_neopax_total_over_root_fsab2"], dtype=float)
        / 1.0e6
    )
    rel_total = np.asarray(payload["comparison"]["relative_error_total_vs_redl"], dtype=float)
    order_scan = payload["comparison"].get("momentum_order_scan", {})
    epsilon = np.asarray(payload["redl"]["epsilon"], dtype=float)
    trapped = np.asarray(payload["redl"]["trapped_fraction"], dtype=float)
    l31 = np.asarray(payload["redl"]["L31"], dtype=float)
    l32 = np.asarray(payload["redl"]["L32"], dtype=float)

    plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.dpi": 220,
            "font.size": 10.0,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(12.6, 8.0), constrained_layout=True)

    axes[0, 0].plot(rho_field, density / 1.0e20, color="#0072b2", lw=2.1, label=r"$n_e$")
    ax_t = axes[0, 0].twinx()
    ax_t.plot(rho_field, temperature / 1.0e3, color="#d55e00", lw=1.8, ls="--", label=r"$T$")
    axes[0, 0].set_title("(a) Shared finite-beta profiles")
    axes[0, 0].set_xlabel(r"$\rho$")
    axes[0, 0].set_ylabel(r"$n_e$ [$10^{20}$ m$^{-3}$]")
    ax_t.set_ylabel(r"$T$ [keV]")

    axes[0, 1].plot(rho, epsilon, color="#009e73", lw=2.0, marker="o", label=r"$\epsilon$")
    axes[0, 1].plot(rho, trapped, color="#cc79a7", lw=1.8, marker="s", label=r"$f_t$")
    axes[0, 1].plot(rho, l31, color="#111111", lw=1.5, ls="--", label=r"$L_{31}^{Redl}$")
    axes[0, 1].plot(rho, l32, color="#666666", lw=1.5, ls=":", label=r"$L_{32}^{Redl}$")
    axes[0, 1].set_title("(b) Redl geometry and closure factors")
    axes[0, 1].set_xlabel(r"$\rho$")
    axes[0, 1].legend(fontsize=8.2)

    axes[1, 0].plot(rho, redl, color="#009e73", lw=2.2, label="Redl")
    axes[1, 0].plot(rho, nomom, color="#0072b2", lw=1.8, ls="--", label="NTX+NEOPAX no momentum")
    axes[1, 0].plot(rho, total, color="#d55e00", lw=1.8, label="NTX+NEOPAX total")
    axes[1, 0].axhline(0.0, color="0.3", lw=0.8)
    axes[1, 0].set_title("(c) Same-grid bootstrap-current observable")
    axes[1, 0].set_xlabel(r"$\rho$")
    axes[1, 0].set_ylabel(r"$\langle J\cdot B\rangle/\sqrt{\langle B^2\rangle}$ [MA m$^{-2}$]")
    axes[1, 0].legend(fontsize=8.0)

    if isinstance(order_scan, dict) and order_scan:
        rows = sorted(order_scan.values(), key=lambda item: int(item["n_order"]))
        orders = np.asarray([int(row["n_order"]) for row in rows], dtype=int)
        max_error = np.asarray(
            [float(row["max_relative_error_total_vs_redl"]) for row in rows],
            dtype=float,
        )
        rms_error = np.asarray(
            [float(row["rms_relative_error_total_vs_redl"]) for row in rows],
            dtype=float,
        )
        axes[1, 1].semilogy(
            orders,
            max_error,
            color="#d55e00",
            lw=2.0,
            marker="o",
            label="max",
        )
        axes[1, 1].semilogy(
            orders,
            rms_error,
            color="#0072b2",
            lw=1.8,
            marker="s",
            label="RMS",
        )
        axes[1, 1].set_xticks(orders)
        axes[1, 1].set_title("(d) Momentum-closure convergence")
        axes[1, 1].set_xlabel("Sonine order")
        axes[1, 1].set_ylabel("relative difference")
        axes[1, 1].legend(fontsize=8.2)
    else:
        axes[1, 1].semilogy(rho, rel_total, color="#d55e00", lw=2.0, marker="o")
        axes[1, 1].set_title("(d) Stress-gap monitor")
        axes[1, 1].set_xlabel(r"$\rho$")
        axes[1, 1].set_ylabel("relative difference")
    axes[1, 1].axhline(1.0e-1, color="0.25", lw=1.0, ls="--", label=r"$10^{-1}$")

    fig.suptitle("Owned finite-beta bootstrap-current comparison", fontsize=13)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_prefix.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", default=DEFAULT_CASE)
    parser.add_argument("--rho", nargs="+", type=float, default=list(DEFAULT_SCAN_RHO))
    parser.add_argument("--nu-v", nargs="+", type=float, default=list(DEFAULT_NU_V))
    parser.add_argument("--es", nargs="+", type=float, default=list(DEFAULT_ES))
    parser.add_argument("--n-theta", type=int, default=DEFAULT_NTX_GRID.n_theta)
    parser.add_argument("--n-zeta", type=int, default=DEFAULT_NTX_GRID.n_zeta)
    parser.add_argument("--n-xi", type=int, default=DEFAULT_NTX_GRID.n_xi)
    parser.add_argument("--field-radial-points", type=int, default=DEFAULT_FIELD_RADIAL_POINTS)
    parser.add_argument("--neopax-x", type=int, default=DEFAULT_NEOPAX_X)
    parser.add_argument("--n-order", type=int, default=DEFAULT_N_ORDER)
    parser.add_argument("--d33-mode", default="spitzer")
    parser.add_argument(
        "--fixed-nu-v",
        action="store_true",
        help="use the --nu-v values directly instead of adaptive physical nu/v support",
    )
    parser.add_argument(
        "--momentum-orders",
        nargs="+",
        type=int,
        default=list(DEFAULT_MOMENTUM_ORDERS),
        help="Sonine orders to include in the convergence sidecar",
    )
    parser.add_argument("--mboz", type=int, default=DEFAULT_MBOZ)
    parser.add_argument("--nboz", type=int, default=DEFAULT_NBOZ)
    parser.add_argument("--redl-ntheta", type=int, default=DEFAULT_REDL_NTHETA)
    parser.add_argument("--helicity-n", type=int, default=0)
    parser.add_argument("--min-bmn-to-load", type=float, default=1.0e-5)
    parser.add_argument("--no-hdf5", action="store_true")
    parser.add_argument("--output-prefix", type=Path, default=OUTPUT_PREFIX)
    parser.add_argument("--output-dir", type=Path, default=WORKDIR)
    args = parser.parse_args()

    payload = build_payload(
        case_id=str(args.case),
        scan_rho=tuple(args.rho),
        nu_v=tuple(args.nu_v),
        es_values=tuple(args.es),
        output_dir=args.output_dir,
        ntx_grid=GridSpec(args.n_theta, args.n_zeta, args.n_xi),
        field_radial_points=int(args.field_radial_points),
        neopax_x=int(args.neopax_x),
        n_order=int(args.n_order),
        d33_mode=str(args.d33_mode),
        mboz=int(args.mboz),
        nboz=int(args.nboz),
        redl_ntheta=int(args.redl_ntheta),
        helicity_n=int(args.helicity_n),
        min_bmn_to_load=float(args.min_bmn_to_load),
        write_hdf5=not bool(args.no_hdf5),
        adaptive_nu=not bool(args.fixed_nu_v),
        momentum_orders=tuple(int(value) for value in args.momentum_orders),
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
