"""Helpers for archived monoenergetic benchmark and regression data."""

from __future__ import annotations

from io import StringIO
from pathlib import Path

import numpy as np

MONOENERGETIC_COLUMNS = (
    "nu_hat",
    "er_hat",
    "n_theta",
    "n_zeta",
    "n_xi",
    "D11",
    "D31",
    "D13",
    "D33",
    "D33_spitzer",
    "wall_time",
    "cpu_time",
)

DKES_RESULT_COLUMNS = (
    "nu_hat",
    "er_hat",
    "omega_hat",
    "b_ref",
    "D11_minus",
    "D11_plus",
    "D31_minus",
    "D31_plus",
    "D33_minus_spitzer",
    "D33_plus_spitzer",
    "v",
    "B0_over_v",
    "lambda_mean_free_path",
    "convergence_metric",
    "iota",
    "psi_scale",
    "unused_0",
    "unused_1",
    "unused_2",
)

TRANSPORT_COLUMNS = ("nu_hat", "er_hat", "D11", "D31", "D33")


def read_monoenergetic_table(path: str | Path) -> np.ndarray:
    """Read a whitespace table with archived monoenergetic coefficients."""

    table = np.genfromtxt(path, skip_header=1, names=MONOENERGETIC_COLUMNS)
    return np.atleast_1d(table)


def read_sfincs_transport_scan(
    path: str | Path,
    *,
    er_hat: float,
    d11_scale: float = 1.0,
    d31_scale: float = 1.0,
) -> np.ndarray:
    """Read an archived SFINCS scan with columns `nu / v, D_11, D_31`."""

    text = Path(path).read_text(encoding="utf-8").replace(",", " ")
    raw = np.genfromtxt(StringIO(text), skip_header=1)
    raw = np.atleast_2d(raw)
    table = np.zeros(raw.shape[0], dtype=_transport_dtype())
    table["nu_hat"] = raw[:, 0]
    table["er_hat"] = er_hat
    table["D11"] = raw[:, 1] * d11_scale
    table["D31"] = raw[:, 2] * d31_scale
    table["D33"] = np.nan
    return table


def read_dkes_transport_scan(
    path: str | Path,
    *,
    d11_scale: float = 1.0,
    d31_scale: float = 1.0,
) -> np.ndarray:
    """Read an archived DKES scan and return physical transport coefficients."""

    raw = np.genfromtxt(path, names=DKES_RESULT_COLUMNS)
    raw = np.atleast_1d(raw)
    table = np.zeros(raw.shape[0], dtype=_transport_dtype())
    table["nu_hat"] = raw["nu_hat"]
    table["er_hat"] = raw["er_hat"]
    table["D11"] = 0.5 * (raw["D11_minus"] + raw["D11_plus"]) * d11_scale
    table["D31"] = 0.5 * (raw["D31_minus"] + raw["D31_plus"]) * d31_scale
    table["D33"] = 0.5 * (raw["D33_minus_spitzer"] + raw["D33_plus_spitzer"])
    return table


def filter_reference_by_er_hat(
    table: np.ndarray,
    er_hat: float,
    *,
    atol: float = 1e-12,
) -> np.ndarray:
    """Return rows with `er_hat` matching the requested electric field."""

    mask = np.isclose(table["er_hat"], er_hat, atol=atol, rtol=0.0)
    return np.atleast_1d(table[mask])


def nearest_reference_row(table: np.ndarray, nu_hat: float, er_hat: float | None = None) -> np.void:
    """Return the row nearest to a requested `(nu_hat, er_hat)` pair."""

    distance = np.abs(np.log10(table["nu_hat"]) - np.log10(nu_hat))
    names = table.dtype.names or ()
    if er_hat is not None and "er_hat" in names:
        distance = distance + np.abs(table["er_hat"] - er_hat)
    return table[int(np.argmin(distance))]


def select_monoenergetic_row(
    table: np.ndarray,
    *,
    nu_hat: float,
    er_hat: float,
    n_theta: int | None = None,
    n_zeta: int | None = None,
    n_xi: int | None = None,
    atol: float = 1e-12,
) -> np.void:
    """Select a monoenergetic reference row, optionally constraining the grid."""

    mask = np.isclose(table["nu_hat"], nu_hat, atol=atol, rtol=0.0)
    mask &= np.isclose(table["er_hat"], er_hat, atol=atol, rtol=0.0)
    if n_theta is not None:
        mask &= np.isclose(table["n_theta"], n_theta, atol=0.0, rtol=0.0)
    if n_zeta is not None:
        mask &= np.isclose(table["n_zeta"], n_zeta, atol=0.0, rtol=0.0)
    if n_xi is not None:
        mask &= np.isclose(table["n_xi"], n_xi, atol=0.0, rtol=0.0)
    matches = np.atleast_1d(table[mask])
    if matches.size == 0:
        requested = {
            "nu_hat": nu_hat,
            "er_hat": er_hat,
            "n_theta": n_theta,
            "n_zeta": n_zeta,
            "n_xi": n_xi,
        }
        msg = f"no monoenergetic row matching {requested}"
        raise ValueError(msg)
    return matches[0]


def coefficient_errors(result: dict[str, float], row: np.void) -> dict[str, float]:
    """Return signed NTX-minus-reference coefficient errors."""

    errors: dict[str, float] = {}
    names = row.dtype.names or ()
    for key in ("D11", "D31", "D13", "D33"):
        if key in names:
            errors[key] = float(result[key] - row[key])
    return errors


def relative_error(value: float, reference: float) -> float:
    """Return a symmetric relative error with finite behavior near zero."""

    scale = max(abs(reference), 1e-30)
    return float(abs(value - reference) / scale)


def _transport_dtype() -> np.dtype:
    return np.dtype([(name, np.float64) for name in TRANSPORT_COLUMNS])
