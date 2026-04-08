"""External benchmark helpers."""

from __future__ import annotations

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


def read_monoenergetic_table(path: str | Path) -> np.ndarray:
    """Read a whitespace table with REFERENCE_EXECUTABLE-style monoenergetic coefficients."""

    table = np.genfromtxt(path, skip_header=1, names=MONOENERGETIC_COLUMNS)
    return np.atleast_1d(table)


def nearest_reference_row(table: np.ndarray, nu_hat: float, er_hat: float) -> np.void:
    """Return the row nearest to a requested `(nu_hat, er_hat)` pair."""

    distance = np.abs(np.log10(table["nu_hat"]) - np.log10(nu_hat)) + np.abs(
        table["er_hat"] - er_hat
    )
    return table[int(np.argmin(distance))]


def coefficient_errors(result: dict[str, float], row: np.void) -> dict[str, float]:
    """Return signed NTX-minus-reference coefficient errors."""

    return {key: float(result[key] - row[key]) for key in ("D11", "D31", "D13", "D33")}
