from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("f90nml")
pytest.importorskip("NEOPAX")

from examples.fixed_field_momentum_correction_diagnostic import (
    SONINE_WEIGHTS,
    _candidate_upar_from_solution,
    _matrix_coefficients,
    _observable_coefficients,
)


def test_candidate_upar_from_solution_modes() -> None:
    solution = np.array([2.0, -5.0, 7.0], dtype=float)

    assert _candidate_upar_from_solution(solution, "c0") == 2.0
    assert np.isclose(
        _candidate_upar_from_solution(solution, "weighted"),
        float(np.dot(SONINE_WEIGHTS, solution)),
    )
    assert _candidate_upar_from_solution(solution, "c2") == 7.0


def test_matrix_and_observable_coefficients_shapes_and_entries() -> None:
    lij = np.arange(25, dtype=float).reshape(5, 5)
    eij = np.arange(100, 125, dtype=float).reshape(5, 5)

    coeff, nucoeff = _matrix_coefficients(lij, eij)
    obs_coeff, obs_nucoeff = _observable_coefficients(lij, eij)

    assert coeff.shape == (3, 3)
    assert nucoeff.shape == (3, 3)
    assert obs_coeff.shape == (3, 2)
    assert obs_nucoeff.shape == (3, 2)

    assert np.isclose(coeff[0, 0], lij[2, 2])
    assert np.isclose(coeff[0, 2], 4.375 * lij[2, 2] - 3.5 * lij[3, 2] + 0.5 * lij[3, 3])
    assert np.isclose(
        nucoeff[2, 2],
        19.140625 * eij[2, 2]
        - 30.625 * eij[3, 2]
        + 16.625 * eij[3, 3]
        - 3.5 * eij[3, 4]
        + 0.25 * eij[4, 4],
    )
    assert np.isclose(obs_coeff[2, 1], lij[3, 0] - 0.8 * lij[3, 1] + 4.0 * lij[4, 1] / 35.0)
    assert np.isclose(
        obs_nucoeff[2, 0],
        4.375 * eij[2, 0] - 3.5 * eij[2, 1] + 0.5 * eij[3, 1],
    )
