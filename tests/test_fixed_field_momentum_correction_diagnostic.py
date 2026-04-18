from __future__ import annotations

import numpy as np

from examples.fixed_field_momentum_correction_diagnostic import (
    SONINE_WEIGHTS,
    _candidate_upar_from_solution,
)


def test_candidate_upar_from_solution_modes() -> None:
    solution = np.array([2.0, -5.0, 7.0], dtype=float)

    assert _candidate_upar_from_solution(solution, "c0") == 2.0
    assert np.isclose(
        _candidate_upar_from_solution(solution, "weighted"),
        float(np.dot(SONINE_WEIGHTS, solution)),
    )
    assert _candidate_upar_from_solution(solution, "c2") == 7.0
