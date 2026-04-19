from __future__ import annotations

import numpy as np

from examples.momentum_correction_mapping_audit import (
    Sample,
    _candidate_current,
    _evaluate_weights,
    _fit_species_weights,
)


def test_fit_species_weights_recovers_linear_map() -> None:
    weights_true = np.array([2.0, -1.0, 0.5], dtype=float)
    samples = [
        Sample(
            dataset="synthetic",
            case="synth",
            rho=float(i),
            species="electron",
            density=1.0 + 0.2 * i,
            solution=np.array([1.0 + i, 2.0 - 0.5 * i, -1.0 + 0.25 * i], dtype=float),
            current_nomom=17.0 + i,
            current_reference=_candidate_current(
                density=1.0 + 0.2 * i,
                solution=np.array([1.0 + i, 2.0 - 0.5 * i, -1.0 + 0.25 * i], dtype=float),
                charge_sign=-1.0,
                weights=weights_true,
            ),
        )
        for i in range(4)
    ]
    fitted = _fit_species_weights(samples)
    np.testing.assert_allclose(fitted, weights_true, rtol=1e-12, atol=1e-12)


def test_evaluate_weights_reports_small_error_for_exact_weights() -> None:
    weights_true = np.array([1.0, 0.25, -0.1], dtype=float)
    sample = Sample(
        dataset="synthetic",
        case="synth",
        rho=0.5,
        species="ion",
        density=3.0,
        solution=np.array([4.0, -2.0, 1.0], dtype=float),
        current_nomom=7.0,
        current_reference=_candidate_current(
            density=3.0,
            solution=np.array([4.0, -2.0, 1.0], dtype=float),
            charge_sign=1.0,
            weights=weights_true,
        ),
    )
    result = _evaluate_weights([sample], weights_true)
    assert result["max_relative_error"] < 1.0e-12
