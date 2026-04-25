from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("f90nml")
pytest.importorskip("NEOPAX")

from NEOPAX._moments import build_transport_projection, build_transport_source_columns

from examples.fixed_field_momentum_correction_diagnostic import (
    SONINE_WEIGHTS,
    _candidate_upar_from_solution,
    _dump_forensic_summary,
    _force_current_contributions,
    _full_observable_coefficient_matrix,
    _matrix_coefficients,
    _observable_coefficients,
    _rhs_force_contributions,
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
    full_obs_coeff, full_obs_nucoeff = _full_observable_coefficient_matrix(lij, eij)

    assert coeff.shape == (3, 3)
    assert nucoeff.shape == (3, 3)
    assert obs_coeff.shape == (3, 2)
    assert obs_nucoeff.shape == (3, 2)
    assert full_obs_coeff.shape == (3, 3)
    assert full_obs_nucoeff.shape == (3, 3)

    assert np.isclose(coeff[0, 0], lij[2, 2])
    np.testing.assert_allclose(coeff, build_transport_projection(lij[2:5, 2:5], 3))
    assert np.isfinite(nucoeff).all()
    np.testing.assert_allclose(
        obs_coeff,
        build_transport_source_columns(
            np.stack([-lij[0, 2:5], -lij[1, 2:5]], axis=1),
            3,
        ),
    )
    assert np.isfinite(obs_nucoeff).all()
    np.testing.assert_allclose(full_obs_coeff[:, :2], obs_coeff)
    np.testing.assert_allclose(full_obs_nucoeff[:, :2], obs_nucoeff)
    np.testing.assert_allclose(full_obs_coeff[:, 2], 0.0)
    np.testing.assert_allclose(full_obs_nucoeff[:, 2], 0.0)


def test_force_and_rhs_forensics_decompose_drive_channels() -> None:
    lij = np.zeros((5, 5), dtype=float)
    lij[2, :3] = [2.0, -3.0, 5.0]
    lij[3, :3] = [7.0, 11.0, -13.0]
    lij[4, :3] = [17.0, -19.0, 23.0]
    forces = np.array([0.5, -2.0, 0.0], dtype=float)

    force = _force_current_contributions(
        lij_block=lij,
        forces=forces,
        density=4.0,
        charge_sign=-1.0,
    )
    rhs = _rhs_force_contributions(lij, forces)

    assert force["current_sum"] != 0.0
    assert force["dominant_current_force"] == "A2_temperature"
    assert force["density_electric_force"] == pytest.approx(-2.5)
    assert force["thermal_effective_coefficient"] == pytest.approx(-6.0)
    assert force["effective_current_sum"] == pytest.approx(force["current_sum"])
    assert force["dominant_effective_current_force"] == "effective_temperature_force"
    np.testing.assert_allclose(rhs["rhs_by_moment"], np.asarray([-7.0, -36.0, -118.625]))
    np.testing.assert_allclose(rhs["effective_rhs_by_moment"], rhs["rhs_by_moment"])
    assert rhs["dominant_rhs_force"] == "A2_temperature"
    assert rhs["dominant_effective_rhs_force"] == "effective_temperature_force"


def test_dump_forensic_summary_classifies_sign_gap() -> None:
    species_template = {
        "current_nomom": 1.0,
        "current_total": -2.0,
        "current_solution_c0": -2.0,
        "current_solution_weighted": -1.0,
        "force_current_contributions": {
            "dominant_current_force": "A2_temperature",
            "dominant_effective_current_force": "effective_temperature_force",
        },
        "rhs_force_contributions": {
            "dominant_rhs_force": "A2_temperature",
            "dominant_effective_rhs_force": "effective_temperature_force",
        },
    }
    dump = {
        "matrix_condition_number": 10.0,
        "relative_residual_norm": 1.0e-14,
        "electron": {**species_template, "reference_current": 3.0},
        "ion": {**species_template, "reference_current": 4.0},
    }

    summary = _dump_forensic_summary(dump)

    assert summary["first_failure_class"] == (
        "momentum_correction_source_collision_or_observable_sign"
    )
    assert "total_current_gap_exceeds_1e-1_gate" in summary["failure_classes"]
