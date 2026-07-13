"""Independent residual and derivative gates for prepared solves."""

import jax.numpy as jnp
import pytest

from ntx import (
    GridSpec,
    MonoenergeticCase,
    audit_prepared_coefficient_derivative,
    example_surface,
    prepare_monoenergetic_system,
)


@pytest.fixture(scope="module")
def prepared_system():
    return prepare_monoenergetic_system(example_surface(), GridSpec(5, 5, 4))


def test_prepared_derivative_audit_passes_independent_gates(prepared_system):
    result = audit_prepared_coefficient_derivative(
        prepared_system,
        MonoenergeticCase(1.0e-2, er_hat=1.0e-3),
        coefficient="D11",
        parameter="er_hat",
    )

    assert bool(result.valid)
    assert float(result.primal_relative_residual) < 1.0e-10
    assert float(result.transpose_relative_residual) < 1.0e-10
    assert float(result.prepared_adjoint_relative_error) < 1.0e-10
    assert float(result.forward_relative_error) < 1.0e-10
    assert float(result.finite_difference_relative_error) < 1.0e-4
    assert result.as_dict()["valid"] is True


def test_prepared_derivative_audit_reports_failed_gate(prepared_system):
    result = audit_prepared_coefficient_derivative(
        prepared_system,
        MonoenergeticCase(1.0e-2, epsi_hat=1.0e-3),
        coefficient="D31",
        parameter="nu_hat",
        residual_tolerance=1.0e-30,
    )

    assert not bool(result.valid)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"coefficient": "D00"}, "coefficient must be one of"),
        ({"parameter": "temperature"}, "parameter must be"),
        ({"residual_tolerance": 0.0}, "tolerances must be positive"),
        ({"finite_difference_step": -1.0}, "step must be positive"),
    ],
)
def test_prepared_derivative_audit_rejects_invalid_options(
    prepared_system,
    kwargs,
    message,
):
    with pytest.raises(ValueError, match=message):
        audit_prepared_coefficient_derivative(
            prepared_system,
            MonoenergeticCase(jnp.asarray(1.0e-2), er_hat=1.0e-3),
            **kwargs,
        )
