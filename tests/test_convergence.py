"""Adaptive angular and Legendre convergence policies."""

import jax.numpy as jnp
import pytest

import ntx.convergence as convergence
from ntx import (
    BoozerSurface,
    MonoenergeticCase,
    TransportResult,
    example_surface,
    solve_monoenergetic_converged,
)


def _uniform_surface():
    return BoozerSurface(
        m=jnp.asarray([0]),
        n=jnp.asarray([0]),
        b_cos=jnp.asarray([1.0]),
        nfp=1,
        iota=0.6,
        psi_p=1.0,
        b_theta=0.1,
        b_zeta=1.0,
    )


def _result(value):
    scalar = jnp.asarray(value)
    return TransportResult(
        D11=scalar,
        D31=scalar * 1.0e-12,
        D13=scalar * 1.0e-12,
        D33=2.0 * scalar,
        D33_spitzer=3.0 * scalar,
        f1_modes=jnp.zeros((3, 1)),
        f3_modes=jnp.zeros((3, 1)),
        residual_l2=jnp.asarray(0.0),
        onsager_residual=jnp.asarray(0.0),
    )


def test_two_successive_refinements_are_required(monkeypatch):
    values = iter([1.0, 1.005, 1.006, 1.007, 1.008])
    monkeypatch.setattr(convergence, "solve_monoenergetic", lambda *args: _result(next(values)))
    audit = solve_monoenergetic_converged(
        _uniform_surface(),
        MonoenergeticCase(1.0e-2),
        angular_resolutions=((3, 3), (5, 5), (7, 7)),
        legendre_orders=(2, 4, 6),
        rtol=1.0e-2,
        atol=(1.0e-10, 1.0e-10, 1.0e-10),
    )
    assert audit.status == "converged"
    assert audit.angular_converged and audit.legendre_converged
    assert len(audit.steps) == 5
    assert [step.accepted for step in audit.steps] == [None, True, True, True, True]


def test_exhausted_ladder_is_not_silently_promoted(monkeypatch):
    values = iter([1.0, 1.5, 1.0, 1.5, 1.0])
    monkeypatch.setattr(convergence, "solve_monoenergetic", lambda *args: _result(next(values)))
    audit = solve_monoenergetic_converged(
        _uniform_surface(),
        MonoenergeticCase(1.0e-2),
        angular_resolutions=((3, 3), (5, 5), (7, 7)),
        legendre_orders=(2, 4, 6),
        rtol=1.0e-2,
    )
    assert audit.status == "unresolved"
    assert not audit.angular_converged
    assert not audit.legendre_converged
    assert audit.result is not None


def test_nonfinite_coefficients_are_model_out_of_scope(monkeypatch):
    monkeypatch.setattr(convergence, "solve_monoenergetic", lambda *args: _result(jnp.nan))
    audit = solve_monoenergetic_converged(
        _uniform_surface(),
        MonoenergeticCase(1.0e-2),
        angular_resolutions=((3, 3), (5, 5), (7, 7)),
        legendre_orders=(2, 4, 6),
    )
    assert audit.status == "model-out-of-scope"
    assert audit.result is None


@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({"rtol": -1.0}, "rtol"),
        ({"required_successive": 0}, "positive"),
        ({"angular_resolutions": ((3, 3),)}, "angular_resolutions"),
        ({"legendre_orders": (2,)}, "legendre_orders"),
        ({"atol": -1.0}, "atol"),
    ],
)
def test_invalid_convergence_policy_is_rejected(kwargs, message):
    options = {
        "angular_resolutions": ((3, 3), (5, 5), (7, 7)),
        "legendre_orders": (2, 4, 6),
    }
    options.update(kwargs)
    with pytest.raises(ValueError, match=message):
        solve_monoenergetic_converged(
            _uniform_surface(), MonoenergeticCase(1.0e-2), **options
        )


def test_uniform_field_passes_the_physical_convergence_gate():
    audit = solve_monoenergetic_converged(
        _uniform_surface(),
        MonoenergeticCase(1.0e-1),
        angular_resolutions=((3, 3), (5, 5), (7, 7)),
        legendre_orders=(2, 4, 6),
        rtol=1.0e-8,
        atol=(1.0e-10, 1.0e-10, 1.0e-10),
    )
    assert audit.status == "converged"
    assert audit.result is not None
    assert abs(float(audit.result.D11)) < 1.0e-10
    assert abs(float(audit.result.D31)) < 1.0e-10


def test_low_collisionality_requires_more_legendre_modes():
    options = {
        "angular_resolutions": ((5, 5), (7, 7), (9, 9), (11, 11)),
        "legendre_orders": (4, 6, 8, 10, 12),
        "rtol": 1.0e-2,
        "atol": (1.0e-10, 1.0e-10, 1.0e-10),
    }
    high = solve_monoenergetic_converged(
        example_surface(), MonoenergeticCase(1.0e-1, er_hat=1.0e-3), **options
    )
    low = solve_monoenergetic_converged(
        example_surface(), MonoenergeticCase(1.0e-3, er_hat=1.0e-3), **options
    )
    assert high.status == "converged"
    assert low.status == "unresolved"
    assert low.angular_converged
    assert not low.legendre_converged
