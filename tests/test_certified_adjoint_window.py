"""The certified adjoint window, and the precision default it depends on.

Two things are checked here that no convergence study would reveal. First, that
a window advertised as certified actually bounds the realized gradient error --
against the exact full-window gradient, which the exact-window construction
makes exact rather than merely converged. Second, that geometry cannot be built
at single precision and then silently promoted by an x64 solve, which produced
coefficients accurate to about seven digits while reporting float64.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

import ntx
from ntx import GridSpec, MonoenergeticCase, example_surface

GRID = GridSpec(9, 9, 28)


@pytest.fixture(scope="module")
def prepared():
    return ntx.prepare_monoenergetic_system(example_surface(), GRID)


def _d11_gradient(prepared, nu_hat, window):
    def objective(value):
        case = MonoenergeticCase(nu_hat=value, epsi_hat=0.0)
        return ntx.solve_prepared(prepared, case, adjoint_window=window).D11

    return jax.grad(objective)(jnp.asarray(nu_hat))


# ------------------------------------------------------- the certificate ----


@pytest.mark.parametrize("nu_hat", [1e-1])
@pytest.mark.parametrize("rtol", [1e-4, 1e-6])
def test_certified_window_bounds_the_realized_gradient_error(prepared, nu_hat, rtol):
    case = MonoenergeticCase(nu_hat=nu_hat, epsi_hat=0.0)
    certificate = ntx.certify_adjoint_window(prepared, case, rtol=rtol)
    assert certificate.certified

    exact = _d11_gradient(prepared, nu_hat, GRID.n_xi + 1)
    got = _d11_gradient(prepared, nu_hat, int(certificate.window))
    realized = float(abs(got - exact) / abs(exact))

    assert certificate.certified_relative_error <= rtol
    assert realized <= rtol, f"realized {realized:.3e} exceeds rtol {rtol:.0e}"


def test_a_weakly_collisional_chain_returns_the_exact_window(prepared):
    # The honest failure mode: where the chain does not contract, the estimator
    # returns the exact window rather than a plausible short one. That is worth
    # pinning, because the tempting bug is to return something optimistic.
    case = MonoenergeticCase(nu_hat=1e-3, epsi_hat=0.0)
    certificate = ntx.certify_adjoint_window(prepared, case, rtol=1e-6)
    assert certificate.status == "full-window"
    assert certificate.window == GRID.n_xi + 1 - 3
    assert certificate.tail_bound == 0.0


def test_the_certified_window_is_never_shorter_than_the_heuristic_is_wrong(prepared):
    # The two answer different questions, so neither dominates -- but a
    # certified window that came out *below* the realized-error threshold would
    # mean the proof was unsound.
    case = MonoenergeticCase(nu_hat=1e-1, epsi_hat=0.0)
    certificate = ntx.certify_adjoint_window(prepared, case, rtol=1e-6)
    exact = _d11_gradient(prepared, 1e-1, GRID.n_xi + 1)
    got = _d11_gradient(prepared, 1e-1, int(certificate.window))
    assert float(abs(got - exact)) <= 1e-6 * float(abs(exact))


def test_a_tighter_tolerance_never_asks_for_a_narrower_window(prepared):
    case = MonoenergeticCase(nu_hat=1e-1, epsi_hat=0.0)
    windows = [
        ntx.certify_adjoint_window(prepared, case, rtol=r).window
        for r in (1e-2, 1e-4, 1e-6, 1e-8)
    ]
    assert windows == sorted(windows)


@pytest.mark.parametrize("coefficient", ["D11", "D33"])
def test_every_coefficient_can_be_certified(prepared, coefficient):
    case = MonoenergeticCase(nu_hat=1e-1, epsi_hat=0.0)
    certificate = ntx.certify_adjoint_window(
        prepared, case, rtol=1e-6, coefficient=coefficient
    )
    assert certificate.certified
    assert 0 <= certificate.window <= GRID.n_xi + 1 - 3


def test_an_unknown_coefficient_is_rejected(prepared):
    case = MonoenergeticCase(nu_hat=1e-1, epsi_hat=0.0)
    with pytest.raises(ValueError, match="coefficient must be one of"):
        ntx.certify_adjoint_window(prepared, case, coefficient="D99")


def test_the_certified_window_passes_straight_to_the_solver(prepared):
    case = MonoenergeticCase(nu_hat=1e-1, epsi_hat=0.0)
    certificate = ntx.certify_adjoint_window(prepared, case, rtol=1e-6)
    result = ntx.solve_prepared(prepared, case, adjoint_window=certificate)
    assert jnp.isfinite(result.D11)


# ---------------------------------------------------------- the precision ----


def test_ntx_is_double_precision_on_import():
    # The documented entry path builds geometry before the first solve. If x64
    # is not already on, that geometry is float32 for the rest of its life and
    # the solve promotes it silently.
    assert jax.config.jax_enable_x64 is True
    assert example_surface().b_cos.dtype == jnp.float64


def test_a_single_precision_surface_is_refused_rather_than_promoted():
    ntx.enable_x64(False)
    try:
        narrow = example_surface()
    finally:
        ntx.enable_x64(True)
    assert narrow.b_cos.dtype == jnp.float32
    with pytest.raises(ValueError, match="narrower precision"):
        ntx.solve_monoenergetic(
            narrow, GridSpec(9, 9, 8), MonoenergeticCase(nu_hat=1e-2, epsi_hat=0.0)
        )


def test_the_single_precision_path_is_still_available_when_asked_for():
    # Opting out deliberately must keep working: it is a throughput lane, not a
    # mistake, as long as the geometry and the grid agree.
    ntx.enable_x64(False)
    try:
        surface = example_surface()
        result = ntx.solve_monoenergetic(
            surface,
            GridSpec(9, 9, 8, dtype="float32", x64=False),
            MonoenergeticCase(nu_hat=1e-2, epsi_hat=0.0),
        )
        assert result.D11.dtype == jnp.float32
    finally:
        ntx.enable_x64(True)
