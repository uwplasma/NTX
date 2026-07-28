"""Gates for the bounded reverse pass through the monoenergetic solve.

The forward solve is unchanged by anything here; what changes is how a
reverse-mode derivative of it is taken. Three properties have to hold, and each
one is a separate way the change could go wrong:

* the transport coefficients must not move at all;
* the gradient at full window must equal the gradient the taped elimination
  gives, because the full window is exact by construction;
* a finite window must bound the reverse pass without bounding the forward one.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from ntx import (
    GridSpec,
    MonoenergeticCase,
    advise_adjoint_window,
    example_surface,
    solve_monoenergetic,
)
from ntx._solver_core import prepare_monoenergetic_system
from ntx._solver_factorization import _solve_modes_with_tail_residual
from ntx.operators import (
    OperatorContext,
    block_parameters,
    coefficients_for_k,
    coefficients_from_parameters,
    source_modes,
)

NU, EPSI = 1.0e-2, 1.0e-3


def _context(n_xi=32, n_theta=5, n_zeta=5):
    grid = GridSpec(n_theta, n_zeta, n_xi)
    prepared = prepare_monoenergetic_system(example_surface(), grid)
    case = MonoenergeticCase(NU, er_hat=EPSI)
    epsi = case.resolved_epsi_hat(prepared.geometry.transport_psi_scale)
    ctx = OperatorContext(
        prepared.surface, prepared.geometry, jnp.asarray(NU), jnp.asarray(epsi)
    )
    return prepared, grid, ctx


def test_parameter_extraction_reproduces_the_block_coefficients():
    """The two ways of building a row must agree exactly, or the refactor lied."""
    _, grid, ctx = _context(n_xi=8)
    params = block_parameters(ctx)
    for k in (0, 1, 5):
        for direct, viaparams in zip(
            coefficients_for_k(ctx, k),
            coefficients_from_parameters(params, k),
            strict=True,
        ):
            assert jnp.array_equal(direct, viaparams)


@pytest.mark.parametrize("n_xi", [16, 48])
def test_transport_coefficients_are_unchanged_by_the_window(n_xi):
    """A window bounds the reverse pass; it must not touch the forward answer."""
    grid = GridSpec(5, 5, n_xi)
    case = MonoenergeticCase(NU, er_hat=EPSI)
    reference = solve_monoenergetic(example_surface(), grid, case)
    for window in (None, 4, 12):
        result = solve_monoenergetic(
            example_surface(), grid, case, adjoint_window=window
        )
        for name in ("D11", "D31", "D13", "D33", "D33_spitzer"):
            assert jnp.array_equal(
                getattr(reference, name), getattr(result, name)
            ), f"{name} moved at window {window}"


def test_full_window_gradient_equals_the_taped_gradient():
    """The full window is exact, so it must reproduce taping to rounding.

    This is the property that makes the default safe: users who never touch
    ``adjoint_window`` get the same numbers they got before, more cheaply.
    """
    prepared, grid, ctx = _context(n_xi=24)
    s1, s3 = source_modes(ctx, grid.n_xi)
    base = block_parameters(ctx)
    weights = jax.random.normal(jax.random.PRNGKey(0), (3, prepared.d_theta.shape[0]))

    def modes(nu_scale, window):
        scaled = OperatorContext(
            prepared.surface, prepared.geometry, ctx.nu_hat * nu_scale, ctx.epsi_hat
        )
        f1, _, _ = _solve_modes_with_tail_residual(
            scaled, grid.n_xi, prepared.d_theta, prepared.d_zeta, s1, s3, window
        )
        return jnp.sum(weights * f1[:3])

    one = jnp.asarray(1.0)
    g_full = jax.grad(lambda a: modes(a, None))(one)
    g_explicit = jax.grad(lambda a: modes(a, grid.n_xi + 1))(one)
    assert jnp.allclose(g_full, g_explicit, rtol=1e-12, atol=0.0)
    assert float(jnp.abs(g_full)) > 0.0
    assert set(base) == set(block_parameters(ctx))


def test_finite_window_bounds_the_reverse_pass_but_not_the_forward_one():
    """The point of the change: reverse memory stops tracking ``n_xi``."""
    def reverse_bytes(n_xi, window):
        prepared, grid, ctx = _context(n_xi=n_xi)
        s1, s3 = source_modes(ctx, grid.n_xi)

        def loss(nu_scale):
            scaled = OperatorContext(
                prepared.surface, prepared.geometry, ctx.nu_hat * nu_scale, ctx.epsi_hat
            )
            f1, _, _ = _solve_modes_with_tail_residual(
                scaled, grid.n_xi, prepared.d_theta, prepared.d_zeta, s1, s3, window
            )
            return jnp.sum(f1[:3] ** 2)

        return (
            jax.jit(jax.grad(loss))
            .lower(jnp.asarray(1.0))
            .compile()
            .memory_analysis()
            .temp_size_in_bytes
        )

    small, large = reverse_bytes(32, 6), reverse_bytes(96, 6)
    assert large < 1.35 * small, (
        f"a fixed window should keep the reverse pass roughly flat in n_xi, "
        f"got {small} -> {large} bytes"
    )
    assert reverse_bytes(96, None) > 1.8 * large


def test_window_advisor_reports_an_uncertified_estimate():
    prepared, grid, ctx = _context(n_xi=64)
    advice = advise_adjoint_window(
        ctx, grid.n_xi, prepared.d_theta, prepared.d_zeta
    )
    assert advice.certified is False
    assert 0 <= advice.window <= grid.n_xi + 1
    assert advice.primal_profile.shape[0] > 0


def test_advised_window_is_usable_and_accurate_enough_to_be_worth_advising():
    """A window the advisor returns should beat a gradient of the wrong sign.

    We do not assert a tolerance the advisor does not promise. We assert the
    weaker, honest thing: the advised window gets the gradient far closer to
    the exact one than truncating at the source support does.
    """
    prepared, grid, ctx = _context(n_xi=64)
    s1, s3 = source_modes(ctx, grid.n_xi)
    advice = advise_adjoint_window(ctx, grid.n_xi, prepared.d_theta, prepared.d_zeta)

    def grad_at(window):
        def loss(nu_scale):
            scaled = OperatorContext(
                prepared.surface, prepared.geometry, ctx.nu_hat * nu_scale, ctx.epsi_hat
            )
            f1, _, _ = _solve_modes_with_tail_residual(
                scaled, grid.n_xi, prepared.d_theta, prepared.d_zeta, s1, s3, window
            )
            return jnp.sum(f1[:3] ** 2)

        return float(jax.grad(loss)(jnp.asarray(1.0)))

    exact = grad_at(None)
    at_zero = abs(grad_at(0) - exact) / abs(exact)
    at_advised = abs(grad_at(advice.window) - exact) / abs(exact)
    assert at_advised < at_zero
    assert at_advised < 1e-2


def test_scan_accepts_a_window_without_moving_the_coefficients():
    """The scan drivers are where a design study spends its reverse memory.

    Threading the window through them is only useful if the scan's own answer
    is untouched, so this pins every scanned coefficient across windows.
    """
    from ntx import solve_monoenergetic_scan

    grid = GridSpec(5, 5, 24)
    nu = jnp.asarray([1.0e-2, 1.0e-1])
    reference = solve_monoenergetic_scan(
        example_surface(), grid, nu, er_hat=jnp.full_like(nu, EPSI)
    )
    for window in (None, 6):
        scanned = solve_monoenergetic_scan(
            example_surface(), grid, nu,
            er_hat=jnp.full_like(nu, EPSI), adjoint_window=window,
        )
        for name in ("D11", "D31", "D13", "D33"):
            assert jnp.array_equal(reference[name], scanned[name]), (
                f"{name} moved at window {window}"
            )


def test_scan_is_not_differentiable_end_to_end_today():
    """Records a pre-existing limit, so the window is not blamed for it.

    ``solve_monoenergetic_scan`` performs Python-level work on its inputs, so
    ``jax.grad`` of it fails whether or not a window is supplied. Threading
    ``adjoint_window`` through the scan is still worth doing --- it reaches the
    per-point solves, which is where a caller differentiating inside their own
    loop spends reverse memory --- but it does not make the scan itself
    differentiable. If this test starts failing, the scan became
    differentiable and the window's benefit there should be measured.
    """
    from ntx import solve_monoenergetic_scan

    def loss(scale):
        nu = jnp.asarray([1.0e-2, 1.0e-1]) * scale
        out = solve_monoenergetic_scan(
            example_surface(), GridSpec(5, 5, 16), nu,
            er_hat=jnp.full_like(nu, EPSI),
        )
        return jnp.sum(out["D11"])

    with pytest.raises(TypeError, match="constant handler"):
        jax.grad(loss)(jnp.asarray(1.0))
