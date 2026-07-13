"""Owned coefficient gates for the SOLVAX block-solver integration."""

import jax.numpy as jnp
import pytest

from ntx import GridSpec, MonoenergeticCase, example_surface, solve_monoenergetic


@pytest.mark.parametrize(
    "n_xi,expected",
    [
        (
            2,
            (
                0.02311756076600616,
                -0.12819189717517873,
                0.12810223369038104,
                2.7582018573441225,
                66.23356280715441,
            ),
        ),
        (
            16,
            (
                0.0018231293651485413,
                -0.09883328713871108,
                0.09917116215592461,
                52.61080425700945,
                66.23356280715441,
            ),
        ),
        (
            32,
            (
                0.0017805623686559266,
                -0.09821996003082374,
                0.09856006594325076,
                53.05742135176217,
                66.23356280715441,
            ),
        ),
        (
            63,
            (
                0.0017805622331006354,
                -0.0982199678525303,
                0.09856007512410699,
                53.05742475250274,
                66.23356280715441,
            ),
        ),
        (
            140,
            (
                0.0017805622331006354,
                -0.0982199678525303,
                0.09856007512410699,
                53.05742475250274,
                66.23356280715441,
            ),
        ),
    ],
)
def test_solvax_migration_preserves_legendre_ladder(n_xi, expected):
    result = solve_monoenergetic(
        example_surface(),
        GridSpec(5, 5, n_xi),
        MonoenergeticCase(1.0e-2, er_hat=1.0e-3),
    )
    actual = jnp.asarray([result.D11, result.D31, result.D13, result.D33, result.D33_spitzer])
    assert jnp.allclose(actual, jnp.asarray(expected), rtol=1.0e-11, atol=1.0e-12)
    assert float(result.residual_l2) < 1.0e-12
