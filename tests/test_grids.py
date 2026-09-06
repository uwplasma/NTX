from __future__ import annotations

import jax.numpy as jnp
import jax

from ntx.grids import fourier_derivative_matrix


def test_fourier_derivative_matrix_differentiates_trig_mode():
    n = 9
    x = jnp.arange(n, dtype=jnp.float64) * 2 * jnp.pi / n
    d = fourier_derivative_matrix(n, 2 * jnp.pi)
    f = jnp.sin(2 * x) + 0.5 * jnp.cos(3 * x)
    expected = 2 * jnp.cos(2 * x) - 1.5 * jnp.sin(3 * x)
    assert jnp.max(jnp.abs(d @ f - expected)) < 1e-12


def test_fourier_derivative_matrix_accepts_a_jitted_dynamic_period():
    """Recorded VMEC surfaces carry nfp through the final compiled scan VJP."""
    n = 9
    period = jnp.asarray(2 * jnp.pi / 3)
    dynamic = jax.jit(lambda value: fourier_derivative_matrix(n, value))(period)
    static = fourier_derivative_matrix(n, period)
    assert jnp.allclose(dynamic, static, rtol=1.0e-12, atol=1.0e-12)
