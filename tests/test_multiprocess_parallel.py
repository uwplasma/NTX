from __future__ import annotations

import jax.numpy as jnp

from ntx import (
    GridSpec,
    load_dkes_surface,
    solve_monoenergetic_multiprocess_scan,
    solve_monoenergetic_scan,
)

from .fixture_data import SAMPLE_DKES


def test_multiprocess_cpu_scan_matches_serial():
    surface = load_dkes_surface(SAMPLE_DKES)
    grid = GridSpec(7, 9, 6)
    nu = jnp.logspace(-4, -2, 12)
    er = jnp.linspace(0.0, 2e-3, 12)
    serial = solve_monoenergetic_scan(surface, grid, nu, er_hat=er)
    parallel = solve_monoenergetic_multiprocess_scan(
        surface,
        grid,
        nu,
        er_hat=er,
        backend="cpu",
        workers=2,
    )
    for key in serial:
        assert jnp.allclose(serial[key], parallel[key], rtol=1e-10, atol=1e-12)
