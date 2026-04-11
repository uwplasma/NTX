from __future__ import annotations

import jax.numpy as jnp

from ntx import (
    GridSpec,
    load_dkes_surface,
    local_parallel_device_count,
    solve_monoenergetic_parallel_scan,
    solve_monoenergetic_scan,
)

from .fixture_data import SAMPLE_DKES


def test_parallel_scan_matches_serial_scan():
    surface = load_dkes_surface(SAMPLE_DKES)
    grid = GridSpec(7, 9, 6)
    nu = jnp.logspace(-4, -2, 10)
    er = jnp.linspace(0.0, 2e-3, 10)
    serial = solve_monoenergetic_scan(surface, grid, nu, er_hat=er)
    parallel = solve_monoenergetic_parallel_scan(surface, grid, nu, er_hat=er)
    for key in serial:
        assert jnp.allclose(serial[key], parallel[key], rtol=1e-10, atol=1e-12)


def test_local_parallel_device_count_is_positive():
    assert local_parallel_device_count() >= 1
