from __future__ import annotations

import jax.numpy as jnp
from vmec_jax import read_wout

from ntx import (
    GridSpec,
    MonoenergeticCase,
    solve_monoenergetic,
    surface_from_vmec_jax_vmec_wout,
    surface_from_vmec_jax_vmec_wout_file,
)

from .fixture_data import SAMPLE_WOUT


def test_surface_from_vmec_jax_vmec_wout_file_matches_in_memory_builder():
    wout = read_wout(SAMPLE_WOUT)
    direct = surface_from_vmec_jax_vmec_wout_file(SAMPLE_WOUT, s=0.25)
    in_memory = surface_from_vmec_jax_vmec_wout(wout, s=0.25, source_path=SAMPLE_WOUT)

    result_direct = solve_monoenergetic(
        direct,
        GridSpec(n_theta=9, n_zeta=9, n_xi=8),
        MonoenergeticCase(nu_hat=1.0e-3, epsi_hat=0.0),
    )
    result_memory = solve_monoenergetic(
        in_memory,
        GridSpec(n_theta=9, n_zeta=9, n_xi=8),
        MonoenergeticCase(nu_hat=1.0e-3, epsi_hat=0.0),
    )
    direct_values = jnp.asarray(
        [result_direct.D11, result_direct.D31, result_direct.D13, result_direct.D33]
    )
    memory_values = jnp.asarray(
        [result_memory.D11, result_memory.D31, result_memory.D13, result_memory.D33]
    )
    assert jnp.max(jnp.abs(direct_values - memory_values)) < 1.0e-12


def test_surface_from_vmec_jax_vmec_wout_file_preserves_metadata():
    direct = surface_from_vmec_jax_vmec_wout_file(SAMPLE_WOUT, s=0.25)
    assert direct.nfp == 2
    assert direct.ns == 5
    assert direct.total_mode_count == 4
    assert direct.loaded_mode_count > 0
    assert direct.path == SAMPLE_WOUT.resolve()
