from pathlib import Path

import jax.numpy as jnp
import pytest

from ntx import (
    GridSpec,
    MonoenergeticCase,
    load_vmec_surface_reference_executable_reference,
    solve_monoenergetic,
    surface_from_vmec_jax_vmec_wout_file,
)

WOUT = Path(
    "/Users/rogeriojorge/local/tests/NEOPAX/tests/inputs/"
    "wout_W7-X_standard_configuration.nc"
)

if not WOUT.exists():
    pytest.skip("local W7-X wout fixture not available", allow_module_level=True)

pytest.importorskip("vmec_jax")


def test_surface_from_vmec_jax_vmec_wout_file_matches_reference_transport():
    direct = surface_from_vmec_jax_vmec_wout_file(WOUT, s=0.25)
    reference = load_vmec_surface_reference_executable_reference(WOUT, s=0.25)
    result_direct = solve_monoenergetic(
        direct,
        GridSpec(n_theta=25, n_zeta=25, n_xi=63),
        MonoenergeticCase(nu_hat=1.0e-4, epsi_hat=0.0),
    )
    result_reference = solve_monoenergetic(
        reference,
        GridSpec(n_theta=25, n_zeta=25, n_xi=63),
        MonoenergeticCase(nu_hat=1.0e-4, epsi_hat=0.0),
    )
    direct_values = jnp.asarray(
        [result_direct.D11, result_direct.D31, result_direct.D13, result_direct.D33]
    )
    reference_values = jnp.asarray(
        [result_reference.D11, result_reference.D31, result_reference.D13, result_reference.D33]
    )
    relative = jnp.abs(
        (direct_values - reference_values) / jnp.maximum(jnp.abs(reference_values), 1.0)
    )
    assert jnp.max(relative) < 1.0e-10


def test_surface_from_vmec_jax_vmec_wout_file_preserves_reference_metadata():
    direct = surface_from_vmec_jax_vmec_wout_file(WOUT, s=0.25)
    reference = load_vmec_surface_reference_executable_reference(WOUT, s=0.25)
    assert direct.nfp == reference.nfp
    assert direct.ns == reference.ns
    assert direct.total_mode_count == reference.total_mode_count
    assert jnp.all(direct.m == reference.m)
    assert jnp.all(direct.n == reference.n)
    assert abs(float(direct.iota - reference.iota)) < 1.0e-12
    assert abs(float(direct.b0 - reference.b0)) < 1.0e-12
