from pathlib import Path

import jax.numpy as jnp
import pytest

from ntx import GridSpec, MonoenergeticCase, solve_monoenergetic, surface_from_vmec_jax_wout

VMEC_INPUT = Path(
    "/Users/rogeriojorge/local/tests/simsopt/tests/test_files/"
    "input.W7-X_standard_configuration"
)
WOUT = Path(
    "/Users/rogeriojorge/local/tests/NEOPAX/tests/inputs/"
    "wout_W7-X_standard_configuration.nc"
)

if not VMEC_INPUT.exists() or not WOUT.exists():
    pytest.skip("local vmec_jax W7-X inputs are not available", allow_module_level=True)

pytest.importorskip("vmec_jax")
pytest.importorskip("booz_xform_jax")


def test_surface_from_vmec_jax_wout_builds_finite_transport():
    surface = surface_from_vmec_jax_wout(
        input_path=VMEC_INPUT,
        wout_path=WOUT,
        s=0.25,
        mboz=6,
        nboz=6,
    )
    result = solve_monoenergetic(
        surface,
        GridSpec(n_theta=9, n_zeta=13, n_xi=24, dtype=jnp.float32),
        MonoenergeticCase(nu_hat=1.0e-4, epsi_hat=0.0),
    )
    assert jnp.isfinite(result.D11)
    assert jnp.isfinite(result.D13)
    assert jnp.isfinite(result.D33)


def test_surface_from_vmec_jax_wout_converges_with_grid_refinement():
    surface = surface_from_vmec_jax_wout(
        input_path=VMEC_INPUT,
        wout_path=WOUT,
        s=0.25,
        mboz=6,
        nboz=6,
    )
    coarse = solve_monoenergetic(
        surface,
        GridSpec(n_theta=9, n_zeta=13, n_xi=24, dtype=jnp.float32),
        MonoenergeticCase(nu_hat=1.0e-4, epsi_hat=0.0),
    )
    fine = solve_monoenergetic(
        surface,
        GridSpec(n_theta=13, n_zeta=17, n_xi=32, dtype=jnp.float32),
        MonoenergeticCase(nu_hat=1.0e-4, epsi_hat=0.0),
    )
    coarse_values = jnp.asarray([coarse.D11, coarse.D31, coarse.D13, coarse.D33])
    fine_values = jnp.asarray([fine.D11, fine.D31, fine.D13, fine.D33])
    assert jnp.all(jnp.isfinite(coarse_values))
    assert jnp.all(jnp.isfinite(fine_values))
    relative = jnp.abs((fine_values - coarse_values) / jnp.maximum(jnp.abs(fine_values), 1.0))
    assert jnp.max(relative) < 0.35
