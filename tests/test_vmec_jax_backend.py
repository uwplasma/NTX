from pathlib import Path

import jax.numpy as jnp
import pytest

from ntx import (
    GridSpec,
    MonoenergeticCase,
    load_boozmn_surface,
    solve_monoenergetic,
    surface_from_vmec_jax_wout,
)

VMEC_INPUT = Path(
    "/Users/rogeriojorge/local/tests/simsopt/tests/test_files/"
    "input.W7-X_standard_configuration"
)
WOUT = Path(
    "/Users/rogeriojorge/local/tests/NEOPAX/tests/inputs/"
    "wout_W7-X_standard_configuration.nc"
)
BOOZ = Path(
    "/Users/rogeriojorge/local/tests/NEOPAX/tests/inputs/"
    "boozmn_wout_W7-X_standard_configuration.nc"
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


def test_surface_from_vmec_jax_wout_matches_boozmn_transport():
    if not BOOZ.exists():
        pytest.skip("local boozmn fixture is not available")

    booz_surface = load_boozmn_surface(BOOZ, rho=0.5).surface
    jax_surface = surface_from_vmec_jax_wout(
        input_path=VMEC_INPUT,
        wout_path=WOUT,
        s=0.25,
        mboz=24,
        nboz=24,
    )
    spec = GridSpec(n_theta=13, n_zeta=17, n_xi=16, dtype=jnp.float32)
    case = MonoenergeticCase(nu_hat=1.0e-3, epsi_hat=1.0e-3)
    booz_result = solve_monoenergetic(booz_surface, spec, case)
    jax_result = solve_monoenergetic(jax_surface, spec, case)
    booz_values = jnp.asarray([booz_result.D11, booz_result.D31, booz_result.D13, booz_result.D33])
    jax_values = jnp.asarray([jax_result.D11, jax_result.D31, jax_result.D13, jax_result.D33])
    relative = jnp.abs((jax_values - booz_values) / jnp.maximum(jnp.abs(booz_values), 1.0))
    assert jnp.max(relative) < 0.02
