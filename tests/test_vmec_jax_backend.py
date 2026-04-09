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
