import sys
from pathlib import Path

import jax.numpy as jnp
import pytest

from ntx import (
    GridSpec,
    build_ntx_neopax_scan,
    load_neopax_reference_scan,
    load_vmec_surface,
    to_neopax_monoenergetic,
)

NEOPAX_ROOT = Path("/Users/rogeriojorge/local/tests/NEOPAX")
if str(NEOPAX_ROOT) not in sys.path:
    sys.path.insert(0, str(NEOPAX_ROOT))

import NEOPAX  # noqa: E402

W7X_WOUT = NEOPAX_ROOT / "tests" / "inputs" / "wout_W7-X_standard_configuration.nc"
W7X_REFERENCE_EXECUTABLE = NEOPAX_ROOT / "tests" / "inputs" / "Dij_NEOPAX_FULL_S_NEW_W7X.h5"


def test_reference_scan_round_trips_into_neopax_constructor():
    scan = load_neopax_reference_scan(W7X_REFERENCE_EXECUTABLE)
    reference = NEOPAX.Monoenergetic.read_reference_executable(1.0, W7X_REFERENCE_EXECUTABLE)
    mapped = to_neopax_monoenergetic(scan, a_b=1.0)
    assert jnp.allclose(mapped.rho, reference.rho)
    assert jnp.allclose(mapped.nu_log, reference.nu_log)
    assert jnp.allclose(mapped.Er_list, reference.Er_list)
    assert jnp.allclose(mapped.D11_log, reference.D11_log)
    assert jnp.allclose(mapped.D13, reference.D13)
    assert jnp.allclose(mapped.D33, reference.D33)


@pytest.mark.benchmark
def test_ntx_scan_maps_into_neopax_and_tracks_reference_subset():
    reference = load_neopax_reference_scan(W7X_REFERENCE_EXECUTABLE)
    rho = reference.rho[:2]
    nu_v = reference.nu_v[2:5]
    Er = reference.Er[:2, :3]
    Es = reference.Es[:2, :3]
    drds = reference.drds[:2]

    def surface_loader(rho_value: float):
        return load_vmec_surface(
            W7X_WOUT,
            psi_n=float(rho_value**2),
            vmec_radial_option=1,
            vmec_nyquist_option=2,
            vmec_mode_convention="filtered_nyquist",
        )

    scan = build_ntx_neopax_scan(
        surface_loader,
        rho=rho,
        nu_v=nu_v,
        Es=Es,
        Er=Er,
        drds=drds,
        grid=GridSpec(n_theta=17, n_zeta=33, n_xi=60),
        source_name="w7x_vmec_subset",
    )
    database = to_neopax_monoenergetic(scan, a_b=1.0)
    assert database.D11_log.shape == (2, 3, 3)
    assert database.D13.shape == (2, 3, 3)
    assert database.D33.shape == (2, 3, 3)
    reference_database = to_neopax_monoenergetic(
        type(scan)(
            rho=rho,
            nu_v=nu_v,
            Er=Er,
            Es=Es,
            drds=drds,
            D11=reference.D11[:2, 2:5, :3],
            D13=reference.D13[:2, 2:5, :3],
            D33=reference.D33[:2, 2:5, :3],
            source_name="reference_subset",
        ),
        a_b=1.0,
    )
    # The adapter should preserve the shape and stay in the same regime as the
    # existing NEOPAX/REFERENCE_EXECUTABLE subset, even though VMEC parity is not exact yet.
    assert jnp.max(jnp.abs(database.D11_log - reference_database.D11_log)) < 1.0
    assert jnp.max(jnp.abs(database.D33 - reference_database.D33) / reference_database.D33) < 0.2
