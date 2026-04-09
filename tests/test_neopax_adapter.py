import sys
from pathlib import Path

import jax.numpy as jnp
import pytest

from ntx import (
    GridSpec,
    build_ntx_neopax_scan,
    load_neopax_reference_scan,
    surface_from_vmec_jax_vmec_wout_file,
    to_neopax_monoenergetic,
)

NEOPAX_ROOT = Path("/Users/rogeriojorge/local/tests/NEOPAX")
if not NEOPAX_ROOT.exists():
    pytest.skip("local NEOPAX checkout not available", allow_module_level=True)

if str(NEOPAX_ROOT) not in sys.path:
    sys.path.insert(0, str(NEOPAX_ROOT))

NEOPAX = pytest.importorskip("NEOPAX")

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
    rho_idx = jnp.asarray([1, 3])
    nu_idx = jnp.asarray([5, 7, 9])
    er_idx = jnp.asarray([0, 7, 9])
    rho = reference.rho[rho_idx]
    nu_v = reference.nu_v[nu_idx]
    Er = reference.Er[rho_idx][:, er_idx]
    Es = reference.Es[rho_idx][:, er_idx]
    drds = reference.drds[rho_idx]

    def surface_loader(rho_value: float):
        return surface_from_vmec_jax_vmec_wout_file(
            W7X_WOUT,
            s=float(rho_value**2),
        )

    scan = build_ntx_neopax_scan(
        surface_loader,
        rho=rho,
        nu_v=nu_v,
        Es=Es,
        Er=Er,
        drds=drds,
        grid=GridSpec(n_theta=25, n_zeta=25, n_xi=63),
        source_name="w7x_vmec_jax_vmec_subset",
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
            D11=reference.D11[rho_idx][:, nu_idx][:, :, er_idx],
            D13=reference.D13[rho_idx][:, nu_idx][:, :, er_idx],
            D33=reference.D33[rho_idx][:, nu_idx][:, :, er_idx],
            source_name="reference_subset",
        ),
        a_b=1.0,
    )
    assert jnp.max(jnp.abs(database.D11_log - reference_database.D11_log)) < 1.0e-2
    assert jnp.max(jnp.abs(database.D13 - reference_database.D13)) < 1.0e-2
    assert (
        jnp.max(
            jnp.abs(database.D33 - reference_database.D33)
            / jnp.maximum(jnp.abs(reference_database.D33), 1.0e-12)
        )
        < 1.0e-2
    )
