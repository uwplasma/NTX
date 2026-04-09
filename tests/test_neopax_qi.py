from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp

from ntx import (
    GridSpec,
    build_ntx_neopax_scan_from_surfaces,
    load_neopax_reference_scan,
    load_vmec_surface,
    write_neopax_scan_hdf5,
)

QI_VMEC = Path(
    "/Users/rogeriojorge/local/.NTX/tests/fixtures/"
    "wout_QI_nfp2_stable_Er_006_000043_hires_scaled.nc"
)


def test_qi_neopax_scan_round_trips_through_hdf5(tmp_path):
    rho = jnp.asarray([0.12247, 0.25])
    surfaces = tuple(load_vmec_surface(QI_VMEC, psi_n=float(rho_value**2)) for rho_value in rho)
    nu_v = jnp.asarray([1.0e-4, 1.0e-3])
    es = jnp.asarray([[0.0, 5.0e-4], [0.0, 5.0e-4]])
    er = jnp.asarray([[0.0, 5.0e-4], [0.0, 5.0e-4]])
    drds = jnp.asarray([1.0, 1.0])

    scan = build_ntx_neopax_scan_from_surfaces(
        surfaces,
        rho=rho,
        nu_v=nu_v,
        Es=es,
        Er=er,
        drds=drds,
        grid=GridSpec(9, 11, 16),
        source_name="qi_ntx_roundtrip",
    )
    output = tmp_path / "qi_ntx_scan.h5"
    write_neopax_scan_hdf5(scan, output)
    loaded = load_neopax_reference_scan(output)

    assert jnp.allclose(loaded.rho, scan.rho)
    assert jnp.allclose(loaded.nu_v, scan.nu_v)
    assert jnp.allclose(loaded.Er, scan.Er)
    assert jnp.allclose(loaded.Es, scan.Es)
    assert jnp.allclose(loaded.D11, scan.D11)
    assert jnp.allclose(loaded.D13, scan.D13)
    assert jnp.allclose(loaded.D33, scan.D33)
