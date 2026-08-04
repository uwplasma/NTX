from __future__ import annotations

import jax.numpy as jnp

from ntx import (
    GridSpec,
    build_ntx_neopax_scan_from_surfaces,
    load_neopax_reference_scan,
    surface_from_vmex_vmec_wout_file,
    write_neopax_scan_hdf5,
)

from .fixture_data import SAMPLE_NEOPAX, SAMPLE_WOUT


def test_neopax_scan_hdf5_round_trip(tmp_path):
    reference = load_neopax_reference_scan(SAMPLE_NEOPAX)
    rho = reference.rho
    surfaces = tuple(
        surface_from_vmex_vmec_wout_file(SAMPLE_WOUT, s=float(rho_value**2)) for rho_value in rho
    )
    scan = build_ntx_neopax_scan_from_surfaces(
        surfaces,
        rho=rho,
        nu_v=reference.nu_v,
        Es=reference.Es,
        Er=reference.Er,
        drds=reference.drds,
        grid=GridSpec(7, 9, 6),
        source_name="sample_neopax_roundtrip",
    )
    output = tmp_path / "scan.h5"
    write_neopax_scan_hdf5(scan, output)
    loaded = load_neopax_reference_scan(output)
    assert jnp.allclose(loaded.rho, scan.rho)
    assert loaded.D11.shape == scan.D11.shape
    assert loaded.D13.shape == scan.D13.shape
    assert loaded.D33.shape == scan.D33.shape
