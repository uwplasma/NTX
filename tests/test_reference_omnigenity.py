from __future__ import annotations

import jax.numpy as jnp

from ntx import (
    build_reference_vmec_scan,
    load_boozmn_surface,
    load_neopax_reference_scan,
    scan_to_neopax_arrays,
)
from ntx._checkout_paths import fixture_path


def test_generated_qi_boozmn_fixture_loads_through_packed_surface_metadata():
    payload = load_boozmn_surface(fixture_path("boozmn_nfp3_QI_fixed_resolution_final.nc"), rho=0.5)
    assert payload.surface.b0 > 0.0
    assert payload.mode_count > 0


def test_omnigenity_external_reference_subsets_match_ntx():
    cases = (
        ("qa", 2.0e-2),
        ("qh", 3.0e-2),
        ("qi", 1.0e-5),
    )
    for label, tolerance in cases:
        reference = load_neopax_reference_scan(
            fixture_path("benchmarks", "omnigenity", f"external_reference_{label}_subset.h5")
        )
        scan = build_reference_vmec_scan(
            fixture_path(f"wout_nfp3_{label.upper()}_fixed_resolution_final.nc"),
            fixture_path(f"boozmn_nfp3_{label.upper()}_fixed_resolution_final.nc"),
            rho=reference.rho,
            nu_v=reference.nu_v,
            er_tilde=reference.Er_tilde,
            nt=25,
            nz=25,
            nl=64,
            source_name=f"ntx_external_reference_{label}",
        )

        for actual, expected in (
            (scan.D11, reference.D11),
            (scan.D31, reference.D31),
            (scan.D13, reference.D13),
            (scan.D33, reference.D33),
        ):
            relative = jnp.abs((actual - expected) / jnp.maximum(jnp.abs(expected), 1.0e-12))
            assert float(jnp.max(relative)) < tolerance, (label, float(jnp.max(relative)))

        actual_arrays = scan_to_neopax_arrays(scan, a_b=float(scan.a_b))
        reference_arrays = scan_to_neopax_arrays(reference, a_b=float(scan.a_b))
        assert jnp.max(jnp.abs(actual_arrays.D11_log - reference_arrays.D11_log)) < tolerance
        d13_relative = jnp.abs(
            (actual_arrays.D13 - reference_arrays.D13)
            / jnp.maximum(jnp.abs(reference_arrays.D13), 1.0e-12)
        )
        assert float(jnp.max(d13_relative)) < tolerance, (label, float(jnp.max(d13_relative)))
        d33_relative = jnp.abs(
            (actual_arrays.D33 - reference_arrays.D33)
            / jnp.maximum(jnp.abs(reference_arrays.D33), 1.0e-12)
        )
        assert float(jnp.max(d33_relative)) < tolerance, (label, float(jnp.max(d33_relative)))
