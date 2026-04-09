from __future__ import annotations

import jax.numpy as jnp
import pytest

from ntx import (
    GridSpec,
    MonoenergeticCase,
    build_ntx_neopax_scan_from_surfaces,
    load_boozmn_surface,
    load_vmec_surface,
    load_vmec_surface_reference_executable_reference,
    scan_to_neopax_arrays,
    solve_monoenergetic,
    surface_from_vmec_jax_vmec_wout_file,
    surface_from_vmec_jax_wout,
)
from ntx._checkout_paths import fixture_path

pytest.importorskip("vmec_jax")
pytest.importorskip("booz_xform_jax")

QA_INPUT = fixture_path("input.nfp3_QA_fixed_resolution_final")
QA_WOUT = fixture_path("wout_nfp3_QA_fixed_resolution_final.nc")
QA_BOOZ = fixture_path("boozmn_nfp3_QA_fixed_resolution_final.nc")
QH_INPUT = fixture_path("input.nfp3_QH_fixed_resolution_final")
QH_WOUT = fixture_path("wout_nfp3_QH_fixed_resolution_final.nc")
QH_BOOZ = fixture_path("boozmn_nfp3_QH_fixed_resolution_final.nc")
QI_WOUT = fixture_path("wout_nfp3_QI_fixed_resolution_final.nc")


@pytest.mark.parametrize(
    ("label", "input_path", "wout_path", "booz_path", "tolerance"),
    [
        ("QA", QA_INPUT, QA_WOUT, QA_BOOZ, 7.0e-2),
        ("QH", QH_INPUT, QH_WOUT, QH_BOOZ, 4.0e-2),
    ],
)
@pytest.mark.parametrize(
    ("nu_hat", "epsi_hat"),
    [
        (1.0e-4, 0.0),
        (1.0e-3, 1.0e-3),
    ],
)
def test_omnigenity_transform_lane_matches_boozmn_transport(
    label: str,
    input_path,
    wout_path,
    booz_path,
    tolerance: float,
    nu_hat: float,
    epsi_hat: float,
):
    booz_surface = load_boozmn_surface(booz_path, rho=0.5).surface
    transformed_surface = surface_from_vmec_jax_wout(
        input_path=input_path,
        wout_path=wout_path,
        s=0.25,
        mboz=24,
        nboz=24,
    )
    grid = GridSpec(n_theta=13, n_zeta=17, n_xi=16, dtype=jnp.float32)
    case = MonoenergeticCase(nu_hat=nu_hat, epsi_hat=epsi_hat)
    booz_result = solve_monoenergetic(booz_surface, grid, case)
    transformed_result = solve_monoenergetic(transformed_surface, grid, case)
    booz_values = jnp.asarray([booz_result.D11, booz_result.D31, booz_result.D13, booz_result.D33])
    transformed_values = jnp.asarray(
        [
            transformed_result.D11,
            transformed_result.D31,
            transformed_result.D13,
            transformed_result.D33,
        ]
    )
    relative = jnp.abs(
        (transformed_values - booz_values) / jnp.maximum(jnp.abs(booz_values), 1.0)
    )
    assert jnp.max(relative) < tolerance, (label, nu_hat, epsi_hat, relative)


def test_omnigenity_qi_vmec_harmonic_lane_matches_reference_loader():
    direct = surface_from_vmec_jax_vmec_wout_file(QI_WOUT, s=0.25)
    reference = load_vmec_surface_reference_executable_reference(QI_WOUT, s=0.25)
    result_direct = solve_monoenergetic(
        direct,
        GridSpec(n_theta=25, n_zeta=25, n_xi=24, dtype=jnp.float32),
        MonoenergeticCase(nu_hat=1.0e-4, epsi_hat=0.0),
    )
    result_reference = solve_monoenergetic(
        reference,
        GridSpec(n_theta=25, n_zeta=25, n_xi=24, dtype=jnp.float32),
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


def test_omnigenity_qi_vmec_file_loader_solves_finite_transport():
    surface = load_vmec_surface(QI_WOUT, psi_n=0.25)
    result = solve_monoenergetic(
        surface,
        GridSpec(n_theta=11, n_zeta=13, n_xi=16, dtype=jnp.float32),
        MonoenergeticCase(nu_hat=1.0e-3, er_hat=5.0e-4),
    )
    values = jnp.asarray([result.D11, result.D31, result.D13, result.D33])
    assert jnp.all(jnp.isfinite(values))


def test_omnigenity_qi_scan_maps_into_neopax_arrays():
    rho = jnp.asarray([0.25, 0.5])
    surfaces = tuple(
        surface_from_vmec_jax_vmec_wout_file(QI_WOUT, s=float(rho_value**2)) for rho_value in rho
    )
    scan = build_ntx_neopax_scan_from_surfaces(
        surfaces,
        rho=rho,
        nu_v=jnp.asarray([1.0e-4, 1.0e-3]),
        Es=jnp.asarray([[0.0, 5.0e-4], [0.0, 5.0e-4]]),
        Er=jnp.asarray([[0.0, 5.0e-4], [0.0, 5.0e-4]]),
        drds=jnp.asarray([1.0, 1.0]),
        grid=GridSpec(n_theta=11, n_zeta=13, n_xi=16, dtype=jnp.float32),
        source_name="omnigenity_qi_vmec_subset",
    )
    arrays = scan_to_neopax_arrays(scan, a_b=1.0)
    assert arrays.D11_log.shape == (2, 2, 2)
    assert arrays.D13.shape == (2, 2, 2)
    assert arrays.D33.shape == (2, 2, 2)
    assert jnp.all(jnp.isfinite(arrays.D11_log))
    assert jnp.all(jnp.isfinite(arrays.D13))
    assert jnp.all(jnp.isfinite(arrays.D33))
