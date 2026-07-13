from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from ntx import (
    GridSpec,
    MonoenergeticCase,
    load_vmec_surface,
    solve_monoenergetic,
    surface_from_vmec_jax_vmec_wout_file,
)
from ntx.geometry import VmecSurface, geometry_on_grid

from .fixture_data import SAMPLE_WOUT


def test_load_vmec_surface_and_geometry():
    surface = load_vmec_surface(SAMPLE_WOUT, psi_n=0.25)
    assert isinstance(surface, VmecSurface)
    assert surface.path == SAMPLE_WOUT.resolve()
    assert surface.nfp == 2
    assert surface.ns == 5
    assert surface.loaded_mode_count > 0
    assert surface.total_mode_count >= surface.loaded_mode_count
    assert surface.psi_p is None
    assert surface.r_n == pytest.approx(0.5)
    assert surface.r_hat == pytest.approx(surface.aminor_p * surface.r_n)
    assert surface.transport_psi_scale == pytest.approx(surface.dpsi_hat_dr_hat)
    assert surface.dr_hat_dpsi_hat == pytest.approx(1.0 / surface.dpsi_hat_dr_hat)
    geom = geometry_on_grid(surface, GridSpec(7, 9, 4))
    assert geom.surface_type == "vmec"
    assert geom.b.shape == (7, 9)
    assert jnp.all(jnp.isfinite(geom.b))
    assert jnp.all(jnp.isfinite(geom.jacobian))
    assert float(geom.b0) > 0.0


def test_vmec_radial_option_snaps_to_grid():
    requested = 0.253
    direct = load_vmec_surface(SAMPLE_WOUT, psi_n=requested, vmec_radial_option=0)
    snapped = load_vmec_surface(SAMPLE_WOUT, psi_n=requested, vmec_radial_option=1)
    assert direct.psi_n == pytest.approx(requested)
    assert snapped.requested_psi_n == pytest.approx(requested)
    assert snapped.psi_n != pytest.approx(requested)


def test_vmec_mode_filtering_reduces_mode_count():
    full = load_vmec_surface(SAMPLE_WOUT, psi_n=0.25, min_bmn_to_load=0.0)
    filtered = load_vmec_surface(SAMPLE_WOUT, psi_n=0.25, min_bmn_to_load=5e-2)
    assert filtered.loaded_mode_count < full.loaded_mode_count


def test_vmec_nyquist_option_switches_total_mode_count():
    primary = load_vmec_surface(SAMPLE_WOUT, psi_n=0.25, vmec_nyquist_option=1)
    nyquist = load_vmec_surface(SAMPLE_WOUT, psi_n=0.25, vmec_nyquist_option=2)
    assert primary.total_mode_count == 3
    assert nyquist.total_mode_count == 4
    assert primary.loaded_mode_count <= nyquist.loaded_mode_count


def test_vmec_filtered_nyquist_convention_uses_filtered_coefficients():
    reduced = load_vmec_surface(SAMPLE_WOUT, psi_n=0.25, vmec_nyquist_option=1)
    filtered = load_vmec_surface(
        SAMPLE_WOUT,
        psi_n=0.25,
        vmec_nyquist_option=1,
        vmec_mode_convention="filtered_nyquist",
    )
    assert filtered.total_mode_count >= reduced.total_mode_count
    assert np.all(np.isfinite(np.asarray(filtered.b_cos)))


def test_vmec_filtered_nyquist_matches_direct_vmec_harmonics_sign_convention():
    direct = surface_from_vmec_jax_vmec_wout_file(SAMPLE_WOUT, s=0.25)
    filtered = load_vmec_surface(
        SAMPLE_WOUT,
        psi_n=0.25,
        vmec_nyquist_option=1,
        vmec_mode_convention="filtered_nyquist",
    )
    result_direct = solve_monoenergetic(
        direct,
        GridSpec(n_theta=9, n_zeta=9, n_xi=8),
        MonoenergeticCase(nu_hat=1.0e-3, epsi_hat=0.0),
    ).as_dict()
    result_filtered = solve_monoenergetic(
        filtered,
        GridSpec(n_theta=9, n_zeta=9, n_xi=8),
        MonoenergeticCase(nu_hat=1.0e-3, epsi_hat=0.0),
    ).as_dict()
    for key, tolerance in {
        "D11": 2.0e-2,
        "D31": 2.0e-2,
        "D13": 2.0e-2,
        "D33": 6.0e-3,
    }.items():
        assert result_filtered[key] == pytest.approx(
            result_direct[key],
            rel=tolerance,
            abs=tolerance,
        )

    assert direct.psi_a_hat == pytest.approx(filtered.psi_a_hat)
    assert direct.r_hat == pytest.approx(filtered.r_hat)
    assert direct.dpsi_hat_dr_hat == pytest.approx(filtered.dpsi_hat_dr_hat)
    assert direct.dr_hat_dpsi_hat == pytest.approx(filtered.dr_hat_dpsi_hat)
    assert direct.transport_psi_scale == pytest.approx(filtered.transport_psi_scale)


def test_vmec_surface_resolves_er_hat_from_transport_scale():
    surface = load_vmec_surface(SAMPLE_WOUT, psi_n=0.25)
    er_hat = 1e-3
    result_from_er = solve_monoenergetic(
        surface,
        GridSpec(7, 9, 4),
        MonoenergeticCase(1e-3, er_hat=er_hat),
    ).as_dict()
    result_from_epsi = solve_monoenergetic(
        surface,
        GridSpec(7, 9, 4),
        MonoenergeticCase(1e-3, epsi_hat=er_hat / surface.transport_psi_scale),
    ).as_dict()
    for key in ("D11", "D31", "D13", "D33", "D33_spitzer"):
        assert result_from_er[key] == pytest.approx(result_from_epsi[key], rel=1e-12, abs=1e-12)
