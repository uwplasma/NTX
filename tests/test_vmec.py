from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
from scipy.io import netcdf_file

from ntx import GridSpec, MonoenergeticCase, load_vmec_surface, solve_monoenergetic
from ntx.geometry import VmecSurface, geometry_on_grid

VMEC_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "wout_w7x_standardConfig.nc"
QI_VMEC_FIXTURE = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "wout_QI_nfp2_stable_Er_006_000043_hires_scaled.nc"
)


def test_load_vmec_surface_and_geometry():
    surface = load_vmec_surface(VMEC_FIXTURE, psi_n=0.25)
    assert isinstance(surface, VmecSurface)
    assert surface.path == VMEC_FIXTURE.resolve()
    assert surface.nfp == 5
    assert surface.ns > 1
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
    direct = load_vmec_surface(VMEC_FIXTURE, psi_n=requested, vmec_radial_option=0)
    snapped = load_vmec_surface(VMEC_FIXTURE, psi_n=requested, vmec_radial_option=1)
    assert direct.psi_n == pytest.approx(requested)
    assert snapped.requested_psi_n == pytest.approx(requested)
    assert snapped.psi_n != pytest.approx(requested)


def test_vmec_mode_filtering_reduces_mode_count():
    full = load_vmec_surface(VMEC_FIXTURE, psi_n=0.25, min_bmn_to_load=0.0)
    filtered = load_vmec_surface(VMEC_FIXTURE, psi_n=0.25, min_bmn_to_load=1e-2)
    assert filtered.loaded_mode_count < full.loaded_mode_count


def test_qi_vmec_surface_loads_with_expected_normalization():
    surface = load_vmec_surface(QI_VMEC_FIXTURE, psi_n=0.12247**2)
    assert surface.nfp == 2
    assert surface.r_n == pytest.approx(0.12247, rel=1e-8)
    assert surface.r_hat == pytest.approx(surface.aminor_p * surface.r_n)
    assert surface.loaded_mode_count == 72
    assert surface.total_mode_count == 72
    assert surface.transport_psi_scale == pytest.approx(surface.dpsi_hat_dr_hat)


def test_vmec_nyquist_option_switches_total_mode_count():
    primary = load_vmec_surface(VMEC_FIXTURE, psi_n=0.25, vmec_nyquist_option=1)
    nyquist = load_vmec_surface(VMEC_FIXTURE, psi_n=0.25, vmec_nyquist_option=2)
    assert primary.total_mode_count == 288
    assert nyquist.total_mode_count == 574
    assert primary.loaded_mode_count < nyquist.loaded_mode_count


def test_vmec_reduced_mode_convention_truncates_coefficients_by_position():
    surface = load_vmec_surface(VMEC_FIXTURE, psi_n=0.25, vmec_nyquist_option=1)
    mode_lookup = {
        (int(m), int(n)): float(b)
        for m, n, b in zip(
            np.asarray(surface.m),
            np.asarray(surface.n),
            np.asarray(surface.b_cos),
            strict=True,
        )
    }
    assert mode_lookup[(2, -12)] == pytest.approx(-0.1290792429334510)
    assert mode_lookup[(0, 1)] == pytest.approx(0.1255195172274547)


def test_vmec_filtered_nyquist_convention_uses_filtered_nyquist_coefficients():
    surface = load_vmec_surface(
        VMEC_FIXTURE,
        psi_n=0.25,
        vmec_nyquist_option=1,
        vmec_mode_convention="filtered_nyquist",
    )
    with netcdf_file(VMEC_FIXTURE, "r", mmap=False) as handle:
        nfp = int(np.asarray(handle.variables["nfp"].data).reshape(()))
        mpol = int(np.asarray(handle.variables["mpol"].data).reshape(()))
        ntor = int(np.asarray(handle.variables["ntor"].data).reshape(()))
        xm_nyq = np.asarray(handle.variables["xm_nyq"].data, dtype=np.int32)
        xn_nyq = np.asarray(handle.variables["xn_nyq"].data, dtype=np.int32)

    include = (np.abs(xm_nyq) < mpol) & (np.abs(xn_nyq / float(nfp)) <= float(ntor))
    selected = np.nonzero(include)[0]

    assert surface.total_mode_count == selected.size
    assert np.array_equal(np.asarray(surface.m), xm_nyq[selected])
    assert np.array_equal(np.asarray(surface.n), np.rint(xn_nyq[selected] / nfp).astype(np.int32))
    mode_lookup = {
        (int(m), int(n)): float(b)
        for m, n, b in zip(
            np.asarray(surface.m),
            np.asarray(surface.n),
            np.asarray(surface.b_cos),
            strict=True,
        )
    }
    assert mode_lookup[(0, 0)] == pytest.approx(2.7846035563558886)
    assert mode_lookup[(2, -12)] == pytest.approx(-0.00014346214156386845, rel=5e-5)
    assert mode_lookup[(0, 1)] == pytest.approx(0.1255195172274547)


def test_vmec_surface_resolves_er_hat_from_transport_scale():
    surface = load_vmec_surface(VMEC_FIXTURE, psi_n=0.25)
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
