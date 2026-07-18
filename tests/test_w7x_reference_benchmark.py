from __future__ import annotations

import numpy as np
import pytest

from ntx import (
    GridSpec,
    MonoenergeticCase,
    build_ntx_neopax_scan,
    load_neopax_reference_scan,
    solve_monoenergetic,
    surface_from_vmex_vmec_wout_file,
)
from ntx._checkout_paths import find_neopax_root


def _w7x_paths():
    root = find_neopax_root()
    if root is None:
        return None, None
    wout = root / "tests" / "inputs" / "wout_W7-X_standard_configuration.nc"
    ref = root / "tests" / "inputs" / "Dij_NEOPAX_FULL_S_NEW_W7X.h5"
    if not wout.exists() or not ref.exists():
        return None, None
    return wout, ref


@pytest.mark.skipif(_w7x_paths()[0] is None, reason="requires local W7-X reference inputs")
def test_w7x_direct_vmec_point_matches_reference_database():
    wout, ref_path = _w7x_paths()
    assert wout is not None and ref_path is not None
    ref = load_neopax_reference_scan(ref_path)
    rho = float(ref.rho[3])
    surface = surface_from_vmex_vmec_wout_file(wout, s=float(rho**2))
    result = solve_monoenergetic(
        surface,
        GridSpec(n_theta=25, n_zeta=25, n_xi=63),
        MonoenergeticCase(nu_hat=float(ref.nu_v[3]), epsi_hat=float(ref.Es[3, 0])),
    )
    np.testing.assert_allclose(
        [result.D11, result.D31, result.D13, result.D33],
        [ref.D11[3, 3, 0], ref.D31[3, 3, 0], ref.D13[3, 3, 0], ref.D33[3, 3, 0]],
        rtol=1.0e-6,
        atol=1.0e-9,
    )


@pytest.mark.skipif(_w7x_paths()[0] is None, reason="requires local W7-X reference inputs")
def test_w7x_subset_scan_matches_reference_database():
    wout, ref_path = _w7x_paths()
    assert wout is not None and ref_path is not None
    ref = load_neopax_reference_scan(ref_path)
    rho_idx = np.array([1, 3], dtype=int)
    nu_idx = np.array([0, 3], dtype=int)
    er_idx = np.array([0, 3], dtype=int)
    scan = build_ntx_neopax_scan(
        lambda rho_value: surface_from_vmex_vmec_wout_file(wout, s=float(rho_value**2)),
        rho=np.asarray(ref.rho)[rho_idx],
        nu_v=np.asarray(ref.nu_v)[nu_idx],
        Es=np.asarray(ref.Es)[rho_idx][:, er_idx],
        Er=np.asarray(ref.Er)[rho_idx][:, er_idx],
        drds=np.asarray(ref.drds)[rho_idx],
        grid=GridSpec(n_theta=25, n_zeta=25, n_xi=63),
        source_name="w7x-reference-test",
    )
    np.testing.assert_allclose(
        np.asarray(scan.D11),
        np.asarray(ref.D11)[rho_idx][:, nu_idx][:, :, er_idx],
        rtol=1.0e-2,
        atol=1.0e-8,
    )
    np.testing.assert_allclose(
        np.asarray(scan.D13),
        np.asarray(ref.D13)[rho_idx][:, nu_idx][:, :, er_idx],
        rtol=1.0e-2,
        atol=1.0e-8,
    )
    np.testing.assert_allclose(
        np.asarray(scan.D33),
        np.asarray(ref.D33)[rho_idx][:, nu_idx][:, :, er_idx],
        rtol=1.0e-2,
        atol=1.0e-6,
    )
