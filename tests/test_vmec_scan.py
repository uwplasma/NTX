from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np

from ntx import (
    GridSpec,
    MonoenergeticCase,
    load_vmec_surface,
    solve_monoenergetic,
    solve_monoenergetic_scan,
)

ROOT = Path(__file__).resolve().parent
W7X_VMEC = ROOT / "fixtures" / "wout_w7x_standardConfig.nc"
QI_VMEC = ROOT / "fixtures" / "wout_QI_nfp2_stable_Er_006_000043_hires_scaled.nc"


def test_w7x_vmec_er_hat_scan_matches_loop_and_regression():
    surface = load_vmec_surface(W7X_VMEC, psi_n=0.25)
    grid = GridSpec(9, 11, 6)
    nu = jnp.asarray([1e-4, 1e-3, 1e-2])
    scan = solve_monoenergetic_scan(surface, grid, nu, er_hat=jnp.full((3,), 1e-3))

    loop = [
        solve_monoenergetic(surface, grid, MonoenergeticCase(float(nu_hat), er_hat=1e-3)).as_dict()
        for nu_hat in nu
    ]
    assert np.allclose(scan["D11"], [entry["D11"] for entry in loop], rtol=1e-9, atol=1e-12)
    assert np.allclose(scan["D31"], [entry["D31"] for entry in loop], rtol=1e-8, atol=1e-12)
    assert np.allclose(scan["D33"], [entry["D33"] for entry in loop], rtol=1e-9, atol=1e-10)

    assert np.allclose(
        np.asarray(scan["D11"]),
        np.asarray([0.11463314117197669, 0.13054788699643827, 0.028693555883509558]),
        rtol=1e-8,
        atol=1e-12,
    )
    assert np.allclose(
        np.asarray(scan["D31"]),
        np.asarray([0.6493782796612592, 1.1160492682188634, 0.3038582877699687]),
        rtol=2e-8,
        atol=1e-12,
    )


def test_qi_vmec_er_hat_scan_matches_explicit_epsi_hat_scan():
    surface = load_vmec_surface(QI_VMEC, psi_n=0.12247**2)
    grid = GridSpec(9, 11, 6)
    nu = jnp.asarray([1e-4, 1e-3, 1e-2])
    er_hat = jnp.asarray([1e-3, 1e-3, 1e-3])
    epsi_hat = er_hat / surface.transport_psi_scale

    er_scan = solve_monoenergetic_scan(surface, grid, nu, er_hat=er_hat)
    epsi_scan = solve_monoenergetic_scan(surface, grid, nu, epsi_hat=epsi_hat)

    for key in ("D11", "D31", "D13", "D33", "D33_spitzer"):
        assert np.allclose(
            np.asarray(er_scan[key]),
            np.asarray(epsi_scan[key]),
            rtol=1e-12,
            atol=1e-12,
        )

    assert np.allclose(
        np.asarray(er_scan["D11"]),
        np.asarray([1.1448629570228259e-04, 1.778129748567723e-05, 1.0368740974298214e-05]),
        rtol=1e-9,
        atol=1e-12,
    )
    assert np.allclose(
        np.asarray(er_scan["D33"]),
        np.asarray([826.5374315404677, 209.95479247294583, 45.90534950254614]),
        rtol=1e-9,
        atol=1e-9,
    )
