from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import pytest

from ntx import GridSpec, MonoenergeticCase, load_vmec_surface, solve_monoenergetic

QI_VMEC_FIXTURE = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "wout_QI_nfp2_stable_Er_006_000043_hires_scaled.nc"
)


def test_qi_vmec_zero_field_is_finite_and_positive():
    surface = load_vmec_surface(QI_VMEC_FIXTURE, psi_n=0.12247**2)
    result = solve_monoenergetic(
        surface,
        GridSpec(9, 11, 6),
        MonoenergeticCase(nu_hat=1e-3, epsi_hat=0.0),
    )
    values = jnp.asarray([result.D11, result.D31, result.D13, result.D33, result.D33_spitzer])
    assert jnp.all(jnp.isfinite(values))
    assert result.D11 >= 0.0
    assert result.D33 > 0.0
    assert result.D33_spitzer > 0.0
    assert result.onsager_residual < 5e-4


def test_qi_vmec_er_hat_matches_explicit_epsi_hat():
    surface = load_vmec_surface(QI_VMEC_FIXTURE, psi_n=0.12247**2)
    er_hat = 1e-3
    result_from_er = solve_monoenergetic(
        surface,
        GridSpec(9, 11, 6),
        MonoenergeticCase(nu_hat=1e-3, er_hat=er_hat),
    ).as_dict()
    result_from_epsi = solve_monoenergetic(
        surface,
        GridSpec(9, 11, 6),
        MonoenergeticCase(nu_hat=1e-3, epsi_hat=er_hat / surface.transport_psi_scale),
    ).as_dict()
    for key in ("D11", "D31", "D13", "D33", "D33_spitzer"):
        assert result_from_er[key] == pytest.approx(result_from_epsi[key], rel=1e-12, abs=1e-12)
