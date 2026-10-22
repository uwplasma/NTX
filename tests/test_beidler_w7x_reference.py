"""NTX monoenergetic coefficients on the W7-X standard configuration against the ICNTS data.

``tests/fixtures/beidler2011_w7x_monoenergetic.csv`` holds D11*, D31* and D33* of Beidler et al.,
Nucl. Fusion 51, 076001 (2011), Figs 17, 26 and 21 (rho = 0.5), digitized from the paper (method
and errors in its header). The NEO-2 circles are the reference at v_E = 0; at v_E > 0 the reference
is the cluster of DKES and Monte Carlo symbols, whose half extent is the scatter between the codes.

The surface is ``w7x_standard_rho05.ddkes2.data``: the 52 largest Boozer harmonics of the W7-X
standard configuration at rho = 0.5 (iota 0.8719, paper 0.8700). Tolerances: D11* 10% (3%
digitization, the rest the scatter of the ICNTS codes), D31* 0.01 absolute, D33* 2.5%.

Normalization of the DKES-unit results, with the effective radius r = sqrt(2 psi / B_00) and the
toroidal flux psi / 2pi at rho = 0.5 of the equilibrium file, so that nu* and D* are the ones DKX
reports for the same surface:

    nu* = R0 nu / (iota v),  R0 = |B_zeta| / B_00,  E_r / v = v_E psi_p / r
    D11* = D11 iota R0 B_00^2 / (pi/8) / (B_00 r / psi_p)^2
    D31* = 3 iota psi_p D31 / (2 R0 f_t),  f_t = 1.46 sqrt(r / R0)
    D33* = D33 / D33_Spitzer.

Measured deviations: D11* +1.5 to +7.0% (nu* 1.1e-2 to 1.1, and 1.1e-3 at 21 x 45 x 64), D31* <= 0.007, D33* <= 1.6%; at v_E = 3e-3 and nu* = 1.8e-4, +17% of the cluster centre (half extent 25%).
"""

from __future__ import annotations

import csv
from pathlib import Path

import jax
import numpy as np
import pytest

import ntx

jax.config.update("jax_enable_x64", True)

FIXTURES = Path(__file__).parent / "fixtures"
PSI_HAT = 0.0962339514814373  # toroidal flux / 2 pi at rho = 0.5 of the equilibrium file
TOL = {"D11": ("rel", 0.10), "D31": ("abs", 0.01), "D33": ("rel", 0.025)}


def _rows(v_e: float, nu_star: float, coefficients=("D11", "D31", "D33")) -> dict:
    lines = (FIXTURES / "beidler2011_w7x_monoenergetic.csv").read_text().splitlines()
    return {
        r["coef"]: (float(r["value"]), float(r["abs_err"]))
        for r in csv.DictReader(line for line in lines if not line.startswith("#"))
        if float(r["v_E"]) == v_e and float(r["nu_star"]) == nu_star and r["coef"] in coefficients
    }


def _solve(grid: tuple[int, int, int], nu_star: float, v_e: float) -> dict:
    surface = ntx.load_dkes_surface(FIXTURES / "w7x_standard_rho05.ddkes2.data")
    iota, psi_p, b00 = abs(float(surface.iota)), float(surface.psi_p), float(surface.b0)
    r0, r = abs(float(surface.b_zeta)) / b00, np.sqrt(2 * PSI_HAT / b00)
    result = ntx.solve_monoenergetic(
        surface,
        ntx.GridSpec(*grid),
        ntx.MonoenergeticCase(nu_hat=nu_star * iota / r0, er_hat=v_e * psi_p / r),
    )
    f_t = 1.46 * np.sqrt(r / r0)
    return {
        "D11": float(result.D11) * iota * r0 * b00**2 / (np.pi / 8) / (b00 * r / psi_p) ** 2,
        "D31": 3 * iota * psi_p * float(result.D31) / (2 * r0 * f_t),
        "D33": float(result.D33) / float(result.D33_spitzer),
    }


@pytest.mark.parametrize("nu_star", [1.09e-3, 1.09e-2, 1.09e-1, 1.09])
def test_w7x_standard_matches_neo2_at_zero_electric_field(nu_star: float) -> None:
    got = _solve((21, 45, 64), nu_star, 0.0)
    for coef, (ref, _err) in _rows(0.0, nu_star).items():
        kind, tol = TOL[coef]
        dev = abs(got[coef] - ref) / (abs(ref) if kind == "rel" else 1.0)
        assert dev < tol, (nu_star, coef, got[coef], ref)


def test_w7x_standard_is_inside_the_code_scatter_at_finite_electric_field() -> None:
    got = _solve((29, 65, 192), 1.82e-4, 3e-3)
    ref, half_extent = _rows(3e-3, 1.82e-4, ("D11",))["D11"]
    assert abs(got["D11"] - ref) < half_extent, (got["D11"], ref, half_extent)
