from __future__ import annotations

from pathlib import Path

import numpy as np

from ntx import GridSpec, MonoenergeticCase, load_vmec_surface, solve_monoenergetic

VMEC_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "wout_w7x_standardConfig.nc"
QI_VMEC_FIXTURE = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "wout_QI_nfp2_stable_Er_006_000043_hires_scaled.nc"
)


def test_vmec_regression_reference_cases():
    cases = [
        (
            VMEC_FIXTURE,
            dict(psi_n=0.25, vmec_radial_option=0, vmec_nyquist_option=1, min_bmn_to_load=0.0),
            GridSpec(9, 11, 6),
            MonoenergeticCase(nu_hat=1e-3, epsi_hat=0.0),
            {
                "D11": 0.162556723978951,
                "D31": 1.4084492461737095,
                "D13": -1.4068152723745493,
                "D33": 279.1036389200348,
                "D33_spitzer": 668.0727552476401,
                "residual_l2": 0.040619781240506296,
                "onsager_residual": 0.0016339737991601933,
            },
        ),
        (
            VMEC_FIXTURE,
            dict(psi_n=0.25, vmec_radial_option=0, vmec_nyquist_option=1, min_bmn_to_load=0.0),
            GridSpec(9, 11, 6),
            MonoenergeticCase(nu_hat=1e-3, epsi_hat=1e-3),
            {
                "D11": 0.1428048971649724,
                "D31": 1.2258994830415761,
                "D13": -1.2195457134061707,
                "D33": 288.94926892464684,
                "D33_spitzer": 668.0727552476401,
                "residual_l2": 0.03937382232407227,
                "onsager_residual": 0.00635376963540546,
            },
        ),
        (
            VMEC_FIXTURE,
            dict(psi_n=0.25, vmec_radial_option=0, vmec_nyquist_option=1, min_bmn_to_load=0.0),
            GridSpec(9, 11, 6),
            MonoenergeticCase(nu_hat=1e-3, er_hat=1e-3),
            {
                "D11": 0.13054788699643827,
                "D31": 1.1160492682188634,
                "D13": -1.107592489445247,
                "D33": 295.3205990293947,
                "D33_spitzer": 668.0727552476401,
                "residual_l2": 0.03867653567272711,
                "onsager_residual": 0.008456778773616502,
            },
        ),
        (
            VMEC_FIXTURE,
            dict(psi_n=0.253, vmec_radial_option=1, vmec_nyquist_option=2, min_bmn_to_load=1e-3),
            GridSpec(9, 11, 6),
            MonoenergeticCase(nu_hat=1e-3, epsi_hat=0.0),
            {
                "D11": 0.10029600151733206,
                "D31": 1.5178858252446186,
                "D13": -1.5589952678694008,
                "D33": 258.0508660609741,
                "D33_spitzer": 668.8721654692864,
                "residual_l2": 0.03384366004621618,
                "onsager_residual": 0.04110944262478222,
            },
        ),
        (
            QI_VMEC_FIXTURE,
            dict(
                psi_n=0.12247**2,
                vmec_radial_option=0,
                vmec_nyquist_option=1,
                min_bmn_to_load=0.0,
            ),
            GridSpec(9, 11, 6),
            MonoenergeticCase(nu_hat=1e-3, epsi_hat=0.0),
            {
                "D11": 1.785555385553368e-05,
                "D31": 0.00048195121480907905,
                "D13": -0.00019901773655429882,
                "D33": 209.9544763442989,
                "D33_spitzer": 636.1236237952869,
                "residual_l2": 0.0006053645572013163,
                "onsager_residual": 0.00028293347825478026,
            },
        ),
        (
            QI_VMEC_FIXTURE,
            dict(
                psi_n=0.12247**2,
                vmec_radial_option=0,
                vmec_nyquist_option=1,
                min_bmn_to_load=0.0,
            ),
            GridSpec(9, 11, 6),
            MonoenergeticCase(nu_hat=1e-3, er_hat=1e-3),
            {
                "D11": 1.778129748567723e-05,
                "D31": 0.00048117892794110684,
                "D13": -0.00019795231811614087,
                "D33": 209.95479247294583,
                "D33_spitzer": 636.1236237952869,
                "residual_l2": 0.0006047585659399759,
                "onsager_residual": 0.00028322660982496594,
            },
        ),
    ]
    for fixture, surface_kwargs, grid, case, expected in cases:
        surface = load_vmec_surface(fixture, **surface_kwargs)
        result = solve_monoenergetic(surface, grid, case).as_dict()
        for key, reference in expected.items():
            assert np.isclose(result[key], reference, rtol=1e-10, atol=1e-10), (key, result[key])
