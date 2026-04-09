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
                    "D11": 0.030131189568414086,
                    "D31": -0.6289184074467363,
                    "D13": 0.7748107582474107,
                    "D33": 139.635606893064,
                    "D33_spitzer": 668.925580843679,
                    "residual_l2": 0.030539927458272224,
                    "onsager_residual": 0.14589235080067442,
                },
            ),
        (
            VMEC_FIXTURE,
            dict(psi_n=0.25, vmec_radial_option=0, vmec_nyquist_option=1, min_bmn_to_load=0.0),
                GridSpec(9, 11, 6),
                MonoenergeticCase(nu_hat=1e-3, epsi_hat=1e-3),
                {
                    "D11": 0.01659440463400121,
                    "D31": -0.3078748197744295,
                    "D13": 0.39657830735746025,
                    "D33": 168.8501859228225,
                    "D33_spitzer": 668.925580843679,
                    "residual_l2": 0.025469066499417974,
                    "onsager_residual": 0.08870348758303076,
                },
            ),
        (
            VMEC_FIXTURE,
            dict(psi_n=0.25, vmec_radial_option=0, vmec_nyquist_option=1, min_bmn_to_load=0.0),
                GridSpec(9, 11, 6),
                MonoenergeticCase(nu_hat=1e-3, er_hat=1e-3),
                {
                    "D11": 0.012981956084321929,
                    "D31": -0.2185620416765943,
                    "D13": 0.28754097090590336,
                    "D33": 180.5577721607964,
                    "D33_spitzer": 668.925580843679,
                    "residual_l2": 0.03341326543109028,
                    "onsager_residual": 0.06897892922930907,
                },
            ),
        (
            VMEC_FIXTURE,
            dict(psi_n=0.253, vmec_radial_option=1, vmec_nyquist_option=2, min_bmn_to_load=1e-3),
                GridSpec(9, 11, 6),
                MonoenergeticCase(nu_hat=1e-3, epsi_hat=0.0),
                {
                    "D11": 0.05962971615875646,
                    "D31": -1.1360823153296393,
                    "D13": 1.1708144703728343,
                    "D33": 264.5986412904613,
                    "D33_spitzer": 668.7250435369303,
                    "residual_l2": 0.03624258684228605,
                    "onsager_residual": 0.034732155043194934,
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
                "D11": 1.2968746235477908e-05,
                "D31": -0.0005793508186733098,
                "D13": 0.000977591031334808,
                "D33": 209.7577924584449,
                "D33_spitzer": 636.1215082509578,
                "residual_l2": 0.000562241888137363,
                "onsager_residual": 0.00039824021266149834,
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
                "D11": 1.294881489660712e-05,
                "D31": -0.0005745221082971282,
                "D13": 0.0009763662334620501,
                "D33": 209.75848276861606,
                "D33_spitzer": 636.1215082509578,
                "residual_l2": 0.0005620431930622537,
                "onsager_residual": 0.000401844125164922,
            },
        ),
    ]
    for fixture, surface_kwargs, grid, case, expected in cases:
        surface = load_vmec_surface(fixture, **surface_kwargs)
        result = solve_monoenergetic(surface, grid, case).as_dict()
        for key, reference in expected.items():
            assert np.isclose(result[key], reference, rtol=5e-6, atol=1e-9), (
                key,
                result[key],
            )
