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
                "D11": 0.825761377818019,
                "D31": 0.3829463032022514,
                "D13": -0.6439664328905367,
                "D33": 0.708375040497854,
                "D33_spitzer": 669.1716707340555,
                "residual_l2": 0.09404151209197671,
                "onsager_residual": 0.26102012968828525,
            },
        ),
        (
            VMEC_FIXTURE,
            dict(psi_n=0.25, vmec_radial_option=0, vmec_nyquist_option=1, min_bmn_to_load=0.0),
            GridSpec(9, 11, 6),
            MonoenergeticCase(nu_hat=1e-3, epsi_hat=1e-3),
            {
                "D11": 0.29950279592711554,
                "D31": 0.3452093917613999,
                "D13": -0.5799586050713154,
                "D33": 1.1497665459683306,
                "D33_spitzer": 669.1716519260499,
                "residual_l2": 0.07030267423729003,
                "onsager_residual": 0.23474921330991544,
            },
        ),
        (
            VMEC_FIXTURE,
            dict(psi_n=0.25, vmec_radial_option=0, vmec_nyquist_option=1, min_bmn_to_load=0.0),
            GridSpec(9, 11, 6),
            MonoenergeticCase(nu_hat=1e-3, er_hat=1e-3),
            {
                "D11": 0.23938386669006056,
                "D31": 0.2975463868697594,
                "D13": -0.5592619833045818,
                "D33": 1.4963778361870492,
                "D33_spitzer": 669.1716519260499,
                "residual_l2": 0.06717995969984304,
                "onsager_residual": 0.26171559643482234,
            },
        ),
        (
            VMEC_FIXTURE,
            dict(psi_n=0.253, vmec_radial_option=1, vmec_nyquist_option=2, min_bmn_to_load=1e-3),
            GridSpec(9, 11, 6),
            MonoenergeticCase(nu_hat=1e-3, epsi_hat=0.0),
            {
                "D11": 0.052239164966525635,
                "D31": -1.0954571863121945,
                "D13": 1.125125843598367,
                "D33": 258.050866060971,
                "D33_spitzer": 668.8721654692864,
                "residual_l2": 0.033843660046216154,
                "onsager_residual": 0.02966865728617263,
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
