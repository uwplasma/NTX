from __future__ import annotations

from pathlib import Path

import numpy as np

from ntx import GridSpec, MonoenergeticCase, load_vmec_surface, solve_monoenergetic

VMEC_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "wout_w7x_standardConfig.nc"


def test_vmec_regression_reference_cases():
    cases = [
        (
            dict(psi_n=0.25, vmec_radial_option=0, vmec_nyquist_option=1, min_bmn_to_load=0.0),
            GridSpec(9, 11, 6),
            MonoenergeticCase(nu_hat=1e-3, epsi_hat=0.0),
            {
                "D11": 0.08297430449915999,
                "D31": -1.0062607729587805,
                "D13": 1.0050933870962055,
                "D33": 279.1036389200348,
                "D33_spitzer": 668.0727552476401,
                "residual_l2": 0.040619781240506296,
                "onsager_residual": 0.0011673858625749212,
            },
        ),
        (
            dict(psi_n=0.25, vmec_radial_option=0, vmec_nyquist_option=1, min_bmn_to_load=0.0),
            GridSpec(9, 11, 6),
            MonoenergeticCase(nu_hat=1e-3, epsi_hat=1e-3),
            {
                "D11": 0.07289232171578433,
                "D31": -0.8758388452593514,
                "D13": 0.8712994206674506,
                "D33": 288.94926892464684,
                "D33_spitzer": 668.0727552476401,
                "residual_l2": 0.03937382232407227,
                "onsager_residual": 0.00453942459190082,
            },
        ),
        (
            dict(psi_n=0.253, vmec_radial_option=1, vmec_nyquist_option=2, min_bmn_to_load=1e-3),
            GridSpec(9, 11, 6),
            MonoenergeticCase(nu_hat=1e-3, epsi_hat=0.0),
            {
                "D11": 0.05223916496652564,
                "D31": -1.0954571863122728,
                "D13": 1.1251258435983715,
                "D33": 258.0508660609741,
                "D33_spitzer": 668.8721654692864,
                "residual_l2": 0.03384366004621618,
                "onsager_residual": 0.029668657286098687,
            },
        ),
    ]
    for surface_kwargs, grid, case, expected in cases:
        surface = load_vmec_surface(VMEC_FIXTURE, **surface_kwargs)
        result = solve_monoenergetic(surface, grid, case).as_dict()
        for key, reference in expected.items():
            assert np.isclose(result[key], reference, rtol=1e-10, atol=1e-10), (key, result[key])
