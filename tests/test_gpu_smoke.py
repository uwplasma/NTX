from __future__ import annotations

from pathlib import Path

import jax
import numpy as np
import pytest

from ntx import (
    GridSpec,
    MonoenergeticCase,
    load_dkes_surface,
    load_vmec_surface,
    solve_monoenergetic,
)

ROOT = Path(__file__).resolve().parents[1]
DKES_FIXTURE = ROOT / "tests" / "fixtures" / "w7x_eim_sample.ddkes2.data"
VMEC_FIXTURE = ROOT / "tests" / "fixtures" / "wout_w7x_standardConfig.nc"


def _require_gpu() -> None:
    if not any(device.platform == "gpu" for device in jax.devices()):
        pytest.skip("JAX GPU device not available")


@pytest.mark.gpu
def test_gpu_dkes_smoke_regression():
    _require_gpu()
    result = solve_monoenergetic(
        load_dkes_surface(DKES_FIXTURE),
        GridSpec(5, 5, 4),
        MonoenergeticCase(nu_hat=1e-5, er_hat=1e-3),
    ).as_dict()
    expected = {
        "D11": 0.0049033269042189735,
        "D31": 0.018601911381559297,
        "D13": -0.01605008327701478,
        "D33": 73.37743322156562,
        "D33_spitzer": 66287.9511900434,
    }
    for key, reference in expected.items():
        assert np.isclose(result[key], reference, rtol=1e-6, atol=1e-9), (key, result[key])


@pytest.mark.gpu
def test_gpu_vmec_smoke_regression():
    _require_gpu()
    result = solve_monoenergetic(
        load_vmec_surface(VMEC_FIXTURE, psi_n=0.25),
        GridSpec(7, 9, 4),
        MonoenergeticCase(nu_hat=1e-3, er_hat=1e-3),
    ).as_dict()
    expected = {
        "D11": 0.26059288868417236,
        "D31": -0.2671905614991318,
        "D13": 0.22917651621927108,
        "D33": 5.505770775408808,
        "D33_spitzer": 668.5079790482463,
    }
    for key, reference in expected.items():
        assert np.isclose(result[key], reference, rtol=1e-6, atol=1e-9), (key, result[key])
