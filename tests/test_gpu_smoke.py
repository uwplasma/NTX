from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

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
DKES_FIXTURE = ROOT / "tests" / "fixtures" / "sample_surface.ddkes2.data"
VMEC_FIXTURE = ROOT / "tests" / "fixtures" / "sample_wout.nc"


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
        "D11": 0.009946201075081042,
        "D31": -0.1730016494448131,
        "D13": 0.17343732611105203,
        "D33": 301.4317825260738,
        "D33_spitzer": 66281.10706157789,
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
        "D11": 0.004220692158278157,
        "D31": 0.02890447770891442,
        "D13": -0.02826287024440189,
        "D33": 318.55527391966154,
        "D33_spitzer": 665.6060710173264,
    }
    for key, reference in expected.items():
        assert np.isclose(result[key], reference, rtol=1e-6, atol=1e-9), (key, result[key])
