from __future__ import annotations

from pathlib import Path

import pytest

from ntx import GridSpec, compare_vmec_geometry_to_sfincs
from ntx._checkout_paths import find_sfincs_jax_root

ROOT = Path(__file__).resolve().parent
W7X_VMEC = ROOT / "fixtures" / "wout_w7x_standardConfig.nc"
QI_VMEC = ROOT / "fixtures" / "wout_QI_nfp2_stable_Er_006_000043_hires_scaled.nc"
SFINCS_REPO = find_sfincs_jax_root()


pytestmark = pytest.mark.skipif(
    SFINCS_REPO is None or not SFINCS_REPO.exists(),
    reason="local sfincs_jax checkout not available",
)


@pytest.mark.parametrize(
    ("wout_path", "psi_n", "vmec_radial_option"),
    [
        (W7X_VMEC, 0.253, 1),
        (QI_VMEC, 0.12247**2, 1),
    ],
)
def test_filtered_nyquist_vmec_geometry_matches_sfincs(
    wout_path: Path,
    psi_n: float,
    vmec_radial_option: int,
):
    payload = compare_vmec_geometry_to_sfincs(
        wout_path=wout_path,
        psi_n=psi_n,
        grid=GridSpec(9, 11, 6),
        vmec_radial_option=vmec_radial_option,
        vmec_nyquist_option=1,
        min_bmn_to_load=0.0,
        sfincs_repo=SFINCS_REPO,
    )
    for field, metrics in payload["comparisons"].items():
        assert metrics["max_abs"] < 1e-10, (field, metrics)
        assert metrics["max_rel"] < 1e-10, (field, metrics)
