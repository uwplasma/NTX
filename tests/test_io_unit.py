from __future__ import annotations

from pathlib import Path

import pytest

from ntx.io import (
    _parse_float,
    _parse_scalar,
    load_boozer_modes_csv,
    load_dkes_surface,
    load_magnetic_configuration_surface,
    write_result_jsonable,
)

ROOT = Path(__file__).resolve().parents[1]
DKES = ROOT / "tests" / "fixtures" / "sample_surface.ddkes2.data"
MAGNETIC = ROOT / "tests" / "fixtures" / "sample_magnetic_configuration.dat"


def test_load_boozer_modes_csv_accepts_whitespace_table(tmp_path):
    path = tmp_path / "modes.dat"
    path.write_text("m n b_cos\n0 0 2.0\n1 0 0.1\n", encoding="utf-8")
    surface = load_boozer_modes_csv(path, nfp=1, iota=0.4, psi_p=1.0, b_theta=0.2, b_zeta=1.3)
    assert surface.nfp == 1
    assert float(surface.b0) == 2.0
    assert surface.stellarator_symmetric


def test_load_boozer_modes_csv_rejects_missing_columns(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("m n\n0 0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="expected columns"):
        load_boozer_modes_csv(path, nfp=1, iota=0.4, psi_p=1.0, b_theta=0.2, b_zeta=1.3)


def test_dkes_and_magnetic_loaders_and_jsonable():
    dkes = load_dkes_surface(DKES)
    magnetic = load_magnetic_configuration_surface(MAGNETIC)
    assert dkes.nfp == 2
    assert magnetic.nfp == 2
    result = write_result_jsonable(type("R", (), {"as_dict": lambda self: {"D11": 1.0}})())
    assert result == {"D11": 1.0}


def test_dkes_loader_rejects_missing_modes(tmp_path):
    path = tmp_path / "bad.ddkes2.data"
    path.write_text("&datain nzperiod=1, psip=1, chip=1, btheta=1, bzeta=1 /\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no borbi"):
        load_dkes_surface(path)


def test_magnetic_loader_rejects_missing_section(tmp_path):
    path = tmp_path / "bad.magnetic_configuration.dat"
    path.write_text(
        "\n".join(
            [
                "Number of periods = 1",
                "psi_p = 1",
                "chi_p = 1",
                "iota = 1",
                "B00 = 1",
                "B_theta = 1",
                "B_zeta = 1",
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="missing Fourier-mode section"):
        load_magnetic_configuration_surface(path)


def test_parse_scalar_and_float_helpers():
    assert _parse_scalar("value = 1.5D+0", "value") == 1.5
    assert _parse_float("1.5d+0") == 1.5
    with pytest.raises(ValueError, match="missing `x`"):
        _parse_scalar("value = 1", "x")
