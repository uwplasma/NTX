from __future__ import annotations

from pathlib import Path

import pytest

from examples import fixed_field_transport_matrix_audit as audit


def test_bridge_from_scalars_matches_expected_formulas():
    bridge = audit._bridge_from_scalars(
        surface_b0=5.9,
        psi_a_hat=8.0,
        b0_over_bbar=5.7,
        g_hat=60.0,
        i_hat=0.2,
        iota=0.4,
        nu_prime=8.31565e-3,
    )
    denom = 60.0 + 0.4 * 0.2
    assert bridge.nu_n == pytest.approx(8.31565e-3 * 5.7 / denom)
    assert bridge.factor_31 == pytest.approx(4.0 * 5.9 * 8.0 / (audit.np.sqrt(audit.np.pi) * 60.0))
    assert bridge.factor_33 == pytest.approx(2.0 * 5.9 / (denom * audit.np.sqrt(audit.np.pi)))


def test_fixed_field_case_discovery(monkeypatch, tmp_path):
    vmec_root = tmp_path / "vmec_jax"
    data = vmec_root / "examples" / "data"
    data.mkdir(parents=True)
    qa = data / "wout_LandremanPaul2021_QA_reactorScale_lowres_reference.nc"
    qh = data / "wout_LandremanPaul2021_QH_reactorScale_lowres_reference.nc"
    qa.write_text("", encoding="utf-8")
    qh.write_text("", encoding="utf-8")

    monkeypatch.setattr(audit, "find_vmec_jax_root", lambda: vmec_root)
    monkeypatch.setattr(audit, "find_qs_zenodo_root", lambda: None)
    cases = audit._fixed_field_cases()

    assert set(cases) == {"qa", "qh"}
    assert cases["qa"].wout_path == qa
    assert cases["qh"].wout_path == qh
    assert cases["qa"].helicity_n == 0
    assert cases["qh"].helicity_n == -1
    assert cases["qa"].source_family == "vmec_jax_examples"


def test_fixed_field_case_discovery_prefers_zenodo(monkeypatch, tmp_path):
    zenodo_root = (
        tmp_path
        / "20220708-01-zenodo_for_QS_optimization_with_self_consistent_bootstrap_current"
    )
    test_files = zenodo_root / "codes" / "simsopt" / "tests" / "test_files"
    test_files.mkdir(parents=True)
    qa = test_files / "wout_LandremanPaul2021_QA_reactorScale_lowres_reference.nc"
    qh = test_files / "wout_LandremanPaul2021_QH_reactorScale_lowres_reference.nc"
    qa.write_text("", encoding="utf-8")
    qh.write_text("", encoding="utf-8")

    monkeypatch.setattr(audit, "find_qs_zenodo_root", lambda: zenodo_root)
    cases = audit._fixed_field_cases()

    assert cases["qa"].wout_path == qa
    assert cases["qh"].wout_path == qh
    assert cases["qa"].source_family == "zenodo_precise_qs"


def test_namelist_text_contains_expected_transport_matrix_settings(tmp_path):
    case = audit.FixedFieldCase(
        name="qa",
        label="QA",
        helicity_n=0,
        wout_path=tmp_path / "wout.nc",
        source_family="test",
    )
    text = audit._namelist_text(case, 0.25)
    assert "RHSMode = 3" in text
    assert "geometryScheme = 5" in text
    assert 'rN_wish = 0.25000000' in text
    assert f'equilibriumFile = "{case.wout_path}"' in text
    assert "nuPrime" in text
    assert "EStar" in text


def test_prepare_sfincs_jax_case_writes_input(monkeypatch, tmp_path):
    case = audit.FixedFieldCase(
        name="qa",
        label="QA",
        helicity_n=0,
        wout_path=Path("/tmp/wout.nc"),
        source_family="test",
    )
    monkeypatch.setattr(audit, "OUTPUT_DIR", tmp_path)
    input_path, matrix_path = audit._prepare_sfincs_jax_case(case, 0.5)
    assert input_path.exists()
    header = input_path.read_text(encoding="utf-8")
    assert header.startswith("! Fixed-field transport-matrix audit case")
    assert matrix_path.name == "transportMatrix.npy"
