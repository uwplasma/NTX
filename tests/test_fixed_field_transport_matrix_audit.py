from __future__ import annotations

from pathlib import Path

from examples import fixed_field_transport_matrix_audit as audit


def test_fixed_field_case_discovery(monkeypatch, tmp_path):
    vmec_root = tmp_path / "vmec_jax"
    data = vmec_root / "examples" / "data"
    data.mkdir(parents=True)
    qa = data / "wout_LandremanPaul2021_QA_reactorScale_lowres_reference.nc"
    qh = data / "wout_LandremanPaul2021_QH_reactorScale_lowres_reference.nc"
    qa.write_text("", encoding="utf-8")
    qh.write_text("", encoding="utf-8")

    monkeypatch.setattr(audit, "find_vmec_jax_root", lambda: vmec_root)
    cases = audit._fixed_field_cases()

    assert set(cases) == {"qa", "qh"}
    assert cases["qa"].wout_path == qa
    assert cases["qh"].wout_path == qh
    assert cases["qa"].helicity_n == 0
    assert cases["qh"].helicity_n == -1


def test_namelist_text_contains_expected_transport_matrix_settings(tmp_path):
    case = audit.FixedFieldCase(
        name="qa",
        label="QA",
        helicity_n=0,
        wout_path=tmp_path / "wout.nc",
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
    )
    monkeypatch.setattr(audit, "OUTPUT_DIR", tmp_path)
    input_path, matrix_path = audit._prepare_sfincs_jax_case(case, 0.5)
    assert input_path.exists()
    header = input_path.read_text(encoding="utf-8")
    assert header.startswith("! Fixed-field transport-matrix audit case")
    assert matrix_path.name == "transportMatrix.npy"
