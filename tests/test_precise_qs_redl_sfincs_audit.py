from __future__ import annotations

import pickle

from examples import precise_qs_redl_sfincs_audit as audit


def test_precise_qs_case_discovery(monkeypatch, tmp_path):
    root = (
        tmp_path
        / "20220708-01-zenodo_for_QS_optimization_with_self_consistent_bootstrap_current"
    )
    wout_root = root / "codes" / "simsopt" / "tests" / "test_files"
    calc_root = (
        root / "calculations" / "20211226-01-sfincs_for_precise_QS_for_Redl_benchmark"
    )
    qa_scan = calc_root / "20211226-01-012_QA_Ntheta25_Nzeta39_Nxi60_Nx7_manySurfaces"
    qh_scan = calc_root / "20211226-01-019_QH_Ntheta25_Nzeta39_Nxi60_Nx7_manySurfaces"
    for directory in (wout_root, qa_scan, qh_scan):
        directory.mkdir(parents=True, exist_ok=True)
    (wout_root / "wout_LandremanPaul2021_QA_reactorScale_lowres_reference.nc").write_text(
        "", encoding="utf-8"
    )
    (wout_root / "wout_LandremanPaul2021_QH_reactorScale_lowres_reference.nc").write_text(
        "", encoding="utf-8"
    )
    (qa_scan / "sfincsScan.dat").write_bytes(b"scan")
    (qh_scan / "sfincsScan.dat").write_bytes(b"scan")

    monkeypatch.setattr(audit, "find_qs_zenodo_root", lambda: root)
    cases = audit._precise_qs_cases()

    assert set(cases) == {"qa", "qh"}
    assert cases["qa"].helicity_n == 0
    assert cases["qh"].helicity_n == -1
    assert (
        cases["qa"].wout_path.name
        == "wout_LandremanPaul2021_QA_reactorScale_lowres_reference.nc"
    )
    assert cases["qh"].sfincs_scan_path.name == "sfincsScan.dat"


def test_load_archived_sfincs_current_converts_to_si(tmp_path):
    payload = {
        "xdata": [None, [0.25, 0.5, 0.75]],
        "ydata": [None, [1.0, -2.0, 3.0]],
        "ylabels": ["something_else", "FSABjHat"],
    }
    path = tmp_path / "sfincsScan.dat"
    with path.open("wb") as handle:
        pickle.dump(payload, handle)

    surfaces, current = audit._load_archived_sfincs_current(path)

    assert surfaces.tolist() == [0.25, 0.5, 0.75]
    assert current.tolist() == [
        audit.SFINCS_SI_FACTOR,
        -2.0 * audit.SFINCS_SI_FACTOR,
        3.0 * audit.SFINCS_SI_FACTOR,
    ]


def test_reference_profiles_are_callable():
    ne, te, ti, zeff = audit._reference_profiles()
    assert float(ne(0.25)) > 0.0
    assert float(te(0.5)) > 0.0
    assert float(ti(0.5)) > 0.0
    assert zeff == 1
