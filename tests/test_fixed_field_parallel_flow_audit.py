from __future__ import annotations

from pathlib import Path

import f90nml
import numpy as np
import pytest

from examples import fixed_field_parallel_flow_audit as audit


def test_case_discovery_uses_zenodo_archive(monkeypatch, tmp_path):
    zenodo_root = (
        tmp_path
        / "20220708-01-zenodo_for_QS_optimization_with_self_consistent_bootstrap_current"
    )
    calc_root = (
        zenodo_root / "calculations" / "20211226-01-sfincs_for_precise_QS_for_Redl_benchmark"
    )
    wout_root = zenodo_root / "codes" / "simsopt" / "tests" / "test_files"
    (calc_root / "20211226-01-012_QA_Ntheta25_Nzeta39_Nxi60_Nx7_manySurfaces").mkdir(parents=True)
    (calc_root / "20211226-01-019_QH_Ntheta25_Nzeta39_Nxi60_Nx7_manySurfaces").mkdir(parents=True)
    wout_root.mkdir(parents=True)
    qa_wout = wout_root / "wout_LandremanPaul2021_QA_reactorScale_lowres_reference.nc"
    qh_wout = wout_root / "wout_LandremanPaul2021_QH_reactorScale_lowres_reference.nc"
    qa_scan = (
        calc_root
        / "20211226-01-012_QA_Ntheta25_Nzeta39_Nxi60_Nx7_manySurfaces"
        / "sfincsScan.dat"
    )
    qh_scan = (
        calc_root
        / "20211226-01-019_QH_Ntheta25_Nzeta39_Nxi60_Nx7_manySurfaces"
        / "sfincsScan.dat"
    )
    for path in (qa_wout, qh_wout, qa_scan, qh_scan):
        path.write_text("", encoding="utf-8")

    monkeypatch.setattr(audit, "find_qs_zenodo_root", lambda: zenodo_root)
    cases = audit._cases()

    assert cases["qa"].wout_path == qa_wout
    assert cases["qh"].sfincs_scan_path == qh_scan


def test_patched_rhsmode2_input_sets_coordinate_and_rhs(tmp_path):
    case = audit.FixedFieldCase(
        name="qa",
        label="QA",
        helicity_n=0,
        wout_path=Path("/tmp/wout.nc"),
        sfincs_scan_path=Path("/tmp/sfincsScan.dat"),
    )
    source = tmp_path / "input.namelist"
    source.write_text(
        (
            "&general\n/\n"
            "&geometryParameters\n  inputRadialCoordinate = 1\n/\n"
            "&speciesParameters\n"
            "  nHats = 4.0, 4.0\n"
            "  dnHatdrHats = -0.1, -0.1\n"
            "  THats = 9.0, 9.0\n"
            "  dTHatdrHats = -2.0, -2.0\n"
            "  Zs = 1, -1\n"
            "  mHats = 1.0, 5.45509e-4\n"
            "/\n"
        ),
        encoding="utf-8",
    )

    patched = audit._patched_rhsmode2_input(case, 0.25, source)
    nml = f90nml.read(patched)

    assert int(nml["general"]["RHSMode"]) == 2
    assert int(nml["geometryParameters"]["inputRadialCoordinate"]) == 3
    assert int(nml["geometryParameters"]["inputRadialCoordinateForGradients"]) == 4
    assert float(nml["geometryParameters"]["rN_wish"]) == pytest.approx(0.5)
    assert nml["geometryParameters"]["equilibriumFile"] == str(case.wout_path)
    assert float(nml["speciesParameters"]["Zs"]) == pytest.approx(1.0)
    assert float(nml["speciesParameters"]["mHats"]) == pytest.approx(1.0)


def test_patched_rhsmode2_input_supports_electron_and_resolution_override(monkeypatch, tmp_path):
    case = audit.FixedFieldCase(
        name="qh",
        label="QH",
        helicity_n=-1,
        wout_path=Path("/tmp/wout_qh.nc"),
        sfincs_scan_path=Path("/tmp/sfincsScan_qh.dat"),
    )
    source = tmp_path / "input_electron.namelist"
    source.write_text(
        (
            "&general\n/\n"
            "&geometryParameters\n  inputRadialCoordinate = 1\n/\n"
            "&speciesParameters\n"
            "  nHats = 4.0, 5.0\n"
            "  dnHatdrHats = -0.1, -0.2\n"
            "  THats = 9.0, 8.0\n"
            "  dTHatdrHats = -2.0, -3.0\n"
            "  Zs = 1, -1\n"
            "  mHats = 1.0, 5.45509e-4\n"
            "/\n"
            "&resolutionParameters\n"
            "  Ntheta = 25\n"
            "  Nzeta = 39\n"
            "  Nxi = 60\n"
            "  Nx = 7\n"
            "/\n"
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(audit, "RHSMODE2_SPECIES", "electron")
    monkeypatch.setattr(audit, "RHSMODE2_NTHETA", 11)
    monkeypatch.setattr(audit, "RHSMODE2_NZETA", 17)
    monkeypatch.setattr(audit, "RHSMODE2_NXI", 24)
    monkeypatch.setattr(audit, "RHSMODE2_NX", 5)

    patched = audit._patched_rhsmode2_input(case, 0.36, source)
    nml = f90nml.read(patched)

    assert float(nml["speciesParameters"]["Zs"]) == pytest.approx(-1.0)
    assert float(nml["speciesParameters"]["mHats"]) == pytest.approx(5.45509e-4)
    assert int(nml["resolutionParameters"]["Ntheta"]) == 11
    assert int(nml["resolutionParameters"]["Nzeta"]) == 17
    assert int(nml["resolutionParameters"]["Nxi"]) == 24
    assert int(nml["resolutionParameters"]["Nx"]) == 5


def test_relative_error_handles_small_reference():
    values = [2.0, 4.0]
    reference = [1.0, 0.0]
    result = audit._relative_error(values, reference)
    assert result[0] == pytest.approx(1.0)
    assert result[1] == pytest.approx(4.0 / 1.0e-16)


def test_archived_profiles_convert_sfincs_er_to_physical_kv_per_m():
    profiles = audit.ArchivedProfiles(
        psi_n=np.asarray([0.25]),
        rho=np.asarray([0.5]),
        n_hat=np.asarray([4.0]),
        t_hat=np.asarray([8.0]),
        dn_hat_drhat=np.asarray([-1.0]),
        dT_hat_drhat=np.asarray([-2.0]),
        er=np.asarray([0.001]),
        alpha=np.asarray([2.0]),
        a_hat=0.5,
    )
    assert profiles.electric_field_kv_per_m == pytest.approx([0.004])
