from __future__ import annotations

import importlib.util
from pathlib import Path

import f90nml
from netCDF4 import Dataset

ROOT = Path(__file__).resolve().parents[1]


def _write_case(root: Path, case_name: str, wout_name: str, scan_dir_name: str) -> None:
    wout_root = root / "codes" / "simsopt" / "tests" / "test_files"
    calc_root = root / "calculations" / "20211226-01-sfincs_for_precise_QS_for_Redl_benchmark"
    scan_dir = calc_root / scan_dir_name
    wout_root.mkdir(parents=True, exist_ok=True)
    scan_dir.mkdir(parents=True, exist_ok=True)
    (scan_dir / "sfincsScan.dat").write_bytes(b"scan")
    with Dataset(wout_root / wout_name, "w") as ds:
        ds.createDimension("scalar", 1)
        aminor = ds.createVariable("Aminor_p", "f8", ("scalar",))
        aminor[:] = [1.0]
    for psi_n, nhat, that, dnhat, dthat, er, alpha in (
        (0.25, 1.2, 1.1, -0.4, -0.6, 0.10, 0.25),
        (0.49, 1.0, 0.95, -0.5, -0.5, 0.06, 0.22),
        (0.81, 0.82, 0.78, -0.45, -0.4, 0.03, 0.20),
    ):
        surface_dir = scan_dir / f"psiN_{psi_n}"
        surface_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "speciesParameters": {
                "nHats": [nhat, nhat],
                "tHats": [that, that],
                "dnhatdrhats": [dnhat, dnhat],
                "dthatdrhats": [dthat, dthat],
            },
            "physicsParameters": {
                "Er": er,
                "alpha": alpha,
            },
        }
        f90nml.write(payload, surface_dir / "input.namelist", force=True)


def test_profile_force_reconstruction_audit_writes_outputs(tmp_path, monkeypatch):
    example_path = ROOT / "examples" / "profile_force_reconstruction_audit.py"
    spec = importlib.util.spec_from_file_location(
        "ntx_profile_force_reconstruction_audit_example",
        example_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    archive_root = (
        tmp_path
        / "20220708-01-zenodo_for_QS_optimization_with_self_consistent_bootstrap_current"
    )
    _write_case(
        archive_root,
        "qa",
        "wout_LandremanPaul2021_QA_reactorScale_lowres_reference.nc",
        "20211226-01-012_QA_Ntheta25_Nzeta39_Nxi60_Nx7_manySurfaces",
    )
    _write_case(
        archive_root,
        "qh",
        "wout_LandremanPaul2021_QH_reactorScale_lowres_reference.nc",
        "20211226-01-019_QH_Ntheta25_Nzeta39_Nxi60_Nx7_manySurfaces",
    )

    monkeypatch.setattr(module, "find_qs_zenodo_root", lambda: archive_root)
    output_prefix = tmp_path / "profile_force_reconstruction_audit"
    module.main(output_prefix)

    assert output_prefix.with_suffix(".png").exists()
    assert output_prefix.with_suffix(".pdf").exists()
    assert output_prefix.with_suffix(".json").exists()
