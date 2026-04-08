from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from ntx.benchmarks import (
    coefficient_errors,
    filter_reference_by_er_hat,
    nearest_reference_row,
    read_dkes_transport_scan,
    read_monoenergetic_table,
    read_sfincs_transport_scan,
    select_monoenergetic_row,
)
from ntx.io import load_magnetic_configuration_surface

ROOT = Path(__file__).resolve().parent
W7X_DKES = ROOT / "fixtures" / "benchmarks" / "W7X-EIM" / "dkes_results.data"
W7X_SFINCS_ER0 = ROOT / "fixtures" / "benchmarks" / "W7X-EIM" / "sfincs_er0.dat"
W7X_SFINCS_ER3E4 = ROOT / "fixtures" / "benchmarks" / "W7X-EIM" / "sfincs_er3e-4.dat"
W7X_MONO = ROOT / "fixtures" / "benchmarks" / "W7X-EIM" / "monoenergetic_reference.dat"
CIEMAT_DKES = ROOT / "fixtures" / "benchmarks" / "CIEMAT-QI" / "dkes_results.data"
CIEMAT_SFINCS_ER0 = ROOT / "fixtures" / "benchmarks" / "CIEMAT-QI" / "sfincs_er0.dat"
CIEMAT_SFINCS_ER1E3 = ROOT / "fixtures" / "benchmarks" / "CIEMAT-QI" / "sfincs_er1e-3.dat"
CIEMAT_MONO = ROOT / "fixtures" / "benchmarks" / "CIEMAT-QI" / "monoenergetic_reference.dat"
KJM_DKES = ROOT / "fixtures" / "benchmarks" / "W7X-KJM" / "dkes_results.data"
KJM_SFINCS_ER0 = ROOT / "fixtures" / "benchmarks" / "W7X-KJM" / "sfincs_er0.dat"
KJM_SFINCS_ER3E4 = ROOT / "fixtures" / "benchmarks" / "W7X-KJM" / "sfincs_er3e-4.dat"
KJM_MONO = ROOT / "fixtures" / "benchmarks" / "W7X-KJM" / "monoenergetic_reference.dat"
KJM_SURFACE = ROOT / "fixtures" / "w7x_kjm_s0204.magnetic_configuration.dat"


def test_read_reference_table_and_compare(tmp_path):
    fixture = ROOT / "fixtures" / "reference_executable_reference_sample.dat"
    path = tmp_path / "reference.dat"
    path.write_text(fixture.read_text(encoding="utf-8"), encoding="utf-8")
    table = read_monoenergetic_table(path)
    row = nearest_reference_row(table, 1e-5, 1e-3)
    errors = coefficient_errors({"D11": 1.5, "D31": 2.5, "D13": -1.5, "D33": 4.0}, row)
    assert np.isclose(row["nu_hat"], 1e-5)
    assert np.isclose(row["er_hat"], 1e-3)
    assert np.isclose(errors["D11"], 1.499049674299271)
    assert np.isclose(errors["D31"], 2.839690088888113)
    assert np.isclose(errors["D13"], -1.8396901771442339)
    assert np.isclose(errors["D33"], 10254.32482903762)


def test_read_archived_w7x_scans_in_raw_ntx_units():
    dkes = read_dkes_transport_scan(W7X_DKES)
    sfincs = read_sfincs_transport_scan(W7X_SFINCS_ER0, er_hat=0.0)
    dkes_row = nearest_reference_row(filter_reference_by_er_hat(dkes, 0.0), 1e-5)
    sfincs_row = nearest_reference_row(sfincs, 1e-5)
    assert np.isclose(dkes_row["D11"], 0.0799255)
    assert np.isclose(dkes_row["D31"], 0.359455)
    assert np.isclose(dkes_row["D33"], 28458.0)
    assert np.isclose(sfincs_row["D11"], 0.07907124664656388)
    assert np.isclose(sfincs_row["D31"], 0.3614688947048704)


def test_read_archived_ciemat_scans_in_raw_ntx_units():
    dkes = read_dkes_transport_scan(CIEMAT_DKES)
    sfincs = read_sfincs_transport_scan(CIEMAT_SFINCS_ER0, er_hat=0.0)
    dkes_row = nearest_reference_row(filter_reference_by_er_hat(dkes, 0.0), 1e-5)
    sfincs_row = nearest_reference_row(sfincs, 1e-5)
    assert np.isclose(dkes_row["D11"], 0.047964)
    assert np.isclose(dkes_row["D31"], 0.171795)
    assert np.isclose(dkes_row["D33"], 47242.5)
    assert np.isclose(sfincs_row["D11"], 0.04910335425058689)
    assert np.isclose(sfincs_row["D31"], 0.0913735365680444)


def test_read_archived_kjm_scans_in_raw_ntx_units():
    dkes = read_dkes_transport_scan(KJM_DKES)
    sfincs = read_sfincs_transport_scan(KJM_SFINCS_ER3E4, er_hat=3e-4)
    dkes_row = nearest_reference_row(filter_reference_by_er_hat(dkes, 3e-4), 1e-5)
    sfincs_row = nearest_reference_row(sfincs, 1e-5)
    assert np.isclose(dkes_row["D11"], 0.0135345)
    assert np.isclose(dkes_row["D31"], 0.26931)
    assert np.isclose(dkes_row["D33"], 34548.5)
    assert np.isclose(sfincs_row["D11"], 0.01354298105924519)
    assert np.isclose(sfincs_row["D31"], 0.271072083839074)


def test_select_monoenergetic_row_and_loader():
    table = read_monoenergetic_table(KJM_MONO)
    row = select_monoenergetic_row(
        table,
        nu_hat=1e-5,
        er_hat=3e-4,
        n_theta=19,
        n_zeta=79,
        n_xi=180,
    )
    assert np.isclose(row["D11"], 0.01375248366091503)
    assert np.isclose(row["D31"], 0.2561412005471273)

    surface = load_magnetic_configuration_surface(KJM_SURFACE)
    assert surface.nfp == 5
    assert np.isclose(surface.psi_p, -0.5132)
    assert np.isclose(surface.iota, -0.878604832424006)
    assert np.isclose(surface.b0, 2.5003)


@pytest.mark.benchmark
def test_archived_benchmark_script_runs(tmp_path):
    import json
    import subprocess
    import sys

    output_json = tmp_path / "archived-benchmarks.json"
    proc = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve().parents[1] / "scripts" / "compare_archived_benchmarks.py"),
            "--case",
            "W7X-EIM",
            "--output-json",
            str(output_json),
        ],
        check=True,
        text=True,
        capture_output=True,
        env={
            **os.environ,
            "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src"),
        },
    )
    payload = json.loads(proc.stdout)
    assert output_json.exists()
    assert [case["name"] for case in payload["cases"]] == ["W7X-EIM"]
    assert payload["cases"][0]["comparisons"][0]["monoenergetic"]["relative_error_D11"] < 1e-6
