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
)

ROOT = Path(__file__).resolve().parent
W7X_DKES = ROOT / "fixtures" / "benchmarks" / "W7X-EIM" / "dkes_results.data"
W7X_SFINCS_ER0 = ROOT / "fixtures" / "benchmarks" / "W7X-EIM" / "sfincs_er0.dat"
W7X_SFINCS_ER3E4 = ROOT / "fixtures" / "benchmarks" / "W7X-EIM" / "sfincs_er3e-4.dat"
CIEMAT_DKES = ROOT / "fixtures" / "benchmarks" / "CIEMAT-QI" / "dkes_results.data"
CIEMAT_SFINCS_ER0 = ROOT / "fixtures" / "benchmarks" / "CIEMAT-QI" / "sfincs_er0.dat"

W7X_PSI_SCALE = 0.5237
CIEMAT_PSI_SCALE = 0.4674


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
    dkes = read_dkes_transport_scan(
        W7X_DKES,
        d11_scale=W7X_PSI_SCALE ** -2,
        d31_scale=W7X_PSI_SCALE ** -1,
    )
    sfincs = read_sfincs_transport_scan(
        W7X_SFINCS_ER0,
        er_hat=0.0,
        d11_scale=W7X_PSI_SCALE ** -2,
        d31_scale=W7X_PSI_SCALE ** -1,
    )
    dkes_row = nearest_reference_row(filter_reference_by_er_hat(dkes, 0.0), 1e-5)
    sfincs_row = nearest_reference_row(sfincs, 1e-5)
    assert np.isclose(dkes_row["D11"], 0.2914205771867007)
    assert np.isclose(dkes_row["D31"], 0.6863757876646934)
    assert np.isclose(dkes_row["D33"], 28458.0)
    assert np.isclose(sfincs_row["D11"], 0.28830583902025786)
    assert np.isclose(sfincs_row["D31"], 0.6902212997992561)


def test_read_archived_ciemat_scans_in_raw_ntx_units():
    dkes = read_dkes_transport_scan(
        CIEMAT_DKES,
        d11_scale=CIEMAT_PSI_SCALE ** -2,
        d31_scale=CIEMAT_PSI_SCALE ** -1,
    )
    sfincs = read_sfincs_transport_scan(
        CIEMAT_SFINCS_ER0,
        er_hat=0.0,
        d11_scale=CIEMAT_PSI_SCALE ** -2,
        d31_scale=CIEMAT_PSI_SCALE ** -1,
    )
    dkes_row = nearest_reference_row(filter_reference_by_er_hat(dkes, 0.0), 1e-4)
    sfincs_row = nearest_reference_row(sfincs, 1e-5)
    assert np.isclose(dkes_row["D11"], 0.023460062483875974)
    assert np.isclose(dkes_row["D31"], 0.012609221223791186)
    assert np.isclose(dkes_row["D33"], 4633.700000000001)
    assert np.isclose(sfincs_row["D11"], 0.22476761829149688)
    assert np.isclose(sfincs_row["D31"], 0.19549323185289774)


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
    assert [case["name"] for case in payload["cases"]] == ["W7X-EIM", "CIEMAT-QI"]
    assert payload["cases"][0]["comparisons"][0]["dkes"]["relative_error_D11"] > 0.0
