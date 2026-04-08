from __future__ import annotations

from pathlib import Path

import numpy as np

from ntx.benchmarks import coefficient_errors, nearest_reference_row, read_monoenergetic_table


def test_read_reference_table_and_compare(tmp_path):
    fixture = Path(__file__).resolve().parent / "fixtures" / "reference_executable_reference_sample.dat"
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
