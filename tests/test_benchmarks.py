from __future__ import annotations

import numpy as np

from ntx.benchmarks import coefficient_errors, nearest_reference_row, read_monoenergetic_table


def test_read_reference_table_and_compare(tmp_path):
    path = tmp_path / "reference.dat"
    path.write_text(
        "nu er nt nz nx D11 D31 D13 D33 D33s wall cpu\n"
        "1e-2 0.0 5 5 4 1.0 2.0 -2.0 3.0 4.0 0.1 0.2\n"
        "1e-1 0.0 5 5 4 10.0 20.0 -20.0 30.0 40.0 0.1 0.2\n",
        encoding="utf-8",
    )
    table = read_monoenergetic_table(path)
    row = nearest_reference_row(table, 1.1e-2, 0.0)
    errors = coefficient_errors({"D11": 1.5, "D31": 2.5, "D13": -1.5, "D33": 4.0}, row)
    assert np.isclose(row["nu_hat"], 1e-2)
    assert errors == {"D11": 0.5, "D31": 0.5, "D13": 0.5, "D33": 1.0}
