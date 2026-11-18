from __future__ import annotations

import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_cross_code_conversion_example_prints_a_table_within_the_gate(capsys):
    runpy.run_path(
        str(ROOT / "examples" / "cross_code_conversion.py"),
        run_name="cross_code_conversion_example",
    )
    out = capsys.readouterr().out
    assert "inline factors match NTX's own" in out
    assert "against the 2% gate: pass" in out
    assert "FAIL" not in out
    assert out.count("rel. err") == 1 and "L33" in out
    assert "Beidler D*" in out and "YANCC/MONKES DKES-scaled" in out
