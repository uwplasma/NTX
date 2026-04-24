from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
if str(EXAMPLES) not in sys.path:
    sys.path.insert(0, str(EXAMPLES))

from _fixed_field_validation_metrics import (  # noqa: E402
    jsonify,
    least_squares_scale,
    relative_error_array,
    sign_mismatch_count,
)


def test_fixed_field_validation_metric_helpers_are_masked_and_scaled() -> None:
    reference = np.asarray([2.0, 4.0, -6.0, 8.0])
    model = np.asarray([1.0, 2.0, 3.0, -4.0])
    mask = np.asarray([True, True, False, False])

    assert least_squares_scale(reference, model, mask) == pytest.approx(2.0)
    assert math.isnan(least_squares_scale(reference, np.zeros_like(model), mask))

    relative = relative_error_array(reference, model)
    assert relative.tolist() == pytest.approx([0.5, 0.5, 1.5, 1.5])
    assert relative_error_array(np.asarray([0.0]), np.asarray([0.25])).item() == (
        pytest.approx(0.25)
    )

    assert sign_mismatch_count(reference, model, np.ones_like(mask, dtype=bool)) == 2
    assert sign_mismatch_count(
        np.asarray([0.0, 1.0, -1.0]),
        np.asarray([-1.0, -1.0, -2.0]),
        np.asarray([True, True, True]),
    ) == 1


def test_fixed_field_jsonify_converts_artifact_payload_types() -> None:
    payload = {
        "path": ROOT / "docs",
        "array": np.asarray([1.0, 2.0]),
        "scalar": np.float64(3.0),
        "nested": (np.int64(4),),
    }

    assert jsonify(payload) == {
        "path": str(ROOT / "docs"),
        "array": [1.0, 2.0],
        "scalar": 3.0,
        "nested": [4],
    }
