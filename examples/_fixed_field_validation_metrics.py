from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def jsonify(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonify(inner) for key, inner in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonify(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def least_squares_scale(reference: np.ndarray, model: np.ndarray, mask: np.ndarray) -> float:
    ref = np.asarray(reference, dtype=float)[mask]
    trial = np.asarray(model, dtype=float)[mask]
    denom = float(np.dot(trial, trial))
    if denom <= 0.0:
        return float("nan")
    return float(np.dot(ref, trial) / denom)


def relative_error_array(reference: np.ndarray, model: np.ndarray) -> np.ndarray:
    ref = np.asarray(reference, dtype=float)
    trial = np.asarray(model, dtype=float)
    return np.abs(trial - ref) / np.maximum(np.abs(ref), 1.0)


def sign_mismatch_count(reference: np.ndarray, model: np.ndarray, mask: np.ndarray) -> int:
    ref = np.asarray(reference, dtype=float)
    trial = np.asarray(model, dtype=float)
    valid = mask & (np.abs(ref) > 1.0e-12)
    return int(np.count_nonzero(np.signbit(ref[valid]) != np.signbit(trial[valid])))


__all__ = [
    "jsonify",
    "least_squares_scale",
    "relative_error_array",
    "sign_mismatch_count",
]
