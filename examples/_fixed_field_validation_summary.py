from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from _fixed_field_validation_metrics import jsonify


def build_fixed_field_summary_payload(
    *,
    results: dict[str, dict[str, dict[str, np.ndarray]]],
    cases: dict[str, Any],
    output_prefix: Path,
    interior_rho_min: float,
    interior_rho_max: float,
    closure_diagnostics: Callable[[Any, dict[str, dict[str, np.ndarray]]], dict[str, Any]],
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "cases": {},
        "figure_png": str(output_prefix.with_suffix(".png")),
        "figure_pdf": str(output_prefix.with_suffix(".pdf")),
    }
    for key, case_results in results.items():
        ref = np.asarray(case_results["SFINCS"]["jdotb"], dtype=float)
        ref_scale = np.maximum(np.abs(ref), 1.0)
        rho_ref = np.asarray(case_results["SFINCS"]["rho"], dtype=float)
        interior = (rho_ref >= interior_rho_min) & (rho_ref <= interior_rho_max)
        out["cases"][key] = {
            name: {subkey: jsonify(value) for subkey, value in payload.items()}
            for name, payload in case_results.items()
        }
        out["cases"][key]["max_relative_error_vs_sfincs"] = {
            name: float(np.max(np.abs(np.asarray(payload["jdotb"], dtype=float) - ref) / ref_scale))
            for name, payload in case_results.items()
            if name != "SFINCS"
        }
        out["cases"][key]["max_relative_error_vs_sfincs_interior"] = {
            name: float(
                np.max(
                    np.abs(np.asarray(payload["jdotb"], dtype=float)[interior] - ref[interior])
                    / np.maximum(np.abs(ref[interior]), 1.0)
                )
            )
            for name, payload in case_results.items()
            if name != "SFINCS"
        }
        out["cases"][key]["closure_diagnostics"] = closure_diagnostics(
            cases[key],
            case_results,
        )
    return out


__all__ = ["build_fixed_field_summary_payload"]
