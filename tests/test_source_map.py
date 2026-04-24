from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

INTERNAL_MODULES_REQUIRING_SOURCE_MAP = (
    "src/ntx/_autodiff_bootstrap.py",
    "src/ntx/_autodiff_helpers.py",
    "src/ntx/_autodiff_types.py",
    "src/ntx/_autodiff_workflows.py",
    "src/ntx/_geometry_eval.py",
    "src/ntx/_geometry_types.py",
    "src/ntx/_inputfiles_model.py",
    "src/ntx/_inputfiles_reporting.py",
    "src/ntx/_inputfiles_run.py",
    "src/ntx/_neopax_bridge.py",
    "src/ntx/_neopax_field.py",
    "src/ntx/_neopax_field_utils.py",
    "src/ntx/_neopax_fluxes.py",
    "src/ntx/_neopax_io.py",
    "src/ntx/_neopax_scan.py",
    "src/ntx/_neopax_types.py",
    "src/ntx/_neopax_vmec_jax_field.py",
    "src/ntx/_profiles_controls.py",
    "src/ntx/_profiles_eval.py",
    "src/ntx/_profiles_transport.py",
    "src/ntx/_profiles_transport_closure.py",
    "src/ntx/_profiles_types.py",
    "src/ntx/_solver_core.py",
    "src/ntx/_solver_factorization.py",
    "src/ntx/_solver_scan.py",
    "src/ntx/_solver_types.py",
)


def test_source_map_mentions_split_internal_modules() -> None:
    text = (ROOT / "docs" / "source-map.md").read_text(encoding="utf-8")

    missing = [
        module_path
        for module_path in INTERNAL_MODULES_REQUIRING_SOURCE_MAP
        if f"`{module_path}`" not in text
    ]

    assert missing == []
