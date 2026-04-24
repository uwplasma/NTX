from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "scripts" / "test_lane_manifest.py"


def _load_manifest_module():
    spec = importlib.util.spec_from_file_location("ntx_test_lane_manifest", MANIFEST)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_ci_lane_manifest_covers_every_test_file_once():
    module = _load_manifest_module()
    assert module.validate_manifest(ROOT) == ()


def test_new_benchmark_examples_are_not_in_core_lane():
    module = _load_manifest_module()
    core = set().union(
        module.LANES["core_foundation"],
        module.LANES["core_cli_workflows"],
        module.LANES["core_io_workflows"],
        module.LANES["core_parallel_workflows"],
        module.LANES["core_neopax_workflows"],
        module.LANES["core_profile_audit_workflow"],
        module.LANES["core_profile_basic_workflows"],
        module.LANES["core_profile_optimization_workflows"],
        module.LANES["core_profile_transport_workflows"],
        module.LANES["core_autodiff_uncertainty_workflow"],
        module.LANES["core_robust_bootstrap_workflow"],
        module.LANES["core_validation"],
    )
    heavy = set().union(
        module.LANES["heavy_examples_derivatives"],
        module.LANES["heavy_examples_boundary"],
        module.LANES["heavy_examples_publication"],
    )

    benchmark_examples = {
        "tests/test_boundary_forward_mode_current_derivative_benchmark_example.py",
        "tests/test_explicit_relaxed_boundary_current_derivative_benchmark_example.py",
        "tests/test_file_backed_geometry_control_derivative_benchmark_example.py",
        "tests/test_geometry_control_derivative_benchmark_example.py",
        "tests/test_implicit_equilibrium_forward_mode_derivative_benchmark_example.py",
    }
    assert benchmark_examples <= heavy
    assert benchmark_examples.isdisjoint(core)
