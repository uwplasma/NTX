from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "make_publication_figures.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("ntx_make_publication_figures", EXAMPLE)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _invoke_main(
    module,
    monkeypatch: pytest.MonkeyPatch,
    output_dir: Path,
    figures: str,
) -> dict[str, list[str]]:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(EXAMPLE),
            "--output-dir",
            str(output_dir),
            "--figures",
            figures,
        ],
    )
    module.main()
    manifest_path = output_dir / "publication_figure_manifest.json"
    assert manifest_path.exists()
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _fake_run(command: list[str]) -> None:
    if "--output-prefix" not in command:
        return
    prefix = Path(command[command.index("--output-prefix") + 1])
    prefix.parent.mkdir(parents=True, exist_ok=True)
    prefix.with_suffix(".png").write_bytes(b"png")
    prefix.with_suffix(".pdf").write_bytes(b"pdf")
    prefix.with_suffix(".json").write_text("{}", encoding="utf-8")


def test_make_publication_figures_subset_writes_manifest(tmp_path, monkeypatch: pytest.MonkeyPatch):
    module = _load_module()
    monkeypatch.setattr(module, "_run", _fake_run)

    output_dir = tmp_path / "figures"
    payload = _invoke_main(module, monkeypatch, output_dir, "validation,science")

    assert set(payload) == {"validation", "science"}
    assert output_dir.joinpath("validation_summary.png").exists()
    assert output_dir.joinpath("validation_summary.pdf").exists()
    assert output_dir.joinpath("validation_summary.json").exists()
    assert output_dir.joinpath("bootstrap_current_optimization.png").exists()
    assert output_dir.joinpath("bootstrap_current_optimization.pdf").exists()
    assert output_dir.joinpath("bootstrap_current_optimization.json").exists()


def test_make_publication_figures_main_text_preset_writes_manifest(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
):
    module = _load_module()
    monkeypatch.setattr(module, "_run", _fake_run)

    output_dir = tmp_path / "figures"
    payload = _invoke_main(module, monkeypatch, output_dir, "main_text")

    assert set(payload) == {
        "validation",
        "closure_validation",
        "w7x_audit",
        "derivative_benchmark",
        "science",
        "performance_heavy",
        "primitive_transport",
    }
    assert any(path.endswith("validation_summary.json") for path in payload["validation"])


def test_make_publication_figures_bootstrap_subset_writes_manifest(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
):
    module = _load_module()
    monkeypatch.setattr(module, "_run", _fake_run)

    output_dir = tmp_path / "figures"
    payload = _invoke_main(module, monkeypatch, output_dir, "bootstrap_proxy")

    assert set(payload) == {"bootstrap_proxy"}
    assert output_dir.joinpath("bootstrap_current_from_vmec_or_boozmn.png").exists()
    assert output_dir.joinpath("bootstrap_current_from_vmec_or_boozmn.pdf").exists()
    assert output_dir.joinpath("bootstrap_current_from_vmec_or_boozmn.json").exists()


@pytest.mark.parametrize(
    ("figures", "expected_keys", "expected_files"),
    [
        (
            "profile_uncertainty",
            {"profile_uncertainty"},
            (
                "autodiff_profile_uncertainty.png",
                "autodiff_profile_uncertainty.pdf",
                "autodiff_profile_uncertainty.json",
            ),
        ),
        (
            "robust_science",
            {"robust_science"},
            (
                "bootstrap_current_robust_optimization.png",
                "bootstrap_current_robust_optimization.pdf",
                "bootstrap_current_robust_optimization.json",
            ),
        ),
        (
            "ambipolar",
            {"ambipolar"},
            ("ambipolar_profile.png", "ambipolar_profile.pdf"),
        ),
        (
            "ambipolar_family",
            {"ambipolar_family"},
            ("ambipolar_profile_family.png", "ambipolar_profile_family.pdf"),
        ),
        (
            "profile_control",
            {"profile_control"},
            ("profile_control_optimization.png", "profile_control_optimization.pdf"),
        ),
        (
            "profile_basis",
            {"profile_basis"},
            ("profile_basis_optimization.png", "profile_basis_optimization.pdf"),
        ),
        (
            "profile_transport",
            {"profile_transport"},
            ("profile_transport_loop.png", "profile_transport_loop.pdf"),
        ),
        (
            "primitive_transport",
            {"primitive_transport"},
            ("primitive_profile_transport.png", "primitive_profile_transport.pdf"),
        ),
        (
            "boundary_forward_mode",
            {"boundary_forward_mode"},
            (
                "boundary_forward_mode_current_derivative_benchmark.png",
                "boundary_forward_mode_current_derivative_benchmark.pdf",
                "boundary_forward_mode_current_derivative_benchmark.json",
            ),
        ),
        (
            "closure_validation",
            {"closure_validation"},
            (
                "closure_validation_report.png",
                "closure_validation_report.pdf",
                "closure_validation_report.json",
                "closure_validation_report.txt",
            ),
        ),
        (
            "file_backed_geometry_derivative",
            {"file_backed_geometry_derivative"},
            (
                "file_backed_geometry_control_derivative_benchmark.png",
                "file_backed_geometry_control_derivative_benchmark.pdf",
                "file_backed_geometry_control_derivative_benchmark.json",
            ),
        ),
        (
            "implicit_equilibrium_forward_mode",
            {"implicit_equilibrium_forward_mode"},
            (
                "implicit_equilibrium_forward_mode_derivative_benchmark.png",
                "implicit_equilibrium_forward_mode_derivative_benchmark.pdf",
                "implicit_equilibrium_forward_mode_derivative_benchmark.json",
            ),
        ),
        (
            "boundary_explicit_relaxed",
            {"boundary_explicit_relaxed"},
            (
                "explicit_relaxed_boundary_current_derivative_benchmark.png",
                "explicit_relaxed_boundary_current_derivative_benchmark.pdf",
                "explicit_relaxed_boundary_current_derivative_benchmark.json",
            ),
        ),
        (
            "geometry_family_breadth",
            {"geometry_family_breadth"},
            (
                "geometry_family_breadth_summary.png",
                "geometry_family_breadth_summary.pdf",
                "geometry_family_breadth_summary.json",
            ),
        ),
        (
            "geometry_family_transport",
            {"geometry_family_transport"},
            (
                "geometry_family_transport_convergence.png",
                "geometry_family_transport_convergence.pdf",
                "geometry_family_transport_convergence.json",
            ),
        ),
        (
            "prepared_geometry_reuse",
            {"prepared_geometry_reuse"},
            (
                "prepared_geometry_reuse_profile.png",
                "prepared_geometry_reuse_profile.pdf",
                "prepared_geometry_reuse_profile.json",
            ),
        ),
    ],
)
def test_make_publication_figures_stubbed_subset_writes_manifest(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    figures: str,
    expected_keys: set[str],
    expected_files: tuple[str, ...],
):
    module = _load_module()
    monkeypatch.setattr(module, "_run", _fake_run)

    output_dir = tmp_path / "figures"
    payload = _invoke_main(module, monkeypatch, output_dir, figures)

    assert set(payload) == expected_keys
    for path in expected_files:
        assert output_dir.joinpath(path).exists()


def test_make_publication_figures_geometry_derivative_subset_writes_manifest(tmp_path):
    output_dir = tmp_path / "figures"
    subprocess.run(
        [
            sys.executable,
            str(EXAMPLE),
            "--output-dir",
            str(output_dir),
            "--figures",
            "geometry_derivative",
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    manifest_path = output_dir / "publication_figure_manifest.json"
    assert manifest_path.exists()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert set(payload) == {"geometry_derivative"}
    assert output_dir.joinpath("geometry_control_derivative_benchmark.png").exists()
    assert output_dir.joinpath("geometry_control_derivative_benchmark.pdf").exists()
    assert output_dir.joinpath("geometry_control_derivative_benchmark.json").exists()
