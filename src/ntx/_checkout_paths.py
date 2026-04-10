"""Helpers for locating sibling research-stack checkouts without hard-coded paths."""

from __future__ import annotations

import os
from pathlib import Path


def repo_root() -> Path:
    """Return the NTX repository root."""

    return Path(__file__).resolve().parents[2]


def workspace_root() -> Path:
    """Return the parent directory that commonly holds sibling checkouts."""

    return repo_root().parent


def fixture_path(*parts: str) -> Path:
    """Return a path under the repository test fixtures directory."""

    return repo_root() / "tests" / "fixtures" / Path(*parts)


def _discover(env_var: str, *relative_candidates: str) -> Path | None:
    env_value = os.environ.get(env_var)
    candidates: list[Path] = []
    if env_value:
        candidates.append(Path(env_value).expanduser())
    root = workspace_root()
    candidates.extend(root / candidate for candidate in relative_candidates)
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.exists():
            return resolved
    return None


def _workspace_checkout_candidates() -> list[Path]:
    root = workspace_root()
    repo = repo_root().resolve()
    candidates = [entry for entry in root.iterdir() if entry.is_dir() and entry.resolve() != repo]
    tests_dir = root / "tests"
    if tests_dir.exists():
        candidates.extend(
            entry for entry in tests_dir.iterdir() if entry.is_dir() and entry.resolve() != repo
        )
    return candidates


def find_booz_xform_jax_root() -> Path | None:
    return _discover("BOOZ_XFORM_JAX_ROOT", "booz_xform_jax", "tests/booz_xform_jax")


def find_reference_python_root() -> Path | None:
    root = _discover("NTX_REFERENCE_PYTHON_ROOT", "tests/reference_python", "reference_python")
    if root is not None:
        return root
    signature = (
        Path("Examples")
        / "DKES_like_database"
        / "Test_Monoenergetic_database_VMEC_s_coordinate_W7X.py"
    )
    for candidate in _workspace_checkout_candidates():
        if (candidate / signature).exists():
            return candidate.resolve()
    return None


def find_reference_executable_root() -> Path | None:
    root = _discover(
        "NTX_REFERENCE_EXECUTABLE_ROOT",
        "tests/reference_executable",
        "reference_executable",
    )
    if root is not None:
        return root
    for candidate in _workspace_checkout_candidates():
        bin_dir = candidate / "bin"
        if not bin_dir.exists():
            continue
        matches = sorted(bin_dir.glob("main_*.x"))
        if matches:
            return candidate.resolve()
    return None


def find_reference_executable() -> Path | None:
    env_value = os.environ.get("NTX_REFERENCE_EXECUTABLE")
    if env_value:
        candidate = Path(env_value).expanduser().resolve()
        return candidate if candidate.exists() else None
    root = find_reference_executable_root()
    if root is None:
        return None
    matches = sorted((root / "bin").glob("main_*.x"))
    return matches[0].resolve() if matches else None


def find_neopax_root() -> Path | None:
    return _discover("NEOPAX_ROOT", "tests/NEOPAX", "NEOPAX")


def find_omnigenity_optimization_root() -> Path | None:
    return _discover(
        "OMNIGENITY_OPTIMIZATION_ROOT",
        "omnigenity_optimization",
        "tests/omnigenity_optimization",
    )


def find_sfincs_jax_root() -> Path | None:
    return _discover("SFINCS_JAX_ROOT", "tests/sfincs_jax", "sfincs_jax")


def find_simsopt_root() -> Path | None:
    return _discover("SIMSOPT_ROOT", "tests/simsopt", "simsopt")


def find_vmec_jax_root() -> Path | None:
    return _discover("VMEC_JAX_ROOT", "vmec_jax", "tests/vmec_jax")


def find_vmec_jax_example_input(name: str = "input.circular_tokamak") -> Path | None:
    root = find_vmec_jax_root()
    if root is None:
        return None
    candidate = root / "examples" / "data" / name
    return candidate if candidate.exists() else None
