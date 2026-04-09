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


def find_booz_xform_jax_root() -> Path | None:
    return _discover("BOOZ_XFORM_JAX_ROOT", "booz_xform_jax", "tests/booz_xform_jax")


def find_reference_executable_edu_root() -> Path | None:
    return _discover("REFERENCE_EXECUTABLE_EDU_ROOT", "tests/reference_executable_edu", "reference_executable_edu")


def find_reference_executable_f0_root() -> Path | None:
    return _discover("REFERENCE_EXECUTABLE_F0_ROOT", "tests/reference_executable_f0", "reference_executable_f0")


def find_reference_executable_root() -> Path | None:
    return _discover("REFERENCE_EXECUTABLE_ROOT", "tests/REFERENCE_EXECUTABLE", "REFERENCE_EXECUTABLE")


def find_reference_executable_executable() -> Path | None:
    root = find_reference_executable_root()
    if root is None:
        return None
    executable = root / "bin" / "main_reference_executable.x"
    return executable if executable.exists() else None


def find_neopax_root() -> Path | None:
    return _discover("NEOPAX_ROOT", "tests/NEOPAX", "NEOPAX")


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
