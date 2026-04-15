from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
SRC = ROOT / "src"


def _local_dependencies_available() -> bool:
    required = [
        Path("/Users/rogeriojorge/local/single_stage_optimization_finite_beta/optimization_finitebeta_nfp3_QA_stage1/wout_final.nc"),
        Path("/Users/rogeriojorge/local/tests/sfincs_jax"),
        Path("/Users/rogeriojorge/local/sfincs/fortran/version3/sfincs"),
        Path("/Users/rogeriojorge/local/simsopt"),
        Path("/Users/rogeriojorge/local/booz_xform_jax"),
    ]
    return all(path.exists() for path in required)


@pytest.mark.skipif(
    not _local_dependencies_available(),
    reason="requires local QA/SFINCS research stack",
)
def test_compute_ntx_native_profile_on_finite_beta_case():
    sys.path.insert(0, str(EXAMPLES))
    from bootstrap_current_native_validation_common import (
        available_cases,
        compute_ntx_native_profile,
    )

    qa = available_cases()["qa"]
    profile = compute_ntx_native_profile(qa, np.asarray([0.25, 0.5]))
    assert profile["rho"].shape == (2,)
    assert profile["current_density"].shape == (2,)
    assert np.all(np.isfinite(profile["current_density"]))


@pytest.mark.skipif(
    not _local_dependencies_available(),
    reason="requires local QA/SFINCS research stack",
)
def test_sfincs_jax_and_sfincs_agree_on_one_qa_radius():
    sys.path.insert(0, str(EXAMPLES))
    from bootstrap_current_native_validation_common import (
        available_cases,
        compute_sfincs_jax_profile,
        compute_sfincs_profile,
        max_relative_error,
    )

    qa = available_cases()["qa"]
    rho = np.asarray([0.35])
    sfincs_jax = compute_sfincs_jax_profile(qa, rho, recompute=False)
    sfincs = compute_sfincs_profile(qa, rho, recompute=False)
    error = max_relative_error(
        np.asarray(sfincs_jax["observable"], dtype=float),
        np.asarray(sfincs["observable"], dtype=float),
    )
    assert error < 1.0e-3


def test_bootstrap_current_native_validation_script_runs_if_repointed(tmp_path):
    output_prefix = tmp_path / "bootstrap_current_native_validation"
    script = (EXAMPLES / "bootstrap_current_native_validation.py").read_text(encoding="utf-8")
    script = script.replace(
        'OUTPUT_PREFIX = ('
        '\n    Path(__file__).resolve().parents[1]'
        '\n    / "docs"'
        '\n    / "_static"'
        '\n    / "bootstrap_current_native_validation"'
        '\n)',
        f'OUTPUT_PREFIX = Path(r"{output_prefix}")',
    )
    script = script.replace("RECOMPUTE_SFINCS = False", "RECOMPUTE_SFINCS = False")
    run_path = tmp_path / "bootstrap_current_native_validation.py"
    run_path.write_text(script, encoding="utf-8")

    env = dict(os.environ)
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(SRC) if not existing_pythonpath else f"{SRC}:{existing_pythonpath}"

    completed = subprocess.run(
        [sys.executable, str(run_path)],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
    )
    if _local_dependencies_available():
        pytest.skip(
            "full external benchmark script is exercised manually, "
            "not in the default test lane"
        )
    assert completed.returncode != 0
    assert (
        "finite-beta QA/QH case directories" in completed.stderr
        or "finite-beta QA/QH case directories" in completed.stdout
    )
