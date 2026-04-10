from __future__ import annotations

import json
import os
import subprocess
import sys

import jax.numpy as jnp
import pytest

from ntx import (
    build_reference_vmec_scan,
    enable_x64,
    load_neopax_reference_scan,
    to_neopax_monoenergetic,
    vmec_reference_factors,
    write_neopax_scan_hdf5,
)
from ntx._checkout_paths import find_neopax_root, find_reference_executable, repo_root

REFERENCE_EXE = find_reference_executable()
NEOPAX_ROOT = find_neopax_root()
ROOT = repo_root()
WOUT = (
    None
    if NEOPAX_ROOT is None
    else NEOPAX_ROOT / "tests" / "inputs" / "wout_W7-X_standard_configuration.nc"
)
BOOZ = (
    None
    if NEOPAX_ROOT is None
    else NEOPAX_ROOT / "tests" / "inputs" / "boozmn_wout_W7-X_standard_configuration.nc"
)
REFERENCE = (
    None
    if NEOPAX_ROOT is None
    else NEOPAX_ROOT / "tests" / "inputs" / "Dij_NEOPAX_FULL_S_NEW_W7X.h5"
)
EXAMPLE = (
    ROOT
    / "examples"
    / "DKES_like_database"
    / "Test_Monoenergetic_database_VMEC_s_coordinate_W7X.py"
)
COMPARE_SCRIPT = ROOT / "scripts" / "compare_reference_executable.py"

if (
    NEOPAX_ROOT is None
    or WOUT is None
    or BOOZ is None
    or REFERENCE is None
    or not WOUT.exists()
    or not BOOZ.exists()
):
    pytest.skip("local NEOPAX VMEC reference inputs not available", allow_module_level=True)

enable_x64(True)


def _index_matches(values: jnp.ndarray, targets: jnp.ndarray) -> list[int]:
    return [int(jnp.where(jnp.isclose(values, target))[0][0]) for target in targets]


@pytest.mark.benchmark
@pytest.mark.skipif(
    REFERENCE_EXE is None or not REFERENCE_EXE.exists(),
    reason="local benchmark executable not available",
)
def test_reference_executable_matches_vmec_example_point(tmp_path):
    input_path = tmp_path / "reference-vmec.toml"
    input_path.write_text(
        "\n".join(
            [
                "[surface]",
                'type = "vmec"',
                f'path = "{WOUT}"',
                "psi_n = 0.25",
                "",
                "[grid]",
                "n_theta = 9",
                "n_zeta = 11",
                "n_xi = 8",
                "",
                "[case]",
                "nu_hat = 1e-3",
                "er_hat = 1e-3",
                "",
                "[output]",
                'npz = "results.npz"',
            ]
        ),
        encoding="utf-8",
    )
    proc = subprocess.run(
        [
            sys.executable,
            str(COMPARE_SCRIPT),
            str(input_path),
            "--reference-exe",
            str(REFERENCE_EXE),
        ],
        check=True,
        text=True,
        capture_output=True,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
    )
    payload = json.loads(proc.stdout)
    for key, tolerance in {"D11": 1.0e-12, "D13": 1.0e-11, "D31": 1.0e-11, "D33": 1.0e-9}.items():
        assert abs(payload["ntx_minus_reference"][key]) < tolerance


@pytest.mark.benchmark
def test_reference_vmec_scan_matches_neopax_reference_subset():
    rho = jnp.asarray([0.25, 0.5])
    nu_v = jnp.asarray([1.0e-4, 1.0e-3, 1.0e-2])
    er_tilde = jnp.asarray([0.0, 1.0e-3, 1.0e-2])
    reference = load_neopax_reference_scan(REFERENCE)
    scan = build_reference_vmec_scan(
        WOUT,
        BOOZ,
        rho=rho,
        nu_v=nu_v,
        er_tilde=er_tilde,
        nt=25,
        nz=25,
        nl=64,
        source_name="ntx_reference_subset",
    )

    rho_idx = jnp.asarray(_index_matches(reference.rho, rho))
    nu_idx = jnp.asarray(_index_matches(reference.nu_v, nu_v))
    er_idx = jnp.asarray(_index_matches(reference.Er_tilde, er_tilde))

    ref_d11 = reference.D11[rho_idx][:, nu_idx][:, :, er_idx]
    ref_d13 = reference.D13[rho_idx][:, nu_idx][:, :, er_idx]
    ref_d33 = reference.D33[rho_idx][:, nu_idx][:, :, er_idx]
    ref_d31 = reference.D31[rho_idx][:, nu_idx][:, :, er_idx]

    for actual, expected in (
        (scan.D11, ref_d11),
        (scan.D13, ref_d13),
        (scan.D33, ref_d33),
        (scan.D31, ref_d31),
    ):
        relative = jnp.abs((actual - expected) / jnp.maximum(jnp.abs(expected), 1.0e-12))
        assert jnp.max(relative) < 1.0e-2

    ntx_db = to_neopax_monoenergetic(scan, a_b=float(scan.a_b))
    reference_db = to_neopax_monoenergetic(
        type(scan)(
            rho=rho,
            nu_v=nu_v,
            Er=reference.Er[rho_idx][:, er_idx],
            Es=reference.Es[rho_idx][:, er_idx],
            drds=reference.drds[rho_idx],
            D11=ref_d11,
            D13=ref_d13,
            D33=ref_d33,
            D31=ref_d31,
            Er_tilde=er_tilde,
            a_b=float(scan.a_b),
        ),
        a_b=float(scan.a_b),
    )
    assert jnp.max(jnp.abs(ntx_db.D11_log - reference_db.D11_log)) < 1.0e-2
    assert jnp.max(jnp.abs(ntx_db.D13 - reference_db.D13)) < 1.0e-2
    assert (
        jnp.max(
            jnp.abs(ntx_db.D33 - reference_db.D33)
            / jnp.maximum(jnp.abs(reference_db.D33), 1.0e-12)
        )
        < 1.0e-2
    )


def test_vmec_reference_factors_are_finite():
    rho = jnp.asarray([0.25, 0.5])
    factors = vmec_reference_factors(WOUT, BOOZ, rho)
    assert jnp.all(jnp.isfinite(factors.b00))
    assert jnp.all(jnp.isfinite(factors.drds))
    assert jnp.all(jnp.isfinite(factors.fac_reference_to_sfincs_11))
    assert jnp.all(jnp.isfinite(factors.fac_sfincs_to_dkes_11))


@pytest.mark.benchmark
def test_ntx_reference_example_writes_hdf5_and_matches_subset(tmp_path):
    output = tmp_path / "ntx_reference_subset.h5"
    command = [
        sys.executable,
        str(EXAMPLE),
        "--vmec",
        str(WOUT),
        "--booz",
        str(BOOZ),
        "--reference-h5",
        str(REFERENCE),
        "--output",
        str(output),
        "--rho",
        "0.25,0.5",
        "--nu-v",
        "1e-4,1e-3",
        "--er-tilde",
        "0.0,1e-3",
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    assert output.exists()
    assert "max relative errors:" in result.stdout

    generated = load_neopax_reference_scan(output)
    reference = load_neopax_reference_scan(REFERENCE)
    rho_idx = jnp.asarray([1, 3])
    nu_idx = jnp.asarray([5, 7])
    er_idx = jnp.asarray([0, 7])
    assert jnp.max(
        jnp.abs(
            generated.D11 - reference.D11[rho_idx][:, nu_idx][:, :, er_idx]
        )
        / jnp.maximum(jnp.abs(reference.D11[rho_idx][:, nu_idx][:, :, er_idx]), 1.0e-12)
    ) < 1.0e-2

    rewritten = tmp_path / "roundtrip.h5"
    write_neopax_scan_hdf5(generated, rewritten)
    assert rewritten.exists()
