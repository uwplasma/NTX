from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_plot_output_file_example(tmp_path):
    output_path = tmp_path / "sample_output.nc"
    input_path = tmp_path / "sample_input.toml"
    sample_surface = ROOT / "tests" / "fixtures" / "sample_surface.ddkes2.data"
    env = dict(os.environ)
    existing_pythonpath = env.get("PYTHONPATH")
    src_path = str(ROOT / "src")
    env["PYTHONPATH"] = (
        src_path if not existing_pythonpath else f"{src_path}:{existing_pythonpath}"
    )
    input_path.write_text(
        "\n".join(
            [
                "[surface]",
                'type = "dkes"',
                f'path = "{sample_surface.as_posix()}"',
                "",
                "[grid]",
                "n_theta = 7",
                "n_zeta = 9",
                "n_xi = 6",
                "",
                "[case]",
                "nu_hat = 1e-3",
                "er_hat = 1e-3",
                "",
                "[output]",
                f'path = "{output_path.as_posix()}"',
                "include_modes = true",
                "",
                "[logging]",
                "verbose = false",
                "",
            ]
        ),
        encoding="utf-8",
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "ntx",
            str(input_path),
        ],
        check=True,
        text=True,
        capture_output=True,
        cwd=ROOT,
        env=env,
    )
    output_prefix = tmp_path / "output_summary"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "examples" / "plot_output_file.py"),
            str(output_path),
            "--output-prefix",
            str(output_prefix),
        ],
        check=True,
        text=True,
        capture_output=True,
        cwd=ROOT,
        env=env,
    )
    assert output_prefix.with_suffix(".png").exists()
    assert output_prefix.with_suffix(".pdf").exists()
