from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp

from ntx import GridSpec, load_dkes_surface, load_run_config, load_vmec_surface, solve_monoenergetic
from ntx.geometry import example_surface
from ntx.solver import MonoenergeticCase


def _load_surface_from_config(config):
    if config.surface.type == "example":
        return example_surface()
    if config.surface.type == "dkes":
        assert config.surface.path is not None
        return load_dkes_surface(config.surface.path)
    assert config.surface.path is not None
    assert config.surface.psi_n is not None
    return load_vmec_surface(
        config.surface.path,
        psi_n=config.surface.psi_n,
        vmec_radial_option=config.surface.vmec_radial_option,
        vmec_nyquist_option=config.surface.vmec_nyquist_option,
        vmec_mode_convention=config.surface.vmec_mode_convention,
        min_bmn_to_load=config.surface.min_bmn_to_load,
    )


def test_example_tomls_parse_and_solve():
    example_dir = Path(__file__).resolve().parents[1] / "examples"
    for path in sorted(example_dir.glob("*.toml")):
        config = load_run_config(path)
        surface = _load_surface_from_config(config)
        grid = GridSpec(config.grid.n_theta, config.grid.n_zeta, min(config.grid.n_xi, 4))
        case = MonoenergeticCase(
            nu_hat=config.case.nu_hat,
            epsi_hat=config.case.epsi_hat,
            er_hat=config.case.er_hat,
        )
        result = solve_monoenergetic(surface, grid, case)
        values = jnp.asarray([result.D11, result.D31, result.D13, result.D33, result.D33_spitzer])
        assert jnp.all(jnp.isfinite(values)), path.name
