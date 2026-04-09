"""Optional VMEC geometry comparisons against a local sfincs_jax checkout."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

from ._checkout_paths import find_sfincs_jax_root
from .geometry import geometry_on_grid
from .grids import GridSpec
from .vmec import load_vmec_surface


def compare_vmec_geometry_to_sfincs(
    *,
    wout_path: str | Path,
    psi_n: float,
    grid: GridSpec,
    vmec_radial_option: int = 0,
    vmec_nyquist_option: int = 1,
    min_bmn_to_load: float = 0.0,
    sfincs_repo: str | Path | None = None,
) -> dict[str, Any]:
    """Compare NTX VMEC geometry against sfincs_jax on the same angular grid.

    Notes
    -----
    The comparison uses NTX's ``filtered_nyquist`` VMEC convention because it
    matches the mode-selection logic in ``sfincs_jax``. ``sfincs_jax`` stores
    the toroidal-angle dependence with the opposite sign convention, so its
    arrays are compared after reversing the sampled zeta coordinate. Its
    Jacobian-like quantity also carries the opposite sign, so the comparison
    uses ``-psi_a_hat / d_hat``.
    """

    repo = (
        Path(sfincs_repo).expanduser().resolve()
        if sfincs_repo is not None
        else find_sfincs_jax_root()
    )
    if repo is None:
        raise FileNotFoundError("sfincs_jax checkout not found")
    if not repo.exists():
        raise FileNotFoundError(str(repo))
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from sfincs_jax.vmec_geometry import vmec_geometry_from_wout_file

    surface = load_vmec_surface(
        wout_path,
        psi_n=psi_n,
        vmec_radial_option=vmec_radial_option,
        vmec_nyquist_option=vmec_nyquist_option,
        vmec_mode_convention="filtered_nyquist",
        min_bmn_to_load=min_bmn_to_load,
    )
    geom = geometry_on_grid(surface, grid)
    sfincs_geom = vmec_geometry_from_wout_file(
        path=wout_path,
        theta=geom.grid.theta,
        zeta=geom.grid.zeta,
        psi_n_wish=psi_n,
        vmec_radial_option=vmec_radial_option,
        vmec_nyquist_option=vmec_nyquist_option,
        min_bmn_to_load=min_bmn_to_load,
    )

    def reverse_zeta(values: np.ndarray) -> np.ndarray:
        return values[:, [0] + list(range(values.shape[1] - 1, 0, -1))]

    def metrics(lhs: np.ndarray, rhs: np.ndarray) -> dict[str, float]:
        diff = np.abs(lhs - rhs)
        rel = diff / np.maximum(1e-14, np.abs(rhs))
        return {
            "max_abs": float(np.max(diff)),
            "max_rel": float(np.max(rel)),
        }

    comparisons = {
        "b": metrics(np.asarray(geom.b), reverse_zeta(np.asarray(sfincs_geom.b_hat))),
        "d_b_dtheta": metrics(
            np.asarray(geom.d_b_dtheta),
            reverse_zeta(np.asarray(sfincs_geom.db_hat_dtheta)),
        ),
        "d_b_dzeta": metrics(
            np.asarray(geom.d_b_dzeta),
            -reverse_zeta(np.asarray(sfincs_geom.db_hat_dzeta)),
        ),
        "b_sub_theta": metrics(
            np.asarray(geom.b_sub_theta),
            reverse_zeta(np.asarray(sfincs_geom.b_hat_sub_theta)),
        ),
        "b_sub_zeta": metrics(
            np.asarray(geom.b_sub_zeta),
            reverse_zeta(np.asarray(sfincs_geom.b_hat_sub_zeta)),
        ),
        "b_sup_theta": metrics(
            np.asarray(geom.b_sup_theta),
            reverse_zeta(np.asarray(sfincs_geom.b_hat_sup_theta)),
        ),
        "b_sup_zeta": metrics(
            np.asarray(geom.b_sup_zeta),
            reverse_zeta(np.asarray(sfincs_geom.b_hat_sup_zeta)),
        ),
        "jacobian": metrics(
            np.asarray(geom.jacobian),
            -reverse_zeta(surface.psi_a_hat / np.asarray(sfincs_geom.d_hat)),
        ),
    }
    return {
        "surface_path": str(Path(wout_path).expanduser().resolve()),
        "psi_n": float(surface.psi_n),
        "vmec_radial_option": int(vmec_radial_option),
        "vmec_nyquist_option": int(vmec_nyquist_option),
        "min_bmn_to_load": float(min_bmn_to_load),
        "sfincs_repo": str(repo),
        "comparisons": comparisons,
    }
