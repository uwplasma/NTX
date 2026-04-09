#!/usr/bin/env python3
"""Build a small NTX database and hand it directly to NEOPAX."""

from __future__ import annotations

import sys
from pathlib import Path

import jax.numpy as jnp

NEOPAX_ROOT = Path("/Users/rogeriojorge/local/tests/NEOPAX")
if str(NEOPAX_ROOT) not in sys.path:
    sys.path.insert(0, str(NEOPAX_ROOT))

from ntx import (  # noqa: E402
    GridSpec,
    build_ntx_neopax_scan,
    load_neopax_reference_scan,
    surface_from_vmec_jax_wout,
    to_neopax_monoenergetic,
)


def main() -> None:
    wout = NEOPAX_ROOT / "tests" / "inputs" / "wout_W7-X_standard_configuration.nc"
    reference_path = NEOPAX_ROOT / "tests" / "inputs" / "Dij_NEOPAX_FULL_S_NEW_W7X.h5"
    input_path = Path(
        "/Users/rogeriojorge/local/tests/simsopt/tests/test_files/"
        "input.W7-X_standard_configuration"
    )
    reference = load_neopax_reference_scan(reference_path)

    rho = reference.rho[:2]
    nu_v = reference.nu_v[2:5]
    Er = reference.Er[:2, :3]
    Es = reference.Es[:2, :3]
    drds = reference.drds[:2]

    def surface_loader(rho_value: float):
        return surface_from_vmec_jax_wout(
            input_path=input_path,
            wout_path=wout,
            s=float(rho_value**2),
            mboz=12,
            nboz=12,
        )

    scan = build_ntx_neopax_scan(
        surface_loader,
        rho=rho,
        nu_v=nu_v,
        Es=Es,
        Er=Er,
        drds=drds,
        grid=GridSpec(n_theta=17, n_zeta=33, n_xi=60),
        source_name="w7x_subset",
    )
    database = to_neopax_monoenergetic(scan, a_b=1.0)

    print("rho:", jnp.asarray(database.rho))
    print("nu_log shape:", database.nu_log.shape)
    print("D11_log shape:", database.D11_log.shape)
    print("D33 shape:", database.D33.shape)


if __name__ == "__main__":
    main()
