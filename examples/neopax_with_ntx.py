#!/usr/bin/env python3
"""Build a small NTX database and hand it directly to NEOPAX."""

from __future__ import annotations

import sys
from pathlib import Path

import jax.numpy as jnp

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ntx._checkout_paths import find_neopax_root  # noqa: E402

NEOPAX_ROOT = find_neopax_root()
if NEOPAX_ROOT is None:
    raise SystemExit("NEOPAX checkout not found. Set NEOPAX_ROOT or place it next to NTX.")
if str(NEOPAX_ROOT) not in sys.path:
    sys.path.insert(0, str(NEOPAX_ROOT))

from ntx import (  # noqa: E402
    GridSpec,
    build_ntx_neopax_scan,
    load_neopax_reference_scan,
    surface_from_vmec_jax_vmec_wout_file,
    to_neopax_monoenergetic,
)


def main() -> None:
    wout = NEOPAX_ROOT / "tests" / "inputs" / "wout_W7-X_standard_configuration.nc"
    reference_path = NEOPAX_ROOT / "tests" / "inputs" / "Dij_NEOPAX_FULL_S_NEW_W7X.h5"
    reference = load_neopax_reference_scan(reference_path)

    rho_idx = jnp.asarray([1, 3])
    nu_idx = jnp.asarray([5, 7, 9])
    er_idx = jnp.asarray([0, 7, 9])
    rho = reference.rho[rho_idx]
    nu_v = reference.nu_v[nu_idx]
    Er = reference.Er[rho_idx][:, er_idx]
    Es = reference.Es[rho_idx][:, er_idx]
    drds = reference.drds[rho_idx]

    def surface_loader(rho_value: float):
        return surface_from_vmec_jax_vmec_wout_file(
            wout,
            s=float(rho_value**2),
        )

    scan = build_ntx_neopax_scan(
        surface_loader,
        rho=rho,
        nu_v=nu_v,
        Es=Es,
        Er=Er,
        drds=drds,
        grid=GridSpec(n_theta=25, n_zeta=25, n_xi=63),
        source_name="w7x_vmec_jax_vmec_subset",
    )
    database = to_neopax_monoenergetic(scan, a_b=1.0)
    reference_database = to_neopax_monoenergetic(
        type(scan)(
            rho=rho,
            nu_v=nu_v,
            Er=Er,
            Es=Es,
            drds=drds,
            D11=reference.D11[rho_idx][:, nu_idx][:, :, er_idx],
            D13=reference.D13[rho_idx][:, nu_idx][:, :, er_idx],
            D33=reference.D33[rho_idx][:, nu_idx][:, :, er_idx],
            D31=reference.D31[rho_idx][:, nu_idx][:, :, er_idx],
            Er_tilde=reference.Er_tilde[er_idx],
            a_b=1.0,
        ),
        a_b=1.0,
    )

    print("rho:", jnp.asarray(database.rho))
    print("nu_log shape:", database.nu_log.shape)
    print("D11_log shape:", database.D11_log.shape)
    print("D33 shape:", database.D33.shape)
    print(
        "max |D11_log - ref|:",
        float(jnp.max(jnp.abs(database.D11_log - reference_database.D11_log))),
    )
    print("max |D13 - ref|:", float(jnp.max(jnp.abs(database.D13 - reference_database.D13))))
    print(
        "max rel D33 diff:",
        float(
            jnp.max(
                jnp.abs(
                    (database.D33 - reference_database.D33)
                    / jnp.maximum(jnp.abs(reference_database.D33), 1.0e-12)
                )
            )
        ),
    )


if __name__ == "__main__":
    main()
