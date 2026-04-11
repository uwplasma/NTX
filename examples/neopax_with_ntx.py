#!/usr/bin/env python3
"""Build a small NTX database and map it into NEOPAX-style arrays."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import jax.numpy as jnp  # noqa: E402

from ntx import (  # noqa: E402
    GridSpec,
    build_ntx_neopax_scan,
    load_neopax_reference_scan,
    surface_from_vmec_jax_vmec_wout_file,
    to_neopax_monoenergetic,
)


def main() -> None:
    wout = ROOT / "tests" / "fixtures" / "sample_wout.nc"
    reference_path = ROOT / "tests" / "fixtures" / "sample_neopax_scan.h5"
    reference = load_neopax_reference_scan(reference_path)
    rho = reference.rho
    nu_v = reference.nu_v
    Er = reference.Er
    Es = reference.Es
    drds = reference.drds

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
        grid=GridSpec(n_theta=9, n_zeta=9, n_xi=8),
        source_name="sample_vmec_subset",
    )
    database = to_neopax_monoenergetic(scan, a_b=1.0)

    print("rho:", jnp.asarray(database.rho))
    print("nu_log shape:", database.nu_log.shape)
    print("D11_log shape:", database.D11_log.shape)
    print("D33 shape:", database.D33.shape)
    print("source:", reference_path.name)


if __name__ == "__main__":
    main()
