#!/usr/bin/env python3
"""Build a small QI VMEC NTX scan and map it into NEOPAX-style arrays."""

from __future__ import annotations

import argparse
from pathlib import Path

import jax.numpy as jnp

from ntx import (
    GridSpec,
    build_ntx_neopax_scan_from_surfaces,
    load_vmec_surface,
    scan_to_neopax_arrays,
    write_neopax_scan_hdf5,
)

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--wout",
        default=str(
            ROOT / "tests" / "fixtures" / "wout_QI_nfp2_stable_Er_006_000043_hires_scaled.nc"
        ),
        help="QI VMEC wout file to use.",
    )
    parser.add_argument(
        "--output",
        default="qi_ntx_scan.h5",
        help="Output HDF5 path for the generated NEOPAX-style scan.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    wout = Path(args.wout).expanduser().resolve()
    rho = jnp.asarray([0.12247, 0.25])
    surfaces = tuple(load_vmec_surface(wout, psi_n=float(rho_value**2)) for rho_value in rho)
    nu_v = jnp.asarray([1.0e-4, 1.0e-3])
    es = jnp.asarray([[0.0, 5.0e-4], [0.0, 5.0e-4]])
    er = jnp.asarray([[0.0, 5.0e-4], [0.0, 5.0e-4]])
    drds = jnp.asarray([1.0, 1.0])

    scan = build_ntx_neopax_scan_from_surfaces(
        surfaces,
        rho=rho,
        nu_v=nu_v,
        Es=es,
        Er=er,
        drds=drds,
        grid=GridSpec(9, 11, 16),
        source_name="qi_ntx_scan",
    )
    mapped = scan_to_neopax_arrays(scan, a_b=1.0)
    output = write_neopax_scan_hdf5(scan, args.output)

    print("rho:", jnp.asarray(scan.rho))
    print("nu_v:", jnp.asarray(scan.nu_v))
    print("D11 shape:", scan.D11.shape)
    print("D13 shape:", scan.D13.shape)
    print("D33 shape:", scan.D33.shape)
    print("mapped Er_list shape:", mapped.Er_list.shape)
    print("output:", output)


if __name__ == "__main__":
    main()
