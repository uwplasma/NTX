"""NTX replacement for Eduardo's REFERENCE_EXECUTABLE W7-X VMEC monoenergetic database script."""

from __future__ import annotations

import argparse
from pathlib import Path

import jax.numpy as jnp

from ntx import (
    build_reference_executable_reference_vmec_scan,
    enable_x64,
    load_neopax_reference_scan,
    to_neopax_monoenergetic,
    write_neopax_scan_hdf5,
)

DEFAULT_WOUT = Path(
    "/Users/rogeriojorge/local/tests/NEOPAX/tests/inputs/wout_W7-X_standard_configuration.nc"
)
DEFAULT_BOOZ = Path(
    "/Users/rogeriojorge/local/tests/NEOPAX/tests/inputs/boozmn_wout_W7-X_standard_configuration.nc"
)
DEFAULT_REFERENCE = Path(
    "/Users/rogeriojorge/local/tests/NEOPAX/tests/inputs/Dij_NEOPAX_FULL_S_NEW_W7X.h5"
)


def _parse_list(text: str) -> jnp.ndarray:
    values = [float(part.strip()) for part in text.split(",") if part.strip()]
    if not values:
        raise ValueError("expected at least one comma-separated float")
    return jnp.asarray(values, dtype=jnp.float64)


def _index_matches(values: jnp.ndarray, targets: jnp.ndarray) -> list[int]:
    return [int(jnp.where(jnp.isclose(values, target))[0][0]) for target in targets]


def main() -> int:
    enable_x64(True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vmec", type=Path, default=DEFAULT_WOUT)
    parser.add_argument("--booz", type=Path, default=DEFAULT_BOOZ)
    parser.add_argument("--reference-h5", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--output", type=Path, default=Path("Dij_NEOPAX_FULL_S_NEW_W7X_ntx.h5"))
    parser.add_argument("--rho", default="0.12247,0.25")
    parser.add_argument("--nu-v", default="1e-4,1e-3,1e-2")
    parser.add_argument("--er-tilde", default="0.0,1e-3,1e-2")
    parser.add_argument("--nt", type=int, default=25)
    parser.add_argument("--nz", type=int, default=25)
    parser.add_argument("--nl", type=int, default=64)
    args = parser.parse_args()

    rho = _parse_list(args.rho)
    nu_v = _parse_list(args.nu_v)
    er_tilde = _parse_list(args.er_tilde)

    scan = build_reference_executable_reference_vmec_scan(
        args.vmec,
        args.booz,
        rho=rho,
        nu_v=nu_v,
        er_tilde=er_tilde,
        nt=args.nt,
        nz=args.nz,
        nl=args.nl,
        source_name="ntx_vmec_reference",
    )
    output_path = write_neopax_scan_hdf5(scan, args.output)
    print(f"wrote {output_path}")

    if args.reference_h5.exists():
        reference = load_neopax_reference_scan(args.reference_h5)
        rho_idx = _index_matches(reference.rho, rho)
        nu_idx = _index_matches(reference.nu_v, nu_v)
        er_idx = _index_matches(reference.Er_tilde, er_tilde)
        rho_sel = jnp.asarray(rho_idx)
        nu_sel = jnp.asarray(nu_idx)
        er_sel = jnp.asarray(er_idx)

        reference_scan = type(scan)(
            rho=reference.rho[rho_sel],
            nu_v=reference.nu_v[nu_sel],
            Er=reference.Er[rho_sel][:, er_sel],
            Es=reference.Es[rho_sel][:, er_sel],
            drds=reference.drds[rho_sel],
            D11=reference.D11[rho_sel][:, nu_sel][:, :, er_sel],
            D13=reference.D13[rho_sel][:, nu_sel][:, :, er_sel],
            D33=reference.D33[rho_sel][:, nu_sel][:, :, er_sel],
            D31=(
                None
                if reference.D31 is None
                else reference.D31[rho_sel][:, nu_sel][:, :, er_sel]
            ),
            Er_tilde=reference.Er_tilde[er_sel] if reference.Er_tilde is not None else None,
            a_b=scan.a_b,
        )
        ntx_db = to_neopax_monoenergetic(scan, a_b=float(scan.a_b))
        ref_db = to_neopax_monoenergetic(reference_scan, a_b=float(scan.a_b))
        d11_rel = jnp.max(
            jnp.abs(
                (scan.D11 - reference_scan.D11)
                / jnp.maximum(jnp.abs(reference_scan.D11), 1.0e-12)
            )
        )
        d13_rel = jnp.max(
            jnp.abs(
                (scan.D13 - reference_scan.D13)
                / jnp.maximum(jnp.abs(reference_scan.D13), 1.0e-12)
            )
        )
        d33_rel = jnp.max(
            jnp.abs(
                (scan.D33 - reference_scan.D33)
                / jnp.maximum(jnp.abs(reference_scan.D33), 1.0e-12)
            )
        )
        print(
            "max relative errors:"
            f" D11={float(d11_rel):.3e}"
            f" D13={float(d13_rel):.3e}"
            f" D33={float(d33_rel):.3e}"
        )
        db_rel = jnp.max(
            jnp.abs(ref_db.D33 - ntx_db.D33) / jnp.maximum(jnp.abs(ref_db.D33), 1.0e-12)
        )
        print(f"max NEOPAX database D33 relative error: {float(db_rel):.3e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
