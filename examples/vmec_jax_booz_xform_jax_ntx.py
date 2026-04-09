#!/usr/bin/env python3
"""Run a small `vmec_jax -> booz_xform_jax -> NTX` solve."""

from __future__ import annotations

import argparse

import jax.numpy as jnp


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="/Users/rogeriojorge/local/vmec_jax/examples/data/input.circular_tokamak",
        help="VMEC input file to solve with vmec_jax.",
    )
    parser.add_argument("--s", type=float, default=0.25, help="Normalized toroidal flux surface.")
    parser.add_argument("--mboz", type=int, default=8)
    parser.add_argument("--nboz", type=int, default=8)
    args = parser.parse_args()

    import vmec_jax as vj

    from ntx import GridSpec, MonoenergeticCase, solve_monoenergetic, surface_from_vmec_jax_state

    run = vj.run_fixed_boundary(
        args.input,
        max_iter=1,
        use_initial_guess=True,
        vmec_project=False,
        verbose=True,
    )
    geom = vj.eval_geom(run.state, run.static)
    signgs = vj.signgs_from_sqrtg(geom.sqrtg, axis_index=1)
    surface = surface_from_vmec_jax_state(
        state=run.state,
        static=run.static,
        indata=run.indata,
        signgs=int(signgs),
        s=args.s,
        mboz=args.mboz,
        nboz=args.nboz,
    )
    result = solve_monoenergetic(
        surface,
        GridSpec(n_theta=17, n_zeta=17, n_xi=40, dtype=jnp.float32),
        MonoenergeticCase(nu_hat=1.0e-4, epsi_hat=0.0),
    )
    print("surface:", args.s)
    print("D11:", float(result.D11))
    print("D13:", float(result.D13))
    print("D33:", float(result.D33))


if __name__ == "__main__":
    main()
