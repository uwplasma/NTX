#!/usr/bin/env python3
"""What an adjoint buys on a design sweep, measured rather than asserted.

A monoenergetic solver that is only a forward map gives derivatives one way:
re-solve per parameter. That is the cost model behind every finite-difference
sensitivity study, and it is why neoclassical optimization is usually done with
a handful of shape parameters rather than a realistic boundary.

This measures the two cost curves on the same solve, same machine, same
tolerance:

  finite differences  central differences, ``2P`` extra solves for ``P``
                      parameters, and an accuracy that depends on the step
  reverse mode        one adjoint solve for the whole gradient, exact to
                      rounding

and reports the crossover. It also records the accuracy each delivers against
an independent forward-mode derivative, because a cheap gradient that is wrong
is not a saving.

    python benchmarks/bench_design_derivatives.py --params 1,2,4,8,16,32
"""

from __future__ import annotations

import argparse
import json
import platform
import time

import jax
import jax.numpy as jnp
import numpy as np

import ntx  # noqa: E402
from ntx import GridSpec, MonoenergeticCase, example_surface  # noqa: E402


def median_seconds(fn, *args, reps: int = 7) -> float:
    # Two warm-ups, not one: the first pays compilation, the second settles the
    # caches. Timing from the first call made the adjoint look like it got
    # *faster* with more parameters, which is a measurement artefact and would
    # have been the wrong thing to publish.
    jax.block_until_ready(fn(*args))
    jax.block_until_ready(fn(*args))
    samples = []
    for _ in range(reps):
        start = time.perf_counter()
        jax.block_until_ready(fn(*args))
        samples.append(time.perf_counter() - start)
    return float(np.median(samples))


def design_surface(n_modes: int):
    """A Boozer surface whose spectrum has ``n_modes`` free coefficients.

    The built-in example carries four modes, which is too few to show a cost
    curve. This keeps its physics -- same field strength, same rotational
    transform -- and extends the spectrum with the higher harmonics a boundary
    optimizer would actually vary, decaying so the surface stays sensible.
    """
    import numpy as _np

    base = example_surface()
    m, n, amplitude = [0], [0], [1.0]
    poloidal, toroidal = 1, 0
    while len(m) < n_modes:
        m.append(poloidal)
        n.append(toroidal)
        amplitude.append(0.06 * 0.7 ** (len(m) - 2))
        toroidal += 1
        if toroidal > poloidal:
            poloidal += 1
            toroidal = -poloidal
    return type(base)(
        **{
            **base.__dict__,
            "m": jnp.asarray(_np.array(m[:n_modes]), dtype=jnp.int32),
            "n": jnp.asarray(_np.array(n[:n_modes]), dtype=jnp.int32),
            "b_cos": jnp.asarray(_np.array(amplitude[:n_modes])),
        }
    )


def _perturbed_surface(surface, deltas):
    """Perturb the Boozer spectrum by ``deltas``, one entry per retained mode."""
    b_cos = surface.b_cos.at[: deltas.size].add(deltas)
    return type(surface)(**{**surface.__dict__, "b_cos": b_cos})


def build(n_theta: int, n_zeta: int, n_xi: int, nu_hat: float, n_modes: int):
    surface = design_surface(n_modes)
    grid = GridSpec(n_theta, n_zeta, n_xi)
    case = MonoenergeticCase(nu_hat=nu_hat, epsi_hat=0.0)

    def d11(deltas):
        perturbed = _perturbed_surface(surface, deltas)
        return ntx.solve_monoenergetic(perturbed, grid, case).D11

    return surface, grid, case, d11


def run(n_theta: int, n_zeta: int, n_xi: int, nu_hat: float, params: list[int]) -> dict:
    n_modes = max(params)
    surface, grid, case, d11 = build(n_theta, n_zeta, n_xi, nu_hat, n_modes)

    rows = []
    for count in params:
        if count > n_modes:
            continue
        zeros = jnp.zeros((count,))

        value_and_grad = jax.jit(jax.value_and_grad(d11))
        reverse_seconds = median_seconds(value_and_grad, zeros)
        _, reverse_gradient = value_and_grad(zeros)

        # Central differences at the step that minimizes the usual
        # truncation/round-off tradeoff for a smooth double-precision function.
        step = 1.0e-6
        solve = jax.jit(d11)

        def finite_difference(z, solve=solve, count=count, step=step):
            out = []
            for i in range(count):
                e = jnp.zeros((count,)).at[i].set(step)
                out.append((solve(z + e) - solve(z - e)) / (2.0 * step))
            return jnp.stack(out)

        fd_seconds = median_seconds(finite_difference, zeros, reps=3)
        fd_gradient = finite_difference(zeros)

        # Forward mode is an independent exact path, so it arbitrates.
        forward_gradient = jax.jacfwd(d11)(zeros)
        denominator = float(jnp.linalg.norm(forward_gradient))
        rows.append(
            {
                "parameters": int(count),
                "reverse_seconds": reverse_seconds,
                "finite_difference_seconds": fd_seconds,
                "speedup": fd_seconds / reverse_seconds,
                "reverse_relative_error": float(
                    jnp.linalg.norm(reverse_gradient - forward_gradient) / denominator
                ),
                "finite_difference_relative_error": float(
                    jnp.linalg.norm(fd_gradient - forward_gradient) / denominator
                ),
            }
        )
        print(
            f"  P={count:3d}  reverse {reverse_seconds*1e3:8.1f} ms   "
            f"FD {fd_seconds*1e3:9.1f} ms   speedup {rows[-1]['speedup']:6.1f}x   "
            f"rev err {rows[-1]['reverse_relative_error']:.1e}   "
            f"FD err {rows[-1]['finite_difference_relative_error']:.1e}"
        )

    return {
        "what": "cost and accuracy of a design gradient, reverse mode vs finite differences",
        "grid": {"n_theta": n_theta, "n_zeta": n_zeta, "n_xi": n_xi},
        "boozer_modes": int(n_modes),
        "nu_hat": nu_hat,
        "rows": rows,
        "provenance": {
            "ntx": ntx.__version__,
            "jax": jax.__version__,
            "backend": jax.default_backend(),
            "device_kind": jax.devices()[0].device_kind,
            "platform": platform.platform(),
            "x64": bool(jax.config.jax_enable_x64),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-theta", type=int, default=13)
    parser.add_argument("--n-zeta", type=int, default=13)
    parser.add_argument("--n-xi", type=int, default=40)
    parser.add_argument("--nu-hat", type=float, default=1e-2)
    parser.add_argument("--params", default="1,2,4")
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()

    counts = [int(v) for v in args.params.split(",")]
    record = run(args.n_theta, args.n_zeta, args.n_xi, args.nu_hat, counts)
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as handle:
            json.dump(record, handle, indent=2)
        print(f"wrote {args.output_json}")


if __name__ == "__main__":
    main()
