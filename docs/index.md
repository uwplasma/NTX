# NTX

NTX is a JAX-native solver for monoenergetic neoclassical transport on
stellarator flux surfaces.

The first solver implements the Legendre-space block-tridiagonal drift-kinetic
equation formulation from Francisco Javier Escoto López's PhD thesis,
[`arXiv:2510.27513`](https://arxiv.org/abs/2510.27513).

## Equation

For each monoenergetic case, NTX solves the truncated Legendre system

```text
L_k f^(k-1) + D_k f^(k) + U_k f^(k+1) = s^(k)
```

and evaluates the monoenergetic geometric coefficients `D11`, `D31`, `D13`,
`D33`, and the Spitzer `D33` normalization.

## Primary Endpoint

The main user entrypoint is:

```bash
ntx input.toml
```

It prints a Rich terminal summary and writes a compressed `.npz` with the
resolved inputs, geometry arrays, transport coefficients, residuals, and
optional low-order Legendre modes.

```{toctree}
:maxdepth: 2

input-file
```
