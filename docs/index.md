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

for modes needed to evaluate the geometric monoenergetic coefficients
`D11`, `D31`, `D13`, and `D33`.

## Example

```python
from ntx import GridSpec, MonoenergeticCase, example_surface, solve_monoenergetic

surface = example_surface()
grid = GridSpec(n_theta=5, n_zeta=5, n_xi=6)
case = MonoenergeticCase(nu_hat=1e-3, er_hat=0.0)
result = solve_monoenergetic(surface, grid, case)
print(result.as_dict())
```

## Benchmarks

Reference comparisons should be generated from external benchmark runs and kept
outside the NTX implementation. Large benchmark artifacts should not be committed
unless deliberately tracked as fixtures.
