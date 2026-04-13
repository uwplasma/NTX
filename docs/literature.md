# Literature And External Packages

## Core Physics Reference

- Javier Escoto, *Fast monoenergetic neoclassical transport coefficients in
  stellarators*, PhD thesis, 2025:
  [arXiv:2510.27513](https://arxiv.org/abs/2510.27513)

This is the primary reference for:

- the monoenergetic formulation
- the Legendre-space block-tridiagonal solve
- Onsager symmetry in the monoenergetic setting
- the derivative and optimization discussion

## JAX And Python Geometry Packages

- [JAX](https://github.com/jax-ml/jax)
- [vmec_jax](https://github.com/uwplasma/vmec_jax)
- [booz_xform_jax](https://github.com/uwplasma/booz_xform_jax)
- [NEOPAX](https://github.com/uwplasma/NEOPAX)

## Independent Validation Ecosystem

NTX users often want to compare against other neoclassical tools or pipelines.
The repository documentation refers to:

- [SFINCS-JAX](https://github.com/uwplasma/sfincs_jax) when discussing
  independent consistency checks

These packages are useful for trust-building and application workflows, but
NTX's equations, numerics, and public interface are defined by its own source
tree and the Escoto thesis.
