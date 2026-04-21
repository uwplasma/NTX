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

## Neoclassical Transport Theory

- Helander and Sigmar, *Collisional Transport in Magnetized Plasmas*:
  [Cambridge University Press](https://www.cambridge.org/core/books/collisional-transport-in-magnetized-plasmas/4A96C54BF9245C61B8A4F0D94574E2D7)
- Helander 2014, theory of non-axisymmetric confinement:
  [Reports on Progress in Physics](https://doi.org/10.1088/0034-4885/77/8/087001)

These are the main references for:

- radially local drift-kinetic ordering
- thermodynamic forces
- neoclassical transport matrix structure
- bootstrap-current interpretation

## Momentum-Restoring Closure Theory

- Taguchi 1992:
  [Physics of Fluids B](https://doi.org/10.1063/1.860372)
- Sugama and Nishimura 2002:
  [Physics of Plasmas](https://doi.org/10.1063/1.1512917)
- Sugama and Nishimura 2008:
  [Physics of Plasmas](https://doi.org/10.1063/1.2902012)
- Maa{\ss}berg et al. 2009:
  [Physics of Plasmas](https://doi.org/10.1063/1.3175328)

These references matter for:

- momentum restoration beyond Lorentz pitch-angle scattering
- Sonine/Laguerre moment equations
- bootstrap-current sensitivity to higher-order closure moments
- physically justified validation gates for reduced closure models

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
