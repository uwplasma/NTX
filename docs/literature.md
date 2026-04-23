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
- Landreman 2011, monoenergetic approximation limits:
  [PPCF](https://doi.org/10.1088/0741-3335/53/8/082003),
  [arXiv:1102.2508](https://arxiv.org/abs/1102.2508)
- Landreman, Smith, Mollen, and Helander 2014, trajectory and
  collision-operator comparisons:
  [Physics of Plasmas](https://doi.org/10.1063/1.4870077),
  [arXiv:1312.6058](https://arxiv.org/abs/1312.6058)
- Beidler et al. 2011, international monoenergetic coefficient benchmark:
  [Nuclear Fusion](https://doi.org/10.1088/0029-5515/51/7/076001)

These are the main references for:

- radially local drift-kinetic ordering
- thermodynamic forces
- neoclassical transport matrix structure
- bootstrap-current interpretation
- expected limits of exact parity between reduced monoenergetic workflows and
  broader drift-kinetic solvers
- the required benchmark surface for `D11`, `D31`, and `D33`

## Differentiable And Optimization Workflows

- Paul, Abel, Landreman, and Dorland 2019, adjoint derivatives for
  neoclassical stellarator optimization:
  [JPP](https://doi.org/10.1017/S0022377819000527),
  [arXiv:1904.06430](https://arxiv.org/abs/1904.06430)
- McGreivy 2024, differentiable programming for computational plasma physics:
  [arXiv:2410.11161](https://arxiv.org/abs/2410.11161)
- Lee, Lazerson, Smith, Beidler, and Pablant 2024, direct optimization of
  neoclassical ion transport in stellarator reactors:
  [Nuclear Fusion](https://doi.org/10.1088/1741-4326/ad75a6),
  [arXiv:2406.04147](https://arxiv.org/abs/2406.04147)

These references anchor NTX's autodiff tests:

- direct automatic differentiation against centered finite differences
- prepared implicit or adjoint derivatives for many controls
- inverse-design recovery from generated targets
- uncertainty propagation from Jacobians
- profile and geometry optimization with explicit physical metrics rather
  than proxy-only validation

## Geometry-Breadth And Future Benchmark Families

- Plunk, Landreman, and Helander 2019, direct construction of omnigenous
  magnetic fields near the magnetic axis:
  [JPP](https://www.cambridge.org/core/journals/journal-of-plasma-physics/article/direct-construction-of-optimized-stellarator-shapes-part-3-omnigenity-near-the-magnetic-axis/A4BDBC48ADD43736C097EC9BFFA0A73D),
  [arXiv:1909.08919](https://arxiv.org/abs/1909.08919)
- Rodríguez, Plunk, and Jorge 2025, second-order quasi-isodynamic near-axis
  construction:
  [JPP](https://doi.org/10.1017/S0022377825000157)
- Bindel, Landreman, and Padidar 2023/2025, direct optimization of fast-ion
  confinement:
  [PPCF](https://doi.org/10.1088/1361-6587/acd141),
  [arXiv:2302.11369](https://arxiv.org/abs/2302.11369)
- Calvo, Velasco, Helander, and Parra 2025, piecewise omnigenous fields with
  zero bootstrap current:
  [Phys. Rev. E](https://doi.org/10.1103/tnh1-mq88),
  [arXiv:2505.02546](https://arxiv.org/abs/2505.02546)
- Liu, Yu, Velasco, and Zhu 2026, combined omnigenity and piecewise-omnigenity
  optimization:
  [arXiv:2603.12139](https://arxiv.org/abs/2603.12139)

These papers motivate the planned geometry-breadth lane. NTX should not promote
hidden-symmetry, quasi-isodynamic, or omnigenous validation claims until the
corresponding reusable geometry inputs, normalization audits, and convergence
ladders are owned by the repository.

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
- [Lineax](https://docs.kidger.site/lineax/) for possible structured linear
  solves after profiling identifies a real solve bottleneck
- [Equinox](https://docs.kidger.site/equinox/) for possible PyTree/module and
  filtered-transform ergonomics after the public API boundaries are settled

Use these packages conservatively. The current NTX performance profile says the
near-term speed lane is stable shapes, reusable compiled functions, prepared
geometry reuse, and clear compile-versus-steady-state accounting. New
dependencies should follow a measured profile improvement, not precede it.

## Independent Validation Ecosystem

NTX users often want to compare against other neoclassical tools or pipelines.
The repository documentation refers to:

- [SFINCS-JAX](https://github.com/uwplasma/sfincs_jax) when discussing
  independent consistency checks

These packages are useful for trust-building and application workflows, but
NTX's equations, numerics, and public interface are defined by its own source
tree and the Escoto thesis.
