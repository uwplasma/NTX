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
- Helander and Simakov 2008, intrinsic ambipolarity and stellarator rotation:
  [Physical Review Letters](https://doi.org/10.1103/PhysRevLett.101.145003),
  [PubMed](https://pubmed.ncbi.nlm.nih.gov/18851538/)
- Landreman 2011, monoenergetic approximation limits:
  [PPCF](https://doi.org/10.1088/0741-3335/53/8/082003),
  [arXiv:1102.2508](https://arxiv.org/abs/1102.2508)
- Landreman, Smith, Mollen, and Helander 2014, trajectory and
  collision-operator comparisons:
  [Physics of Plasmas](https://doi.org/10.1063/1.4870077),
  [arXiv:1312.6058](https://arxiv.org/abs/1312.6058)
- Redl, Angioni, Belli, and Sauter 2021, analytic bootstrap-current and
  neoclassical-conductivity formulae:
  [PDF](https://pure.mpg.de/pubman/item/item_3288698_4/component/file_3288920/Redl_New.pdf)
- Landreman, Buller, and Drevlak 2022, quasisymmetric-stellarator use of the
  Redl bootstrap-current formula and comparison to a 4D drift-kinetic solver:
  [arXiv:2205.02914](https://arxiv.org/abs/2205.02914)
- Ferraro et al. 2025, implementation of Redl-style bootstrap-current modeling
  in an extended-MHD workflow using trapped fraction, collisionality, effective
  charge, and geometry factors:
  [JPP](https://www.cambridge.org/core/journals/journal-of-plasma-physics/article/bootstrap-current-modeling-in-m3dc1/07AEC30A1077F0D427FF2EA7BF42AC4B)
- Beidler et al. 2011, international monoenergetic coefficient benchmark:
  [Nuclear Fusion](https://doi.org/10.1088/0029-5515/51/7/076001)

These are the main references for:

- radially local drift-kinetic ordering
- thermodynamic forces
- the ambipolar radial-current condition that determines `E_r` in
  non-quasisymmetric stellarators
- neoclassical transport matrix structure
- bootstrap-current interpretation
- expected limits of exact parity between reduced monoenergetic workflows and
  broader drift-kinetic solvers
- why the Redl precise-QS comparison is a separate analytic bootstrap-current
  validation from the reduced NTX+NEOPAX closure stress metric
- why the finite-beta closure-target audit ranks local drivers such as
  `epsilon`, trapped fraction, and collisionality instead of introducing a
  scalar fitted current correction
- the required benchmark surface for `D11`, `D31`, and `D33`

## Spectral Collocation And Aliasing

- Orszag 1971, filtering and alias elimination for Fourier representations:
  [Journal of the Atmospheric Sciences](https://doi.org/10.1175/1520-0469%281971%29028%3C1074%3AOTEOAI%3E2.0.CO%3B2)
- Escoto et al. 2024, spectral spatial/velocity discretization and convergence
  studies for fast monoenergetic neoclassical coefficients:
  [arXiv:2312.12248](https://arxiv.org/abs/2312.12248)
- Escoto 2025 thesis, low-collisionality angular and Legendre convergence:
  [arXiv:2510.27513](https://arxiv.org/abs/2510.27513)

The classical product rules motivate explicit aliasing checks, but NTX's
operator contains reciprocal and geometry-derived coefficients that are not
strictly band limited by the retained input spectrum. The implemented policy
therefore preserves the physical collocation operator, enforces the retained-
mode Nyquist floor, measures oversampling error on several geometries, and
retains successive-grid convergence as the final gate.

## Differentiable And Optimization Workflows

- Paul, Abel, Landreman, and Dorland 2019, adjoint derivatives for
  neoclassical stellarator optimization:
  [JPP](https://doi.org/10.1017/S0022377819000527),
  [arXiv:1904.06430](https://arxiv.org/abs/1904.06430)
- McGreivy 2024, differentiable programming for computational plasma physics:
  [arXiv:2410.11161](https://arxiv.org/abs/2410.11161)
- Blondel et al. 2022, modular implicit differentiation from converged
  optimality or residual equations:
  [arXiv:2105.15183](https://arxiv.org/abs/2105.15183)
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
  than reduced-response-only validation

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
- [JAX gradient checkpointing](https://docs.jax.dev/en/latest/gradient-checkpointing.html),
  including the warning that checkpointing an entire function can add
  recomputation without reducing saved state
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
