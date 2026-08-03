# NTX

NTX is a JAX-native monoenergetic neoclassical transport solver based on the
Legendre-space drift-kinetic formulation in Javier Escoto's PhD thesis,
[arXiv:2510.27513](https://arxiv.org/abs/2510.27513).

NTX provides:

- built-in analytic sample surfaces
- DKES-style Boozer inputs
- VMEC `wout` inputs through `vmex`
- Boozer `boozmn` inputs through `booz_xform_jax`
- direct NEOPAX-style scan and HDF5 mapping helpers
- CPU and GPU execution through the same JAX solver path

## Fastest Start

Install the package:

```bash
pip install ntx
```

Run the bundled analytic sample:

```bash
ntx examples/example_surface.toml --plot
```

That solves one monoenergetic case, prints a Rich summary, and writes a
NetCDF payload plus a PDF summary panel.

Inspect that output graphically:

```bash
python examples/plot_output_file.py examples/outputs/example_surface.nc
```

## Main User Entry Point

```bash
ntx input.toml
```

## What The Code Solves

NTX solves the normalized local monoenergetic drift-kinetic equation for the
non-adiabatic response on a single flux surface,

```{math}
\left[
\xi \frac{1}{B}\left(B^\theta\partial_\theta+B^\zeta\partial_\zeta\right)
+ \frac{\hat E_\psi}{\mathcal J\langle B^2\rangle}
\left(-B_\zeta\partial_\theta+B_\theta\partial_\zeta\right)
- \frac{1-\xi^2}{2B^2}
\left(B^\theta\partial_\theta B+B^\zeta\partial_\zeta B\right)\partial_\xi
- C_L
\right] f = s,
```

where `\xi=v_\parallel/v`, `\mathcal J` is the Boozer-coordinate Jacobian, and
`C_L` is the Lorentz pitch-angle-scattering operator. Speed is fixed. The two
source systems are the radial-transport drive and the parallel-flow drive used
by downstream bootstrap-current closures. Projecting this equation onto
Legendre polynomials in pitch angle gives the system solved by the code,

```{math}
L_k f^{(k-1)} + D_k f^{(k)} + U_k f^{(k+1)} = s^{(k)}
```

on a single stellarator flux surface, then evaluates the monoenergetic
coefficients

```{math}
\hat D_{11},\ \hat D_{31},\ \hat D_{13},\ \hat D_{33},\ \hat D_{33,\mathrm{Sp}}.
```

The CLI is tuned for production solves and output inspection. The imported JAX
lane is tuned for scans, autodiff, optimization, NEOPAX coupling, and
parallel-throughput workflows.

## Documentation Guide

- [Install](install.md): package installation and extras
- [Input File](input-file.md): full TOML schema, outputs, and CLI behavior
- [Physics Model](physics.md): the equations and normalizations
- [Physics Gates](physics-gates.md): analytical identities and benchmark
  acceptance rules
- [Geometry And Inputs](geometry.md): how surfaces are loaded and evaluated
- [Numerics And Algorithms](numerics.md): discretization, dense solve, and
  JAX/parallel execution
- [Resolution And Convergence](convergence.md): residual semantics, Nyquist
  gates, and adaptive angular/Legendre ladders
- [Source-Code Map](source-map.md): where each model component lives in `src/`
- [API Reference](api.rst): generated public module, class, and function docs
- [Glossary](glossary.md): coordinates, normalizations, grids, and workflow terms
- [Autodiff](autodiff.md): inverse problems, derivative audits, and prepared derivatives
- [Profiles](profiles.md): ambipolar electric-field and reduced current-response workflows
- [Examples](examples.md): runnable workflows and figure generators
- [Validation](validation.md): current status and benchmark philosophy
- [Testing And QA](testing.md): test structure and quality gates
- [Benchmark Matrix](benchmark-matrix.md): claim-to-script/test/artifact mapping
- [Repository Hygiene](repo-hygiene.md): local cleanup and commit-batch plan
- [Pre-Merge Ship Checklist](ship-checklist.md): release blockers and acceptance criteria
- [NEOPAX](neopax.md): imported database-building workflows
- [GPU](gpu.md): hardware execution notes
- [Performance](performance.md): throughput guidance and scaling figures
- [Research Roadmap](research-roadmap.md): next research-grade development lanes
- [Manuscript Figures](manuscript.md): publication-ready figure inventory
- [Literature](literature.md): thesis and package links
- [Release Notes 0.3.0](release-notes-0.3.0.md): current release notes
- [Release Notes 0.2.4](release-notes-0.2.4.md): previous release notes

## Contents

```{toctree}
:maxdepth: 2

install
input-file
physics
physics-gates
geometry
algorithm
numerics
convergence
source-map
api
glossary
autodiff
profiles
examples
validation
sfincs-jax-rhsmode1-profile-current-handoff
testing
benchmark-matrix
repo-hygiene
ship-checklist
neopax
gpu
performance
research-roadmap
manuscript
literature
release
release-notes-0.3.0
release-notes-0.2.4
release-notes-0.2.3
release-notes-0.2.2
release-notes-0.2.1
release-notes-0.2.0
```
