# NTX

NTX is a JAX-native monoenergetic neoclassical transport solver based on the
Legendre-space drift-kinetic formulation in Javier Escoto's PhD thesis,
[arXiv:2510.27513](https://arxiv.org/abs/2510.27513).

NTX provides:

- built-in analytic sample surfaces
- DKES-style Boozer inputs
- VMEC `wout` inputs through `vmec_jax`
- Boozer `boozmn` inputs through `booz_xform_jax`
- direct NEOPAX-style scan and HDF5 mapping helpers
- CPU and GPU execution through the same JAX solver path

## Fastest Start

Install the package:

```bash
python -m pip install ntx
```

Run the bundled analytic sample:

```bash
ntx examples/example_surface.toml
```

That solves one monoenergetic case, prints a Rich summary, and writes a
compressed `.npz` payload next to the input file.

Inspect that output graphically:

```bash
python examples/plot_output_npz.py examples/example_surface.npz
```

## Main User Entry Point

```bash
ntx input.toml
```

## What The Code Solves

NTX solves the Legendre-space monoenergetic system

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
- [Geometry And Inputs](geometry.md): how surfaces are loaded and evaluated
- [Numerics And Algorithms](numerics.md): discretization, dense solve, and
  JAX/parallel execution
- [Source-Code Map](source-map.md): where each model component lives in `src/`
- [Autodiff](autodiff.md): inverse problems, derivative audits, and prepared derivatives
- [Profiles](profiles.md): ambipolar electric-field and bootstrap-current proxy workflows
- [Examples](examples.md): runnable workflows and figure generators
- [Validation](validation.md): current status and benchmark philosophy
- [Testing And QA](testing.md): test structure and quality gates
- [NEOPAX](neopax.md): imported database-building workflows
- [GPU](gpu.md): hardware execution notes
- [Performance](performance.md): throughput guidance and scaling figures
- [Research Roadmap](research-roadmap.md): next research-grade development lanes
- [Manuscript Figures](manuscript.md): publication-ready figure inventory
- [Literature](literature.md): thesis and package links

## Contents

```{toctree}
:maxdepth: 2

install
input-file
physics
geometry
algorithm
numerics
source-map
autodiff
profiles
examples
validation
testing
neopax
gpu
performance
research-roadmap
manuscript
literature
release
```
