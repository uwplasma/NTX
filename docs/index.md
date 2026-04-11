# NTX

NTX is a JAX-native monoenergetic neoclassical transport solver based on the
Legendre-space drift-kinetic formulation in Javier Escoto's PhD thesis,
[arXiv:2510.27513](https://arxiv.org/abs/2510.27513).

The current codebase supports:

- built-in analytic sample surfaces
- DKES-style Boozer inputs
- VMEC `wout` inputs through `vmec_jax`
- Boozer `boozmn` inputs through `booz_xform_jax`
- direct NEOPAX-style scan and HDF5 mapping helpers
- CPU and GPU execution through the same JAX solver path

## Main Entry Point

```bash
ntx input.toml
```

## Contents

```{toctree}
:maxdepth: 2

install
input-file
algorithm
autodiff
examples
validation
neopax
gpu
release
```
