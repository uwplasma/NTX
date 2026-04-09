# NTX

NTX is a JAX-native monoenergetic neoclassical transport code for stellarator
flux surfaces. It follows the Legendre-space drift-kinetic formulation
introduced in Javier Escoto's PhD thesis,
[`arXiv:2510.27513`](https://arxiv.org/abs/2510.27513).

The current implementation supports:

- DKES-style Boozer inputs
- VMEC `wout` inputs
- VMEC `er_hat` normalization from `psi_a_hat`, `psi_n`, and `Aminor_p`
- `vmec_jax -> booz_xform_jax -> NTX` imported workflows
- direct NTX-to-NEOPAX monoenergetic database mapping
- Rich terminal summaries for file-driven runs
- compressed `.npz` outputs with geometry, metadata, diagnostics, and modes
- archived DKES and SFINCS benchmark comparison scripts
- runtime profiling scripts for CPU and GPU scans

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
examples
validation
neopax
gpu
```
