# NTX Finalization Plan

## Goal

Ship NTX as a clean JAX-native implementation of the monoenergetic transport
formulation described in Javier Escoto's PhD thesis:
[arXiv:2510.27513](https://arxiv.org/abs/2510.27513).

## Final State

- [x] JAX-native monoenergetic solver
- [x] CLI entry point: `ntx input.toml`
- [x] DKES-style, magnetic-configuration, VMEC, and Boozer loaders
- [x] differentiable imported solve lane
- [x] direct NEOPAX scan and HDF5 mapping helpers
- [x] CPU and GPU smoke/regression scripts
- [x] package, build, and documentation scaffolding
- [x] removal of vendored benchmark families and non-NEOPAX external datasets
- [x] replacement of external repository fixtures with NTX-authored synthetic fixtures

## Current Validation Summary

- local test suite passes
- GPU smoke tests are available and skip cleanly on non-GPU machines
- office GPU hardware validation closed successfully with the NTX-owned smoke cases

## Remaining Maintenance Work

- keep synthetic loader fixtures minimal and readable
- keep NEOPAX mapping helpers aligned with the active NEOPAX interface
- continue profiling larger production grids when performance work is needed
