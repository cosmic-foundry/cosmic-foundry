# Epoch 1 — Kernel abstraction and multi-backend core

> Part of the [Cosmic Foundry roadmap](index.md).

The one piece of infrastructure we genuinely have to invent:

- A `Kernel` descriptor — a declarative spec of a stencil or
  element-wise operation over typed arrays with named axes — plus
  adapters that lower it to JAX, Numba, Taichi, Warp, or Triton at
  runtime.
- A `ShardedField` distributed-array primitive built on
  `jax.distributed` + mpi4py for cross-device / cross-host cases.
- Parallel HDF5 I/O via h5py+MPI.
- Deterministic structured logging and error handling.

**Exit criterion:** a 3-D 7-point Laplacian benchmark runs under
every backend at documented roofline fractions; a multi-rank
correctness test proves backend and sharding equivalence; and a
reference render of one benchmark slice under the house colormap
is committed as the first production visual-regression artifact.
