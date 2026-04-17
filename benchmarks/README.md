# Benchmarks

Run benchmark scripts from the repository root inside the project
environment.

## Laplacian Roofline

```bash
python benchmarks/laplacian_roofline.py --n 128 --repeats 10
```

The Laplacian roofline benchmark times the public `Dispatch` path for the
3-D seven-point stencil. It reports an effective memory bandwidth using
eight double-precision memory operations per interior output cell: seven
input reads and one output write.

The memory roofline denominator is a local STREAM-like triad measured in the
same process with JAX:

```python
a + scalar * b
```

The reported `memory_roofline_fraction` is:

```text
laplacian_effective_gb_s / stream_triad_gb_s
```

This keeps the number portable across developer laptops and CI runners
without committing a machine-specific peak bandwidth into the repository.
Very small grids are useful for smoke tests but are dominated by dispatch and
timer overhead; use production-sized grids when recording roofline fractions.
