# Benchmarks

Run benchmark scripts from the repository root inside the project
environment.

## Pointwise Triad Roofline

```bash
python benchmarks/pointwise_roofline.py --n 128 --repeats 10
```

The roofline benchmark times a pointwise triad through the public `Dispatch`
path:

```python
c = a + 0.5 * b
```

The pointwise Op uses `Stencil((0, 0, 0))` as the temporary zero-halo access
pattern. That keeps the benchmark inside the current access-pattern surface
while avoiding the cache-reuse ambiguity of a stencil benchmark.

The memory roofline denominator is a local STREAM-like triad measured in the
same process with JAX:

```python
a + scalar * b
```

The reported `memory_roofline_fraction` is:

```text
dispatch_triad_gb_s / stream_triad_gb_s
```

The bandwidth values use median timings from the measured repetitions so the
ratio is not dominated by one unusually fast sample.

This keeps the number portable across developer laptops and CI runners
without committing a machine-specific peak bandwidth into the repository.
Very small grids are useful for smoke tests but are dominated by dispatch and
timer overhead; use production-sized grids when recording roofline fractions.
