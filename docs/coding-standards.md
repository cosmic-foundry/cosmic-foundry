# Coding standards

## Style

- **Formatter:** [black](https://black.readthedocs.io/en/stable/) (line length 88,
  default settings). Run automatically by the pre-commit hook.
- **Linter:** [ruff](https://docs.astral.sh/ruff/) with the rule sets
  configured in `pyproject.toml` (pycodestyle, pyflakes, isort, bugbear,
  pyupgrade, numpy). Ruff auto-fixes safe issues on commit.
- **Editor config:** `.editorconfig` at the repo root enforces 4-space
  indentation, LF line endings, and trailing-whitespace trimming.

## Type annotations

All public functions and methods in `cosmic_foundry/` must carry full type
annotations. mypy is run in CI with the settings in `[tool.mypy]` in
`pyproject.toml` (curated strict-ish — see that section for the exact flags).

JAX arrays are typed as `jax.Array`. Avoid bare `Any` in signatures; if a
JAX stub forces it, add a `# type: ignore[...]` with a comment explaining why.

## Docstrings

Use [NumPy docstring format](https://numpydoc.readthedocs.io/en/latest/format.html):

```python
def pressure(rho: float, gamma: float) -> float:
    """Compute ideal-gas pressure.

    Parameters
    ----------
    rho:
        Mass density [g cm⁻³].
    gamma:
        Adiabatic index (dimensionless).

    Returns
    -------
    float
        Pressure [dyn cm⁻²].
    """
    return (gamma - 1.0) * rho
```

## Testing

- Tests live under `tests/`, mirroring the package layout.
- Every new capability needs a test. See
  [ADR-0007](https://github.com/cosmic-foundry/cosmic-foundry/blob/main/adr/ADR-0007-replication-workflow.md)
  for the verification-first discipline applied to numerical
  capabilities.
- Visual regression tests use `pytest-mpl` and live under `tests/visual/`.
  Generate new baselines with `pytest --mpl-generate-path=tests/visual/baseline`.
- The `multihost` marker gates tests that require `jax.distributed`; they are
  not run in default CI (see `pyproject.toml` markers).

## Commit size

Target ~150 lines of diff per commit, ceiling ~400. See
[`AI.md`](https://github.com/cosmic-foundry/cosmic-foundry/blob/main/AI.md)
for the full guideline (exemptions for generated files, documentation,
and pure scaffolding).
