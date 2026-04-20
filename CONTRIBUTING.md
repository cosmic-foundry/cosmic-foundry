# Contributing to Cosmic Foundry

## Environment setup

The repo ships its own Python environment. You must use it — running
with system Python or a different conda env causes silent misconfiguration.

```bash
# One-time: install miniforge, create the cosmic_foundry conda env,
# install the package in editable mode, and register the pre-commit hook
bash scripts/setup_environment.sh

# Every session: activate before doing any work
source scripts/activate_environment.sh
```

After a pull that changes `pyproject.toml`, refresh the editable install
inside the activated env with `pip install -e .[dev,docs]` — no need to
re-run the full setup script.

## Workflow

1. **Branch** from `main`:
   ```bash
   git checkout -b feat/my-change main
   ```
2. **Make changes.** Target ~150 lines of diff per commit, ceiling ~400.
   Documentation diffs, generated files, and pure scaffolding are exempt.
3. **Check before pushing:**
   ```bash
   pre-commit run --all-files
   pytest -q
   ```
4. **Open a PR against `main`**:
   ```bash
   gh pr create \
     --repo cosmic-foundry/cosmic-foundry \
     --base main
   ```

## Commit style

- One logical change per commit.
- Prefix with a type: `feat:`, `fix:`, `docs:`, `chore:`, `test:`.
- AI-agent commits carry a `Co-Authored-By` trailer.
- No local absolute paths in commit messages or PR descriptions.

## Getting help

Open a GitHub issue on `cosmic-foundry/cosmic-foundry`.
