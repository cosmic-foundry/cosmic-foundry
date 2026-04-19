# Contributing to Cosmic Foundry

## Environment setup

The repo ships its own Python environment. You must use it — running
with system Python or a different conda env causes silent misconfiguration.

```bash
# One-time: install miniforge, create the cosmic_foundry conda env,
# install the package in editable mode, and register the pre-commit hook
bash environment/setup_environment.sh

# Every session: activate before doing any work
source environment/activate_environment.sh
```

After a pull that changes `pyproject.toml`, refresh the editable install
inside the activated env with `pip install -e .[dev,docs]` — no need to
re-run the full setup script.

## Workflow

1. **Fork** the upstream repo (`cosmic-foundry/cosmic-foundry`).
2. **Branch** from `origin/main` (your fork's main), not from `upstream/main`:
   ```bash
   git checkout -b feat/my-change origin/main
   ```
3. **Make changes.** Target ~150 lines of diff per commit, ceiling ~400.
   Documentation diffs, generated files, and pure scaffolding are exempt.
4. **Check before pushing:**
   ```bash
   pre-commit run --all-files
   pytest -q
   ```
5. **Open a PR against `upstream/main`** (not against the fork):
   ```bash
   gh pr create \
     --repo cosmic-foundry/cosmic-foundry \
     --base main \
     --head <fork-owner>:<branch>
   ```

## Commit style

- One logical change per commit.
- Prefix with a type: `feat:`, `fix:`, `docs:`, `chore:`, `test:`.
- AI-agent commits carry a `Co-Authored-By` trailer.
- No local absolute paths in commit messages or PR descriptions.

## Architectural decisions

New decisions go in the appropriate architecture plane under
`adr/object-level/` or `adr/meta-level/`. Copy `adr/adr-template.md`,
fill it in, mark it Proposed, and add a line to `adr/README.md` in the
same PR. See [AI.md](AI.md) for the full ADR process.

## Code standards

See [DEVELOPMENT.md](DEVELOPMENT.md) for style, type-annotation, docstring
format, and test requirements.

## Getting help

Open a GitHub issue on `cosmic-foundry/cosmic-foundry`.
