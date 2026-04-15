# Contributing to Cosmic Foundry

Full contributor guide to be written in Epoch 0 (§0.8). The short version:

1. Fork the repo, create a topic branch from `origin/main`.
2. Run `bash environment/setup_environment.sh` once; then
   `source environment/activate_environment.sh` to enter the env.
3. `pip install -e .[dev]` and `pre-commit install`.
4. Make changes, run `pre-commit run --all-files` and `pytest`.
5. Open a PR against `upstream/main` (not the fork).

See [AI.md](AI.md) for the authoritative branch, commit-size, and
attribution rules.
