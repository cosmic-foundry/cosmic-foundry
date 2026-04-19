# CI scripts

Python checks invoked from `.github/workflows/ci.yml`. Most are
stdlib-only; `check_action_versions.py` uses `packaging.version`
(already a transitive dep via pip). Each script is runnable directly
(`python scripts/ci/<name>.py`), exits non-zero on failure, and
prints every error to stderr so CI logs surface the full problem
rather than the first failure.

Checks currently wired:

- **`check_markdown_links.py`** — every relative markdown link
  (`[text](path)` or `[text](path#anchor)`) must resolve to an
  existing file. External URLs and pure in-page fragments are
  skipped. Fenced code blocks are ignored.
- **`check_action_versions.py`** — `uses: owner/action@vN` pins in
  `.github/workflows/*.yml` must not regress below their version on
  `origin/main`. Intentional downgrades require a
  `# allow-downgrade: <reason>` comment on the same line as the
  `uses:` entry or the line directly above it. SHA and branch pins
  are skipped. Requires `origin/main` to be fetched; CI does this
  via `git fetch --depth=1 origin main` before running pre-commit.

Planned next:

- **`check_replication_specs.py`** — capability and problem specs
  under `replication/` are mutually consistent: every capability
  referenced by a problem's "Capabilities required" field exists,
  and every capability's "Dependents" list matches the set of
  problems that require it.

Adding a check: write `scripts/ci/check_<name>.py` using stdlib
only, add one step to `.github/workflows/ci.yml` invoking it, and
list the check in this README.
