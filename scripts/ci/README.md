# CI scripts

Stdlib-only Python checks invoked from `.github/workflows/ci.yml`.
Each script is runnable directly (`python scripts/ci/<name>.py`),
exits non-zero on failure, and prints every error to stderr so CI
logs surface the full problem rather than the first failure.

Checks currently wired:

- **`check_markdown_links.py`** — every relative markdown link
  (`[text](path)` or `[text](path#anchor)`) must resolve to an
  existing file. External URLs and pure in-page fragments are
  skipped. Fenced code blocks are ignored.
- **`check_adr_index.py`** — every ADR file in `adr/` is linked
  from `adr/README.md`, and every link in the index points to a
  file that exists.

Planned next:

- **`check_replication_specs.py`** — capability and problem specs
  under `replication/` are mutually consistent: every capability
  referenced by a problem's "Capabilities required" field exists,
  and every capability's "Dependents" list matches the set of
  problems that require it.

Adding a check: write `scripts/ci/check_<name>.py` using stdlib
only, add one step to `.github/workflows/ci.yml` invoking it, and
list the check in this README.
