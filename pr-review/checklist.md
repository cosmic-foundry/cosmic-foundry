# PR-review checklist — historical failure modes

The synthesis reviewer walks this file on every PR. It is a
catalog of failure modes that have escaped self-review on this
repository, organized by theme.

**Scope:** the reviewer works against the PR diff as a whole
(squash merge is enforced upstream, so the diff against the
target branch is the unit of review — individual commit history
is not examined).

**Relationship to pre-commit:** mechanical pattern-matching
checks belong in `scripts/ci/` and `.pre-commit-config.yaml`,
not in this checklist. Items marked *(pending pre-commit)* are
not yet automated; once a script exists and is wired to the
hook, delete the entry here. Items already covered by pre-commit
are not listed.

Add new entries when a failure escapes review. Delete entries
when pre-commit now catches them.

---

## Docs integrity

- **Index entries missing.** Adding a research-notes subsection
  requires a link from `docs/research/index.md`. Adding a capability epoch
  requires a row in `ROADMAP.md`. Adding an architectural decision
  requires a paragraph in `ARCHITECTURE.md`. Check these manually —
  they are not automated.
- **Cross-references broken by renames/moves.** If the PR
  renames or moves a file, grep the repo for the old path.
  `check_markdown_links.py` validates own-repo GitHub URLs but
  not relative markdown links between files in the same tree.
- **Sphinx `{include}` targets reachable.** *(pending
  pre-commit)* MyST-NB `{include}` directives fail silently if
  the target is outside the docs source tree or the path is
  wrong. Verify any new include target exists and is reachable
  from `docs/`.
- **Stale codebase references.** Comments, docstrings, and prose
  that describe the *previous* state of the code rather than its
  current state are defects. Flag: "previously X", "the old
  implementation", "before the refactor", "this replaces Y",
  references to symbols or files that no longer exist, and
  forward-references to code that is not yet present (these
  should be explicit `TODO` markers with a tracking reference,
  not narrative prose). The test: does every claim hold against
  the post-merge working tree, with no knowledge of how it
  arrived there?

## CI / pre-commit parity

- **New check added to only one of CI or pre-commit.** The
  project treats them as a single gate. If the PR adds a hook
  to `.pre-commit-config.yaml` without a matching CI step (or
  vice versa), call it out and explain why — deliberate
  asymmetry is fine when justified.

## Bootstrap and cross-PR dependencies

- **Bootstrap ignores that leave debt.** *(pending pre-commit)*
  Grep the diff for newly added `_ignore`, `xfail`, `skip`,
  `TODO`, or `FIXME` patterns with a comment like "we'll fix
  this in the next PR". The project's position is to solve the
  problem at the right layer rather than defer.
- **Dependencies on an unmerged PR.** If the PR references a
  file, symbol, or config that only exists on another open
  branch, the merge order must be explicit in the PR
  description.

## Redundancy with existing scripts / tooling

- **Reimplementing a script or hook.** Before approving new
  setup / lint / check logic, check `scripts/` and
  `.pre-commit-config.yaml` for existing coverage.
- **Duplicated install/activate steps in docs.** *(pending
  pre-commit)* If the PR adds a manual `pip install -e` or
  `pre-commit install` step to docs, and those already run
  inside `scripts/setup_environment.sh`, the doc should
  point at the script instead.

## Project framing and tone

- **Tech-stack-first framing.** Top-level descriptions (README,
  `docs/index.md`, roadmap epoch summaries) should lead with the
  *physics / mission* goal, not "built on JAX" or similar. JAX,
  unyt, Zarr, etc. belong in `ARCHITECTURE.md` and deeper docs.
- **Backwards-compatibility cruft.** The project has not cut a
  stable API. If the PR preserves legacy names, adds
  deprecation warnings, or keeps a shim "just in case", push
  back — the expected mode is structural edits.
- **Conversational or agent-session prose.** Code comments,
  docstrings, `ARCHITECTURE.md`, and roadmap prose must read as durable
  documentation, not transcribed conversation. Flag: first- or
  second-person voice ("we decided", "you should", "I think"),
  phrases that presuppose a live exchange ("as discussed", "as
  mentioned above", "based on our conversation"), and reasoning
  that only makes sense in the context of a specific session
  rather than the repository itself. The test: does this
  sentence stand alone to a reader who was not present?

## Spelling and copy

- **British vs. American spelling.** Project standard is
  American English, enforced via codespell
  (`--builtin=en-GB_to_en-US`) on `.md`, `.py`, and `.rst`
  files. If the PR touches file types not covered by the hook,
  spot-check manually.

## Architectural-option bias

- **Simpler option not named.** Per AI.md §*Weighing
  architectural options*, when the PR justifies an
  architectural choice, confirm the lower-complexity
  alternative was named explicitly and compared on downstream
  cost (reviewer load, ops cost, reversibility, correctness,
  blast radius). "Richer option chosen because it was easy to
  type" is the specific failure mode.

## Tests

- **New code path without a test.** For non-trivial logic:
  anything with a branch, I/O, user input, or kernel/numerics.
- **Test asserts the wrong thing.** Tests that only assert "no
  exception raised" or compare a value to itself are
  tautological. A green suite is not proof of correctness.
- **Unexplained golden-data changes.** If the PR updates a
  fixture or baseline image, the commit message or PR
  description must explain the delta. An unexplained change is
  a review blocker.

## Durable metadata leaks

- **Local filesystem paths.** *(pending pre-commit)* Scan the
  diff, commit messages, and PR body for `/Users/`, `/home/`,
  `C:\Users\` patterns. Use repository-relative paths instead.
- **Secrets.** *(pending pre-commit)* Scan the diff for
  anything resembling an API key, token, or password: long
  base64 strings, `sk-…` / `ghp_…` / `AKIA…` prefixes,
  `password\s*=`, `token\s*=`. If one landed, the fix is
  rotation, not history rewriting.

## Scope and minimalism

- **PR diff size.** Squash merge is enforced, so the PR diff is
  the commit. Non-docs, non-generated code over ~400 LOC
  warrants a flag — either split or explicit justification in
  the PR description.
- **Scope creep.** Does every hunk serve the PR's stated goal?
  Drive-by refactors, formatting churn in untouched files, and
  opportunistic renames should be a separate PR unless called
  out in the description.
- **Premature abstraction.** New helpers, base classes, or
  configuration knobs with exactly one caller — inline them
  unless the second caller is imminent and named.
- **Unused code.** Dead functions, unused imports, stubs "for
  later". CI catches some; reviewers catch the rest.
