# PR-review checklist — historical failure modes

The reviewer walks this file end-to-end on every PR. It is a
catalogue of failure modes that have actually escaped
self-review on this repository, organised by theme. Each theme
states *what to look for*, *why it matters*, and (where useful)
*how to probe it* in the current PR.

Add new entries when a failure escapes review. Delete entries
when the failure mode becomes impossible because a linter,
pre-commit hook, or CI check now catches it — don't leave stale
line items.

## Docs integrity

- **Own-repo URLs point at unmerged paths.** Markdown links to
  `https://github.com/cosmic-foundry/cosmic-foundry/{blob,tree,raw}/main/…`
  are validated by `scripts/ci/check_markdown_links.py` against
  the working tree. Check any newly added URL of that shape
  resolves to a file that exists *in this PR*, not one the PR
  author is about to add in a later PR.
- **Sphinx `{include}` across directory trees.** MyST-NB
  `{include}` directives fail silently if the included file is
  outside the docs source tree or the path is wrong. Verify that
  any new include target is reachable from `docs/`.
- **Index entries missing.** Adding an ADR requires a line in
  `adr/README.md`. Adding a research-notes subsection requires
  a link from `research/index.md`. Adding a roadmap epoch
  requires an entry in `roadmap/index.md`. Confirm the index
  was updated in the same PR.
- **Cross-references broken by renames/moves.** If the PR
  renames or moves a file, grep the repo for the old path.
  Broken cross-refs will not fail CI if they are written as
  plain markdown links between files in the same tree.

## CI / pre-commit parity

- **New check added to CI but not pre-commit (or vice versa).**
  The project treats pre-commit and CI as a single gate. A
  check added to `.pre-commit-config.yaml` should also run in
  the CI workflow, and vice versa. If the PR adds a check in
  only one place, call it out.
- **GitHub Actions version drift.** `scripts/ci/check_action_versions.py`
  exists to catch downgrades. If the PR touches `.github/workflows/`,
  confirm all action versions are pinned to a SHA or a specific tag
  and that none are downgraded relative to `main`.
- **Skipped hooks.** Any use of `--no-verify`, `--no-gpg-sign`,
  `SKIP=<hook>`, or disabled pre-commit stages is a red flag and
  should be explicitly justified in the PR.

## Bootstrap and cross-PR dependencies

- **Bootstrap ignores that leave debt for a follow-up PR.** If
  the PR adds an `_ignore` or `xfail` or `skip` pattern with a
  comment like "we'll fix this in the next PR", flag it. The
  project's position is to solve the problem at the right layer
  rather than defer.
- **Dependencies on an unmerged PR.** If this PR references a
  file, symbol, or config that only exists on another open
  branch, the merge order must be explicit. Check whether the
  PR description calls that out.

## Redundancy with existing scripts / tooling

- **Reimplementing a script or hook.** Before approving new
  setup / lint / check logic, grep `scripts/` and
  `.pre-commit-config.yaml` for existing coverage. The project
  has repeatedly tried to add a "manual" setup step for
  something that is already automated elsewhere.
- **Duplicated install/activate steps in docs.** If the PR adds
  a manual `pip install -e` or `pre-commit install` step to
  docs, and those already run inside
  `environment/setup_environment.sh`, the doc change should
  point at the script, not duplicate it.

## Project framing and tone

- **Tech-stack-first framing.** Top-level descriptions (README,
  `docs/index.md`, ADR motivations, roadmap epoch summaries)
  should lead with the *physics / mission* goal, not "built on
  JAX" or similar. JAX, unyt, Zarr, etc. belong in ADRs and
  deeper docs. If the PR edits framing-layer copy, check the
  reframing didn't drift back toward tech-first.
- **Backwards-compatibility cruft.** The project has not cut a
  stable API. If the PR preserves legacy names, adds deprecation
  warnings, or keeps a shim "just in case", push back — the
  expected mode is structural edits.

## Spelling and copy

- **British vs. American spelling.** Project standard is
  American English, enforced via codespell. If codespell isn't
  yet in pre-commit for this PR's file types (check
  `.pre-commit-config.yaml`), spot-check new copy for
  `colour/color`, `behaviour/behavior`, `organise/organize`,
  `centre/center`.

## Architectural-option bias

- **Recommendations skipping the simpler option.** Per AI.md
  §*Weighing architectural options*, when the PR justifies a
  choice in an ADR or design note, confirm the lower-complexity
  alternative was named explicitly and compared on downstream
  cost (reviewer load, ops cost, reversibility, correctness,
  blast radius), not on author effort. "Richer option chosen
  because it was easy to type" is a specific failure mode here.

## Tests

- **New code path without a test.** For non-trivial logic, look
  for a paired test. "Non-trivial" here includes: anything with
  a branch, anything with I/O, anything that handles user
  input, and any kernel/numerics change.
- **Test asserts the wrong thing.** Watch for tests that only
  assert "no exception raised" or that compare a value to
  itself. A green test suite is not proof of correctness if the
  asserts are tautological.
- **Golden-data regressions.** If the PR updates a fixture or
  baseline image without stating why, that's a review blocker
  until the delta is explained.

## Commit hygiene

- **Commit size over ceiling without justification.** Code
  diffs target ~150 LOC with a soft ceiling of ~400. Past the
  ceiling the PR description should either split the commit or
  explain why a split would harm review. Docs-only diffs are
  exempt.
- **One commit mixing logical changes.** "Refactor + bugfix +
  new feature" in one commit makes bisect useless. Flag and
  suggest a split.
- **Missing attribution trailer.** AI-agent commits must carry
  a `Co-Authored-By` trailer naming the agent and model
  (ADR-0005). Check every commit in the PR.
- **Force-push on an open PR.** Per ADR-0005, force-push on an
  open PR is not allowed without explicit user approval. If the
  PR was force-pushed, check the discussion for that approval.

## Durable metadata leaks

- **Local filesystem paths in commit messages or PR body.** No
  `/Users/…`, `/home/…`, `C:\Users\…` in commit messages, PR
  titles, PR descriptions, ADR text, or any tracked file. Use
  repository-relative paths.
- **Secrets.** Scan the diff for anything that looks like an
  API key, token, or password. If one landed, the fix is
  rotation, not history rewriting.

## Scope and minimalism

- **Scope creep.** Does every hunk serve the PR's stated goal?
  Drive-by refactors, formatting churn in untouched files, and
  opportunistic renames should be split into a follow-up PR
  unless the PR description calls them out.
- **Premature abstraction.** New helpers, base classes, or
  configuration knobs that have exactly one caller are a smell
  — inline them unless the second caller is imminent and named.
- **Unused code.** Dead functions, unused imports, and stubs
  "for later" should not land. CI catches some of this;
  reviewers catch the rest.
