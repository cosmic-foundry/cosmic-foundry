# PR-review checklist — historical failure modes

The reviewer walks this file end-to-end on every PR. It is a
catalog of failure modes that have actually escaped
self-review on this repository, organized by theme. Each theme
states *what to look for*, *why it matters*, and (where useful)
*how to probe it* in the current PR.

Add new entries when a failure escapes review. Delete entries
when the failure mode becomes impossible because a linter,
pre-commit hook, or CI check now catches it — don't leave stale
line items.

## Review-tier tags

Each bullet is prefixed with `[sweep]` or `[synthesis]`:

- **`[sweep]`** — mechanical checks. Pattern-matching, grep,
  file-presence, diff statistics. Handled by the cheap/fast
  sweep pass.
- **`[synthesis]`** — judgment-heavy checks. Intent, scope,
  taste, consistency-with-ADRs, gap-hunt. Handled by the
  stronger synthesis pass, which also receives the sweep
  output as input.

The split is deliberately conservative: when in doubt, an item
is tagged `[synthesis]`. A false-negative from the sweep (missed
a judgment item) is worse than a false-positive (sweep flags
something the synthesis then silences).

## Docs integrity

- `[sweep]` **Own-repo URLs point at unmerged paths.** Markdown
  links to
  `https://github.com/cosmic-foundry/cosmic-foundry/{blob,tree,raw}/main/…`
  are validated by `scripts/ci/check_markdown_links.py` against
  the working tree. Grep the diff for that URL shape; for each
  hit, confirm the path resolves to a file that exists *in this
  PR*.
- `[sweep]` **Sphinx `{include}` across directory trees.**
  MyST-NB `{include}` directives fail silently if the included
  file is outside the docs source tree. Grep the diff for
  `{include}` directives; verify each target path exists and is
  reachable from `docs/`.
- `[sweep]` **Index entries missing.** Adding an ADR requires a
  line in `adr/README.md`. Adding a research-notes subsection
  requires a link from `research/index.md`. Adding a roadmap
  epoch requires an entry in `roadmap/index.md`. If the diff
  adds a new file in those dirs, verify the corresponding
  index was also touched.
- `[sweep]` **Cross-references broken by renames/moves.** If
  the PR renames or moves a file, grep the repo for the old
  path. Broken cross-refs will not fail CI if they are written
  as plain markdown links between files in the same tree.

## CI / pre-commit parity

- `[synthesis]` **New check added to CI but not pre-commit (or
  vice versa).** The project treats pre-commit and CI as a
  single gate. A check added to `.pre-commit-config.yaml`
  should also run in the CI workflow, and vice versa. Whether a
  given CI change *should* have a pre-commit counterpart
  requires reading the check's semantics.
- `[sweep]` **GitHub Actions version drift.**
  `scripts/ci/check_action_versions.py` exists to catch
  downgrades. If the PR touches `.github/workflows/`, confirm
  all action versions are pinned to a SHA or a specific tag and
  that none are downgraded relative to `main`.
- `[sweep]` **Skipped hooks.** Grep the diff and commit
  messages for `--no-verify`, `--no-gpg-sign`, `SKIP=<hook>`,
  or disabled pre-commit stages. Any hit is a red flag; the
  synthesis pass decides whether it is justified.

## Bootstrap and cross-PR dependencies

- `[sweep]` **Bootstrap ignores that leave debt for a follow-up
  PR.** Grep the diff for `_ignore`, `xfail`, `skip`, or
  `TODO`/`FIXME` patterns *newly added* by this PR. Any hit
  with a comment like "we'll fix this in the next PR" is an
  automatic flag.
- `[synthesis]` **Dependencies on an unmerged PR.** If this PR
  references a file, symbol, or config that only exists on
  another open branch, the merge order must be explicit. Check
  whether the PR description calls that out. Requires reading
  the diff for what symbols/paths are referenced and checking
  whether they exist on `main`.

## Redundancy with existing scripts / tooling

- `[synthesis]` **Reimplementing a script or hook.** Before
  approving new setup / lint / check logic, grep `scripts/` and
  `.pre-commit-config.yaml` for existing coverage and reason
  about functional overlap. The project has repeatedly tried to
  add a "manual" setup step for something that is already
  automated elsewhere.
- `[sweep]` **Duplicated install/activate steps in docs.** If
  the PR adds a manual `pip install -e` or `pre-commit install`
  step to docs, and those already run inside
  `environment/setup_environment.sh`, the doc change should
  point at the script. Grep the diff for those commands in
  markdown files.

## Project framing and tone

- `[synthesis]` **Tech-stack-first framing.** Top-level
  descriptions (README, `docs/index.md`, ADR motivations,
  roadmap epoch summaries) should lead with the *physics /
  mission* goal, not "built on JAX" or similar. JAX, unyt,
  Zarr, etc. belong in ADRs and deeper docs. Requires reading
  the edited copy as a whole.
- `[synthesis]` **Backwards-compatibility cruft.** The project
  has not cut a stable API. If the PR preserves legacy names,
  adds deprecation warnings, or keeps a shim "just in case",
  push back — the expected mode is structural edits.

## Spelling and copy

- `[sweep]` **British vs. American spelling.** Project standard
  is American English. Codespell enforcement is pending (it
  lands with PR #36); until then, spot-check new copy for
  `colour/color`, `behaviour/behavior`, `organise/organize`,
  `centre/center`, `catalogue/catalog`, `summarise/summarize`,
  `specialise/specialize`. Once codespell is in
  `.pre-commit-config.yaml`, tighten this entry back to
  "enforced via codespell" and leave the spot-check list only
  as a reviewer aide-memoire.

## Architectural-option bias

- `[synthesis]` **Recommendations skipping the simpler
  option.** Per AI.md §*Weighing architectural options*, when
  the PR justifies a choice in an ADR or design note, confirm
  the lower-complexity alternative was named explicitly and
  compared on downstream cost (reviewer load, ops cost,
  reversibility, correctness, blast radius), not on author
  effort. "Richer option chosen because it was easy to type"
  is a specific failure mode here.

## Tests

- `[synthesis]` **New code path without a test.** For
  non-trivial logic, look for a paired test. "Non-trivial"
  here includes: anything with a branch, anything with I/O,
  anything that handles user input, and any kernel/numerics
  change. Requires judging what counts as non-trivial.
- `[synthesis]` **Test asserts the wrong thing.** Watch for
  tests that only assert "no exception raised" or that compare
  a value to itself. A green test suite is not proof of
  correctness if the asserts are tautological.
- `[sweep]` **Golden-data regressions.** If the PR updates a
  fixture or baseline image without a visible explanation in
  the commit message or PR description, flag it. Grep the diff
  for changes under `tests/**/golden/`, `tests/**/baseline*`,
  `tests/**/*.png`, `tests/**/*.npy`. The synthesis pass
  decides whether the explanation, if present, is adequate.

## Commit hygiene

- `[sweep]` **Commit size over ceiling without justification.**
  Code diffs target ~150 LOC with a soft ceiling of ~400. Run
  `git log --numstat` across the PR commits; flag any
  non-docs, non-generated commit over 400 LOC. Docs-only diffs
  are exempt.
- `[synthesis]` **One commit mixing logical changes.**
  "Refactor + bugfix + new feature" in one commit makes bisect
  useless. Requires reading each commit's diff to decide
  whether it's one logical change.
- `[sweep]` **Missing attribution trailer.** AI-agent commits
  must carry a `Co-Authored-By` trailer naming the agent and
  model (ADR-0005). Grep every commit message in the PR for
  `Co-Authored-By:`.
- `[sweep]` **Force-push on an open PR.** Per ADR-0005,
  force-push on an open PR is not allowed without explicit
  user approval. Check the PR timeline via
  `gh pr view <n> --json timelineItems` (or equivalent) for
  `HeadRefForcePushedEvent`. Any hit requires matching approval
  in the discussion.

## Durable metadata leaks

- `[sweep]` **Local filesystem paths in commit messages or PR
  body.** Grep commit messages, the PR title, the PR body, and
  the full diff for `/Users/`, `/home/`, `C:\Users\` patterns.
- `[sweep]` **Secrets.** Scan the diff for anything that looks
  like an API key, token, or password: long base64 strings,
  `sk-…` / `ghp_…` / `AKIA…` prefixes, `password\s*=`,
  `token\s*=`. If one landed, the fix is rotation, not history
  rewriting.

## Scope and minimalism

- `[synthesis]` **Scope creep.** Does every hunk serve the
  PR's stated goal? Drive-by refactors, formatting churn in
  untouched files, and opportunistic renames should be split
  into a follow-up PR unless the PR description calls them out.
- `[synthesis]` **Premature abstraction.** New helpers, base
  classes, or configuration knobs that have exactly one caller
  are a smell — inline them unless the second caller is
  imminent and named.
- `[synthesis]` **Unused code.** Dead functions, unused
  imports, and stubs "for later" should not land. CI catches
  some of this; reviewers catch the rest.
