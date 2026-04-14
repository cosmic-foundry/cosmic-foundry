# ADR-0005 — Branch, PR, and attribution discipline

- **Status:** Accepted
- **Date:** 2026-04-14

## Context

Cosmic Foundry expects contributions from humans and AI coding
agents interchangeably. Several forms of discipline — branch
hygiene, commit size, PR workflow, attribution, and metadata
cleanliness — are the difference between a repository that stays
reviewable and one that becomes unreviewable at even modest scale.
Those rules already appear informally in [`AI.md`](../AI.md) and
were reserved for formalisation in
[`roadmap/epoch-00-bootstrap.md`](../roadmap/epoch-00-bootstrap.md)
§0.6 as ADR-0005.

This ADR codifies the rules and resolves minor drift between AI.md
and §0.6's preview wording (§0.6 mentioned a "single-file
documentation-commit exception"; AI.md already relaxes this to
"larger diffs are fine to documentation" — the permissive
formulation is the one adopted here).

## Decision

Contributions to Cosmic Foundry follow these disciplines. AI.md
remains the informal quick-reference; this ADR is the authoritative
source.

### Fork and branch workflow

- **Fork-based only.** All work happens on a fork of the upstream
  repository. No contributor — human or agent — pushes directly to
  `upstream/main`.
- **Branch from `origin/main`.** Topic branches are created from
  `origin/main` (the fork's main), not from `upstream/main`
  directly. Syncing `origin/main` to `upstream/main` is a separate
  explicit step, performed when the contributor intends it.
- **One branch per PR.** Every change lands via a pull request;
  `upstream/main` has no merge commits authored outside a PR.
- **CI must be green.** The `pre-commit` job is a required status
  check on `upstream/main`. PRs cannot merge red.

### Commit size and scope

- **~100 LOC per code commit.** A commit that touches project code
  should be on the order of 100 lines of diff or smaller. This is
  a guideline, not a hard limit; small overshoots are acceptable
  when a clean split would fragment a single logical change.
- **Documentation diffs may exceed the guideline.** Docs-only
  commits (ADRs, research notes, roadmap edits, README / AI.md /
  similar) are not bound by the ~100-LOC guideline. This supersedes
  §0.6's preliminary "single-file documentation-commit exception"
  phrasing — the exception is by content, not file count.
- **One decision per ADR.** Architectural Decision Records are
  one-decision-per-file and each lands in its own PR (or in a
  tight group when ADRs reference each other, as with ADRs 0001
  – 0003 and 0004 – 0005). Mutating an accepted ADR is prohibited;
  future decisions supersede old ones via a new ADR.

### History and force-push

- **Never force-push a branch with an open PR or merged commits.**
  Force-pushing published history that reviewers or downstream
  branches depend on is prohibited without exception.
- **Pre-PR topic branches are allowed one force-push-with-lease
  only under explicit user approval**, and only for fixes that
  cannot be resolved with a forward commit — e.g. correcting the
  `user.email` on a just-pushed commit, redacting a file that
  should not have been staged. The agent or contributor must ask;
  acting autonomously is not permitted.
- **Never alter merged history.** `git rebase`, `git reset --hard`,
  amending merged commits, and similar operations are prohibited
  on merged branches. Reverts happen through new commits or revert
  PRs.

### Durable metadata and paths

- **No local filesystem paths in durable metadata.** Commit
  messages, PR titles, PR descriptions, and ADR text must not
  include absolute paths from a contributor's workstation (e.g.
  `/Users/…`, `/home/…`, `C:\Users\…`). Use repository-relative
  paths or generic tool commands instead. Private workstation
  details do not belong in the public history.
- **No API keys, tokens, or credentials.** These never appear in
  the repository, not even in example snippets. If one is pasted
  by accident, it is rotated immediately rather than removed via
  history rewrite (rewriting history does not un-leak a secret
  that has already been pushed).

### Attribution

- **AI-agent commits carry a `Co-Authored-By` trailer** naming the
  agent and model. The author of record is the human user driving
  the session; the `Co-Authored-By` trailer acknowledges the
  agent's contribution without claiming human authorship.
- **PR descriptions disclose AI-agent involvement** when an agent
  generated substantial content. A trailing line such as "Generated
  with Claude Code" is sufficient; the exact formulation is per
  the agent tool, not this ADR.
- **Upstream references use relative paths.** Cross-references
  between docs, between ADRs, and from code comments use
  repository-relative paths, not URLs to specific upstream or
  origin branches that may move.

## Consequences

- **Positive.** A reviewer can land any PR confident that the
  commit is small enough to read, the history is linear and un-
  rewritten, durable metadata is clean, and attribution is honest.
  Mixing human and AI-agent contributions does not degrade review
  quality because the discipline is symmetric.
- **Negative.** Author-identity or trailer mistakes are more
  expensive to fix than in repositories with a relaxed force-push
  policy — each fix requires an explicit approval turn. In
  practice this occurs rarely, and the audit-trail benefit exceeds
  the cost. Contributors new to the fork-based flow incur a one-
  time learning cost; CONTRIBUTING.md (landing in Epoch 0 §0.8)
  is the rampdown path.
- **Neutral.** AI.md's informal summary is kept in sync with this
  ADR. When the two disagree, this ADR is authoritative. AI.md may
  be edited to track rewordings in this ADR without a new ADR; the
  ADR is edited only by supersession.

## Alternatives considered

- **Rebase-and-force-push workflow.** Common in other projects;
  keeps linear history per branch at the cost of making reviewer
  and agent coordination harder (a reviewer's comment can reference
  a commit SHA that disappears). Rejected because the repository
  optimises for reviewability over branch-level tidiness, and
  squash-merge at PR close already produces linear `main` history.
- **Trunk-based development with direct commits to `upstream/main`.**
  Simpler for solo work; incompatible with the branch-protection
  gate and PR-required workflow that CI depends on.
- **Silent AI-agent authorship.** Some projects do not disclose
  agent involvement. Rejected because concealed authorship
  degrades the trust surface that downstream reviewers and
  citation-based replication (ADR-0007) depend on.
- **Hard LOC cap on every commit, including docs.** A cap simple
  enough to mechanise (e.g. "no commit exceeds 100 LOC, period")
  would push docs-heavy ADR or roadmap PRs into many small fragments
  that reviewers cannot follow as a coherent argument. Rejected in
  favour of the content-based exception above.

## Cross-references

- [`AI.md`](../AI.md) — informal quick-reference; kept aligned with
  this ADR and may be edited without a new ADR to match rewordings
  here.
- [`roadmap/epoch-00-bootstrap.md`](../roadmap/epoch-00-bootstrap.md)
  §0.6 (reservation), §0.8 (CONTRIBUTING.md ramp-down).
- ADR-0007 (Replication workflow) — depends on attribution honesty
  for the citation-backed verification trail.
