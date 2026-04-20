# Agent Instructions

These guidelines apply to all AI agents working on this repository,
regardless of platform (Claude Code, Codex, Gemini, or others).

For development workflow rules that apply to all contributors (branch
discipline, commit size, environment setup, roadmap management,
physics lanes), see [`DEVELOPMENT.md`](DEVELOPMENT.md). This
file covers only behavior that is specific to AI agents.

---

## Platform role

Cosmic Foundry is the **organizational platform** for the simulation
ecosystem. Application repositories — covering stellar physics,
cosmology, galactic dynamics, planetary formation, and other domains
— build on top of it. The platform/application split is documented in
`ARCHITECTURE.md §Platform and application split`.

In practice this means:

- Reusable computation infrastructure (kernels, mesh, fields, I/O,
  diagnostics) and manifest infrastructure (`cosmic_foundry.manifests`:
  HTTP client, `ValidationAdapter`, `Provenance`, base schemas) belong
  here.
- Domain-specific physics implementations and observational validation
  data belong in the relevant application repo.
- If a task spans the platform and an application repo, use separate
  branches and pull requests for each repository. Keep the platform
  change minimal and self-contained; the application repo change
  depends on it.
- Cross-scale workflows that compose two or more application domains
  belong in their own repository, not here.
- **This file is the authoritative source for all AI-agent-specific
  behavior across the organization.** `DEVELOPMENT.md` is the
  authoritative source for workflow rules shared by humans and agents.
  Application repo `AI.md` files are intentionally thin: they delegate
  here for agent behavior and to `DEVELOPMENT.md` for workflow rules,
  adding only what genuinely differs for that repo.

---

## Session startup

**At the start of every session**, in this order:

1. **Run the health check:**
   ```bash
   ./scripts/agent_health_check.sh
   ```
   The script verifies that (a) the `cosmic_foundry` conda environment
   is active, (b) `pre_commit` is importable, and (c) the git
   pre-commit hook is installed.

   **If the env check fails** (script prints `✗ WRONG ENVIRONMENT` and
   exits non-zero), stop immediately and warn the user:

   > ⚠️ The `cosmic_foundry` conda environment is not active. All
   > Python commands in this repo (`python`, `pytest`, `mypy`,
   > `pre-commit`, `sphinx-build`) must run inside this environment.
   > Using the wrong environment causes silent misconfiguration errors.
   >
   > The correct way to start an agent session is:
   > ```bash
   > ./scripts/start_agent.sh claude   # or gemini / codex
   > ```
   > `start_agent.sh` activates the environment automatically before
   > launching the agent. Do not proceed until the user confirms the
   > session has been restarted this way, or manually activates the
   > env:
   > ```bash
   > source scripts/activate_environment.sh
   > ```
   > then re-launches the agent from that shell.

   **If the env check passes but either follow-up check fails**, re-run
   `setup_environment.sh` or the remediation commands printed by the
   script.

2. **Read `STATUS.md`** — current planned work and navigation anchor.

3. **Read `ARCHITECTURE.md`** — all live architectural decisions. When
   work touches a topic documented there, read it before making changes.

---

## In-session PR review

Inside an active session, treat user requests of the form "Review PR N"
or "Review N" as a request to run the adversarial reviewer:

1. Read `pr-review/agent.md` and `pr-review/checklist.md` in full.
   If the PR is architecture-changing, also read
   `pr-review/architecture-checklist.md`.
2. Fetch PR metadata and diff:
   ```bash
   gh pr view N --repo cosmic-foundry/cosmic-foundry
   gh pr diff N --repo cosmic-foundry/cosmic-foundry
   ```
3. Perform the review using the fetched data and the working tree
   (read-only inspection only).
4. Return the report in the exact format required by
   `pr-review/agent.md`.

---

## Physics lane selection

The three lanes (A, B, C) are defined in
[`DEVELOPMENT.md §Physics capability implementation paths`](DEVELOPMENT.md#physics-capability-implementation-paths).

For any task that touches a physics capability:

1. **Classify the lane.** Look up the reference code's license in
   `docs/research/06-12-licensing.md`. If no reference code exists, or
   the user's framing implies generalization or novel work (phrases
   like "extend," "generalize," "we might need to break new ground,"
   "give us our own version of X"), Lane C applies.
2. **Propose the derivation-first lane when it appears to apply.**
   If the default would be Lane A but the task framing suggests Lane C,
   propose Lane C to the user before writing code. If the reference is
   copyleft, Lane B is not a proposal — state it as the required lane
   and confirm the user agrees before proceeding.
3. **Record the lane in the PR description** in the first paragraph,
   e.g. `Lane C (origination). Reference papers: [...]`. For Lane B,
   explicitly record that reference source was not consulted.
4. **When uncertain, propose the derivation-first lane (B or C,
   whichever fits) and ask the user to confirm** rather than defaulting
   silently to Lane A.

The lane choice is the user's decision; the agent's job is to surface
the decision transparently, not to make it silently.

---

## Weighing architectural options

You are an AI agent. Writing code and prose costs you nothing. This
means implementation effort is not a meaningful criterion when
comparing architectural options — it is a rounding error, not a
trade-off. The costs that actually matter are all downstream:

- reviewer cognitive load,
- ongoing operational and maintenance burden,
- reversibility if the choice turns out to be wrong,
- correctness and safety guarantees,
- blast radius of a failure.

Rank options by these. Include the simpler option in every comparison
even when you intend to recommend the richer one — the user needs the
full option space to make an informed decision.

---

## Code Quality

- Write code that is:
  - Self-documenting (clear variable names, simple logic)
  - Minimal (only what's needed, no over-engineering)
  - Testable (existing tests must pass, new features need tests)
