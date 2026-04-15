# Helper Scripts

### `start_agent.sh`

Launches a selected AI agent CLI with an automated initialization prompt.

**What it does:**
1. Detects whether it was launched from the `cosmic-foundry` repo root or from
   a parent workspace containing `./cosmic-foundry`.
2. Verifies that the engine repo's `miniforge/` environment exists.
3. Starts a new agent session (`gemini`, `claude`, or `codex`).
4. Passes absolute paths to the core documents an agent must read before
   starting work:
   - `AI.md`
   - The agent-specific document (`GEMINI.md`, `CLAUDE.md`, or `CODEX.md`).
5. When launched from a parent workspace, includes the workspace path and any
   sibling git repositories in the initialization prompt so the agent can
   coordinate paired changes across repositories.
6. Tells the agent to read local instructions before changing a sibling
   repository.

**Single-repo usage:**
```bash
cd /path/to/cosmic-foundry
./scripts/start_agent.sh [gemini|claude|codex]
```

**Multi-repo workspace usage:**
```bash
cd /path/to/parent-workspace
./cosmic-foundry/scripts/start_agent.sh [gemini|claude|codex]
```

Expected layout:
```text
/path/to/parent/workspace/
  cosmic-foundry/
  stellar-foundry/
```

In the multi-repo workflow, reusable engine changes should stay in
`cosmic-foundry`, application-layer changes should stay in the domain repo,
and each repository should get its own branch, worktree, commit, and pull
request.

### `install_claude_glue.sh`

Generates the Claude Code invocation glue for the adversarial PR
reviewer (`.claude/commands/review-pr.md`,
`.claude/agents/pr-reviewer.md`)
from the in-repo reviewer spec at `pr-review/`. Called unconditionally
by `environment/setup_environment.sh`; idempotent, safe to rerun.

`.claude/` is gitignored so the project-artifact layer (`pr-review/`)
stays the single source of truth.

### `review_pr_with_codex.sh`

Runs the same adversarial PR reviewer through Codex:

```bash
./scripts/review_pr_with_codex.sh <pr-number>
```

The wrapper fetches PR metadata and diff with `gh`, then passes that
context plus the tracked `pr-review/` instructions to a read-only
`codex exec` session. Set `COSMIC_FOUNDRY_PR_REPO` to override the
default upstream repository (`cosmic-foundry/cosmic-foundry`) when
testing against another remote.

### `review_pr_with_gemini.sh`

Runs the same adversarial PR reviewer through Gemini CLI:

```bash
./scripts/review_pr_with_gemini.sh <pr-number>
```

The wrapper fetches PR metadata and diff with `gh`, then passes that
context plus the tracked `pr-review/` instructions to a Gemini CLI
session. It uses the `pr-reviewer` skill defined in `.gemini/skills/`.
Set `COSMIC_FOUNDRY_PR_REPO` to override the default upstream
repository (`cosmic-foundry/cosmic-foundry`) when testing against another
remote.
