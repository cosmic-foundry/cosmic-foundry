# Helper Scripts

### `start_agent.sh`

Launches a selected AI agent CLI with an automated initialization prompt.

**What it does:**
1. Detects whether it was launched from the `cosmic-foundry` repo root, from
   a parent workspace containing `./cosmic-foundry`, or from a sibling
   repository next to `cosmic-foundry`.
2. Verifies that the engine repo's `environment/miniforge/` environment exists.
3. Starts a new agent session (`gemini`, `claude`, or `codex`).
4. Passes absolute paths to the core document an agent must read before
   starting work: `AI.md`
5. When launched from a parent workspace or sibling repository, includes the
   workspace path and any sibling git repositories in the initialization
   prompt so the agent can coordinate paired changes across repositories.
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

**Sibling-repo usage:**
```bash
cd /path/to/parent-workspace/<sibling-repo>
../cosmic-foundry/scripts/start_agent.sh [gemini|claude|codex]
```

When launched this way, the initialization prompt tells the agent to treat the
current sibling repository as the primary working repository while still using
the `cosmic-foundry` environment and shared project rules.

Expected layout:
```text
/path/to/parent/workspace/
  cosmic-foundry/
  <application-repo>/
```

In the multi-repo workflow, reusable engine changes should stay in
`cosmic-foundry`, application-layer changes should stay in the domain repo,
and each repository should get its own branch, worktree, commit, and pull
request.
