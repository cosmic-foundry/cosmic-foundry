---
description: Run the adversarial PR reviewer against a pull request.
argument-hint: <pr-number>
---

<!-- GENERATED FILE — edits will be lost on the next
     setup_environment.sh run. Source of truth:
     pr-review/ (tracked) + scripts/install_claude_glue.sh. -->

You are orchestrating the project's adversarial PR review. The
reviewer's role, protocol, and output format are defined in
`pr-review/agent.md`. The checklist of historical failure modes
is in `pr-review/checklist.md`.

Perform these steps:

1. If `$ARGUMENTS` is empty, ask the user for the PR number
   before proceeding.
2. Fetch PR metadata and diff:
   ```bash
   gh pr view $ARGUMENTS --repo cosmic-foundry/cosmic-foundry
   gh pr diff $ARGUMENTS --repo cosmic-foundry/cosmic-foundry
   ```
3. Spawn the `pr-reviewer` subagent (Sonnet). Pass the PR
   number, the fetched metadata, and the diff as context.
   The subagent uses only Read/Grep/Glob for working-tree
   inspection — do not ask it to run shell commands.
4. Return the subagent's report verbatim. Do not summarize,
   soften, or re-order findings. Surface the "Confidence and
   blind spots" section prominently.

If the subagent returns a malformed output, say so explicitly
rather than papering over it.
