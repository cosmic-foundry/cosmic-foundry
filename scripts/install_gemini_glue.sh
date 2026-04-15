#!/bin/bash
# Install Gemini CLI invocation glue for this repository.
#
# Writes files into `.gemini/` (gitignored). The glue is a thin
# pointer to the in-repo reviewer spec under `pr-review/` — that
# directory is the single source of truth. Tool-specific glue stays
# out of the repo so the project artifact can evolve independently
# of any particular AI tool.
#
# Idempotent: safe to rerun. Overwrites generated files each
# invocation so updates propagate on the next setup.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
GEMINI_DIR="${REPO_ROOT}/.gemini"
SKILLS_DIR="${GEMINI_DIR}/skills"
REVIEWER_SKILL_DIR="${SKILLS_DIR}/pr-reviewer"

mkdir -p "$REVIEWER_SKILL_DIR"

cat >"${REVIEWER_SKILL_DIR}/SKILL.md" <<'EOF'
---
name: pr-reviewer
description: Run the project's adversarial PR reviewer for a pull request. Walks the checklist of historical failure modes and produces a structured report.
---

<!-- GENERATED FILE — edits will be lost on the next
     setup_environment.sh run. Source of truth:
     pr-review/ (tracked) + scripts/install_gemini_glue.sh. -->

# PR Reviewer Skill

This skill allows the Gemini CLI to perform an adversarial PR review for this repository. It follows the protocol, role, and output format defined in `pr-review/agent.md` and uses the checklist in `pr-review/checklist.md`.

## Instructions

1.  **Read Context**: Read `pr-review/agent.md` and `pr-review/checklist.md` in full before starting the review.
2.  **Fetch PR Data**: Use the `gh` tool to fetch PR metadata and diff for the PR number provided by the user:
    ```bash
    gh pr view <pr-number> --repo cosmic-foundry/cosmic-foundry
    gh pr diff <pr-number> --repo cosmic-foundry/cosmic-foundry
    ```
3.  **Perform Review**: Follow the protocol in `pr-review/agent.md` using the fetched data and the working tree (read-only).
4.  **Report**: Produce the report in the exact format specified by `pr-review/agent.md`.

## Usage

You can activate this skill by asking the Gemini CLI to "Review PR <number>" or by manually calling `activate_skill("pr-reviewer")`.
EOF

echo "Installed Gemini CLI glue:"
echo "  ${REVIEWER_SKILL_DIR}/SKILL.md"
