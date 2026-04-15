#!/usr/bin/env python3
"""Verify that scripts/install_claude_glue.sh produces the expected files.

The installer writes Claude Code invocation glue under .claude/ at
setup time. That directory is gitignored, so without this gate a
breaking edit to the installer — bad heredoc, renamed pr-review/
path, stray syntax error — would only surface on the next fresh
clone's setup_environment.sh run.

This hook runs the installer and asserts the generated files exist
with the expected frontmatter fields (agent name, model tier).
It also verifies the in-repo files the glue points at still exist,
so renaming pr-review/agent.md or pr-review/checklist.md without
updating the installer fails loudly.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
INSTALLER = REPO_ROOT / "scripts" / "install_claude_glue.sh"
CLAUDE_DIR = REPO_ROOT / ".claude"

EXPECTED_FILES: dict[Path, dict[str, str]] = {
    CLAUDE_DIR / "commands" / "review-pr.md": {
        "description": r".+",
        "argument-hint": r".+",
    },
    CLAUDE_DIR / "agents" / "pr-reviewer-sweep.md": {
        "name": r"pr-reviewer-sweep",
        "model": r"haiku",
    },
    CLAUDE_DIR / "agents" / "pr-reviewer.md": {
        "name": r"pr-reviewer",
        "model": r"sonnet",
    },
}

REFERENCED_REPO_FILES = [
    REPO_ROOT / "pr-review" / "agent.md",
    REPO_ROOT / "pr-review" / "checklist.md",
]

FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---", re.DOTALL)


def _frontmatter_field(text: str, key: str) -> str | None:
    m = FRONTMATTER_RE.match(text)
    if not m:
        return None
    for line in m.group(1).splitlines():
        if ":" not in line:
            continue
        k, _, v = line.partition(":")
        if k.strip() == key:
            return v.strip()
    return None


def main() -> int:
    errors: list[str] = []

    for path in REFERENCED_REPO_FILES:
        if not path.exists():
            errors.append(
                f"installer references missing file: "
                f"{path.relative_to(REPO_ROOT)}"
            )
    if errors:
        for e in errors:
            print(e, file=sys.stderr)
        return 1

    result = subprocess.run(
        ["bash", str(INSTALLER)],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    if result.returncode != 0:
        print("install_claude_glue.sh failed:", file=sys.stderr)
        print(result.stdout, file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        return 1

    for path, required in EXPECTED_FILES.items():
        rel = path.relative_to(REPO_ROOT)
        if not path.exists():
            errors.append(f"installer did not produce: {rel}")
            continue
        text = path.read_text()
        for key, value_re in required.items():
            got = _frontmatter_field(text, key)
            if got is None:
                errors.append(f"{rel}: missing frontmatter field '{key}'")
            elif not re.fullmatch(value_re, got):
                errors.append(
                    f"{rel}: frontmatter '{key}' is {got!r}, "
                    f"expected /{value_re}/"
                )

    if errors:
        for e in errors:
            print(e, file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
