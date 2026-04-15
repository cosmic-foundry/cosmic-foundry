#!/usr/bin/env python3
"""Run install_claude_glue.sh and verify its output is correct.

Running the installer is intentionally in scope: .claude/ is
gitignored and regenerated at setup time, so running it here is
idempotent and has no effect on the git index. The side effect is
desirable — contributors who trigger this hook via pre-commit on a
fresh clone get .claude/ populated without having to run
setup_environment.sh first.

The check fails if:
- pr-review/agent.md or pr-review/checklist.md are missing (the
  installer would produce broken glue pointing at absent files).
- The installer exits non-zero (syntax error, bad heredoc, etc.).
- Any expected output file is absent or has wrong frontmatter fields
  (agent name, model tier). This catches installer edits that change
  the output without updating this check, and vice versa.
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
