#!/usr/bin/env python3
"""Detect version downgrades of GitHub Actions in .github/workflows/*.yml.

Dependabot proposes `uses: owner/action@vN` version bumps; a careless
full-file rewrite can silently regress them. This hook compares every
`uses:` pin in the working tree against the same pin on `origin/main`
and fails if a parseable version has moved backwards.

Escape hatch: an intentional downgrade must carry a
`# allow-downgrade: <reason>` comment either on the same line as the
`uses:` entry or on the line immediately above it.

Skipped silently: SHA pins, branch refs, and any pin where either side
does not parse as a version.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

from packaging.version import InvalidVersion, Version

USES_RE = re.compile(
    r"^(?P<indent>\s*)(?:-\s*)?uses:\s*(?P<action>[^\s@#]+)@(?P<ref>[^\s#]+)\s*(?P<trailing>#.*)?$"
)
ALLOW_RE = re.compile(r"#\s*allow-downgrade\s*:\s*\S")
SHA_RE = re.compile(r"^[0-9a-f]{40}$")
BASE_REF = "origin/main"


def _parse_uses(line: str) -> tuple[str, str, str] | None:
    """Return (action, ref, trailing_comment) for a uses: line, else None."""
    m = USES_RE.match(line)
    if not m:
        return None
    return m.group("action"), m.group("ref"), m.group("trailing") or ""


def _as_version(ref: str) -> Version | None:
    """Return Version if ref looks like a parseable version, else None.

    Branch names and SHAs return None so they're skipped from comparison.
    """
    if SHA_RE.match(ref):
        return None
    try:
        return Version(ref)
    except InvalidVersion:
        return None


def _git_show(path: Path, ref: str = BASE_REF) -> str | None:
    """Return the file contents at `ref`, or None if the file doesn't exist there."""
    result = subprocess.run(
        ["git", "show", f"{ref}:{path.as_posix()}"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    return result.stdout


def _has_allow_downgrade(lines: list[str], idx: int) -> bool:
    """Check for `# allow-downgrade: <reason>` on the uses: line or line above."""
    if ALLOW_RE.search(lines[idx]):
        return True
    if idx > 0 and ALLOW_RE.search(lines[idx - 1]):
        return True
    return False


def _base_pins(base_text: str) -> dict[str, str]:
    """Map action -> ref for every uses: line in the base file."""
    pins: dict[str, str] = {}
    for line in base_text.splitlines():
        parsed = _parse_uses(line)
        if parsed is None:
            continue
        action, ref, _ = parsed
        pins[action] = ref
    return pins


def _check_file(path: Path) -> list[str]:
    errors: list[str] = []
    current_text = path.read_text(encoding="utf-8")
    base_text = _git_show(path)
    if base_text is None:
        return errors
    base = _base_pins(base_text)

    lines = current_text.splitlines()
    for idx, line in enumerate(lines):
        parsed = _parse_uses(line)
        if parsed is None:
            continue
        action, new_ref, _ = parsed
        old_ref = base.get(action)
        if old_ref is None or old_ref == new_ref:
            continue
        old_v = _as_version(old_ref)
        new_v = _as_version(new_ref)
        if old_v is None or new_v is None:
            continue
        if new_v >= old_v:
            continue
        if _has_allow_downgrade(lines, idx):
            continue
        errors.append(
            f"{path}:{idx + 1}: {action} downgraded "
            f"{old_ref} -> {new_ref} (origin/main has {old_ref}). "
            "Add `# allow-downgrade: <reason>` on this line or the line above "
            "if intentional."
        )
    return errors


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    workflow_dir = repo_root / ".github" / "workflows"
    if not workflow_dir.is_dir():
        return 0

    # Verify the base ref is available — CI shallow clones need `git fetch
    # origin main` before this hook runs.
    probe = subprocess.run(
        ["git", "rev-parse", "--verify", BASE_REF],
        capture_output=True,
        text=True,
    )
    if probe.returncode != 0:
        print(
            f"check_action_versions: cannot resolve {BASE_REF}. "
            "Run `git fetch origin main` and retry.",
            file=sys.stderr,
        )
        return 1

    errors: list[str] = []
    for path in sorted(workflow_dir.glob("*.yml")):
        errors.extend(_check_file(path.relative_to(repo_root)))
    for path in sorted(workflow_dir.glob("*.yaml")):
        errors.extend(_check_file(path.relative_to(repo_root)))

    for err in errors:
        print(err, file=sys.stderr)
    if errors:
        return 1
    print("No GitHub Actions version downgrades detected.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
