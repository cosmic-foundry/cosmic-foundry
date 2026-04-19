"""Shared codegen utilities for kernel modules with _derive()/generate() pattern.

Each kernel module (e.g., laplacian.py) can implement:
- _derive() -> dict — runs SymPy, returns constants and stencil structure
- generate() -> str — produces the full generated block for splicing

The generator script (scripts/generate_kernels.py) discovers all such modules
and splices their output via these utilities. The hash verifies that the
committed block matches the derivation (catches hand-edits).
"""

from __future__ import annotations

import hashlib

SENTINEL_BEGIN = (
    "# BEGIN GENERATED — do not edit; regenerate with scripts/generate_kernels.py\n"
)
SENTINEL_END = "# END GENERATED\n"


def make_hash(constants: dict[str, int]) -> str:
    """SHA-256 hash of the canonical constants string (first 16 hex chars).

    Parameters
    ----------
    constants
        dict with int values to hash (e.g., {"CENTER_WEIGHT": -6, ...})

    Returns
    -------
    str
        First 16 hex characters of SHA-256(canonical_string)

    Notes
    -----
    The canonical string is "KEY=val;KEY=val;..." with keys sorted for
    determinism, so the hash is order-independent.
    """
    canonical = ";".join(f"{k}={v}" for k, v in sorted(constants.items()))
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def extract_block(source: str) -> str:
    """Extract the content between BEGIN GENERATED and END GENERATED sentinels.

    Parameters
    ----------
    source
        The full source code as a string

    Returns
    -------
    str
        Content between the sentinels (exclusive of the sentinel lines themselves)
    """
    start = source.index(SENTINEL_BEGIN) + len(SENTINEL_BEGIN)
    end = source.index(SENTINEL_END)
    return source[start:end]


def splice(source: str, block: str) -> str:
    """Return *source* with the generated block replaced by *block*.

    Parameters
    ----------
    source
        The full source code as a string
    block
        The new content to insert between the sentinels

    Returns
    -------
    str
        Source with the generated block replaced (sentinel lines preserved)

    Notes
    -----
    The BEGIN GENERATED and END GENERATED sentinel lines are preserved;
    only the content between them is replaced.
    """
    start = source.index(SENTINEL_BEGIN) + len(SENTINEL_BEGIN)
    end = source.index(SENTINEL_END)
    return source[:start] + block + source[end:]
