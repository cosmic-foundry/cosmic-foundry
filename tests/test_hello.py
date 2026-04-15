"""Integration tests for the `cosmic-foundry hello` CLI command."""

from __future__ import annotations

import os
import subprocess
import sys


def _hello_cmd() -> list[str]:
    """Return the command that invokes `cosmic-foundry hello`.

    Uses the same Python interpreter that is running pytest so the test
    works whether the package is installed via the console-script entry
    point or run directly from an editable install.
    """
    return [
        sys.executable,
        "-c",
        "from cosmic_foundry.cli import main; main()",
        "hello",
    ]


def _run_hello(env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        _hello_cmd(),
        capture_output=True,
        text=True,
        env=env,
    )


def test_hello_exit_zero() -> None:
    result = _run_hello()
    assert result.returncode == 0, result.stderr


def test_hello_reports_backend() -> None:
    result = _run_hello()
    assert "JAX backend" in result.stdout


def test_hello_reports_devices() -> None:
    result = _run_hello()
    assert "Local devs" in result.stdout
    assert "Global devs" in result.stdout


def test_hello_jit_smoke_passes() -> None:
    result = _run_hello()
    assert "JIT smoke   : ok" in result.stdout


def test_hello_no_distributed_without_env() -> None:
    """Without JAX_COORDINATOR_ADDRESS, hello must succeed in single-process mode."""
    env = {k: v for k, v in os.environ.items() if k != "JAX_COORDINATOR_ADDRESS"}
    result = _run_hello(env=env)
    assert result.returncode == 0, result.stderr
    assert "JIT smoke   : ok" in result.stdout
