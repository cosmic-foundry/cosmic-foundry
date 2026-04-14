"""Entry points for the cosmic-foundry command-line interface."""

from __future__ import annotations

import os
import sys

import click

from cosmic_foundry._version import __version__


@click.group()
@click.version_option(__version__, prog_name="cosmic-foundry")
def main() -> None:
    """Cosmic Foundry — high-performance computational astrophysics engine."""


@main.command()
def hello() -> None:
    """Report the JAX environment and run a smoke test.

    In single-process mode (no JAX_COORDINATOR_ADDRESS set), reports
    local devices and confirms the JIT path is functional.

    In distributed mode (JAX_COORDINATOR_ADDRESS set), initialises
    jax.distributed and additionally reports process_index and
    process_count.
    """
    try:
        import jax
    except ImportError:
        click.echo("ERROR: JAX is not installed.", err=True)
        sys.exit(1)

    distributed = bool(os.environ.get("JAX_COORDINATOR_ADDRESS"))

    if distributed:
        try:
            jax.distributed.initialize()
        except Exception as exc:
            click.echo(
                f"ERROR: jax.distributed.initialize() failed: {exc}\n"
                "Check that JAX_COORDINATOR_ADDRESS, JAX_NUM_PROCESSES, "
                "and JAX_PROCESS_ID are set correctly.",
                err=True,
            )
            sys.exit(1)

    process_index: int = jax.process_index()
    process_count: int = jax.process_count()
    local_devices = jax.local_devices()
    global_devices = jax.devices()

    if process_index == 0:
        click.echo(f"cosmic-foundry {__version__}")
        click.echo(f"JAX backend : {jax.default_backend()}")
        click.echo(f"Processes  : {process_index + 1}/{process_count}")
        click.echo(f"Local devs  : {[str(d) for d in local_devices]}")
        click.echo(f"Global devs : {[str(d) for d in global_devices]}")

        # JIT smoke test: 32³ Laplacian on a small array.
        try:
            _laplacian_smoke_test()
            click.echo("JIT smoke   : ok")
        except Exception as exc:
            click.echo(f"ERROR: JIT smoke test failed: {exc}", err=True)
            sys.exit(1)


def _laplacian_smoke_test() -> None:
    """Run a trivial JIT-compiled Laplacian to confirm the XLA path works."""
    import jax
    import jax.numpy as jnp

    n = 32

    @jax.jit
    def laplacian(u: jax.Array) -> jax.Array:
        return (
            jnp.roll(u, 1, 0)
            + jnp.roll(u, -1, 0)
            + jnp.roll(u, 1, 1)
            + jnp.roll(u, -1, 1)
            + jnp.roll(u, 1, 2)
            + jnp.roll(u, -1, 2)
            - 6 * u
        )

    u = jnp.ones((n, n, n))
    result = laplacian(u)
    # All interior values should be zero for a constant field.
    assert result.shape == (n, n, n)
