# Getting started

## Prerequisites

- Linux (Ubuntu 22.04+ or equivalent)
- [miniforge](https://github.com/conda-forge/miniforge) (the repo ships a
  setup script that installs it for you)

## Installation

```bash
# 1. Clone the fork and enter the repo
git clone git@github.com:<your-fork>/cosmic-foundry.git
cd cosmic-foundry

# 2. Install miniforge, create the conda env, install the package
#    in editable mode, and register the pre-commit hook (~5 min, one-time)
bash environment/setup_environment.sh

# 3. Activate the environment (required at the start of every session)
source environment/activate_environment.sh
```

### Keeping the env current

After a `git pull` that changes `pyproject.toml` extras or entry points,
refresh the editable install from inside the activated environment:

```bash
pip install -e .[dev,docs]
```

No need to re-run `setup_environment.sh` for this — it's intended as a
one-time bootstrap.

## Verify the installation

```bash
# Run the test suite
pytest

# Run the hello smoke test
cosmic-foundry hello
```

A successful `hello` prints the JAX backend, device list, and confirms the
JIT path:

```
cosmic-foundry 0.1.0.dev0
JAX backend : cpu
Processes   : 1/1
Local devs  : ['CpuDevice(id=0)']
Global devs : ['CpuDevice(id=0)']
JIT smoke   : ok
```

## Running in distributed mode

`hello` detects distributed mode from the `JAX_COORDINATOR_ADDRESS`
environment variable. Set `JAX_COORDINATOR_ADDRESS`, `JAX_NUM_PROCESSES`,
and `JAX_PROCESS_ID` before launching to enable `jax.distributed`. See
[ADR-0003](https://github.com/cosmic-foundry/cosmic-foundry/blob/main/adr/ADR-0003-jax-distributed-host-parallelism.md)
for the design rationale.
