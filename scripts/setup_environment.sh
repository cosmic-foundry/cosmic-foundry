#!/bin/bash
set -e

# Download and set up Miniforge with the cosmic_foundry conda environment

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
MINIFORGE_DIR="${REPO_ROOT}/environment/miniforge"
ENV_FILE="${REPO_ROOT}/environment/cosmic_foundry.yml"
VERSIONS_FILE="${REPO_ROOT}/environment/versions.yml"
MINIFORGE_INSTALLER="${REPO_ROOT}/environment/miniforge-installer.sh"

# Parse pinned Miniforge version from YAML, or resolve "latest" from GitHub.
MINIFORGE_VERSION=$(grep -A 1 "miniforge:" "$VERSIONS_FILE" | grep "version:" | awk '{print $2}' | tr -d '"')

if [ "$MINIFORGE_VERSION" = "latest" ]; then
  echo "Resolving latest Miniforge version from GitHub..."
  MINIFORGE_VERSION=$(curl -fsSL "https://api.github.com/repos/conda-forge/miniforge/releases/latest" \
    | grep '"tag_name"' | head -1 | cut -d'"' -f4)
  if [ -z "$MINIFORGE_VERSION" ]; then
    echo "Error: Could not resolve latest Miniforge version from GitHub API."
    exit 1
  fi
fi

echo "Miniforge version: $MINIFORGE_VERSION"

# Detect platform
OS=$(uname -s)
ARCH=$(uname -m)

if [ "$OS" = "Linux" ]; then
  if [ "$ARCH" = "x86_64" ]; then
    MINIFORGE_FILENAME="Miniforge3-Linux-x86_64.sh"
  elif [ "$ARCH" = "aarch64" ]; then
    MINIFORGE_FILENAME="Miniforge3-Linux-aarch64.sh"
  else
    echo "Error: Unsupported Linux architecture: $ARCH"
    exit 1
  fi
elif [ "$OS" = "Darwin" ]; then
  if [ "$ARCH" = "x86_64" ]; then
    MINIFORGE_FILENAME="Miniforge3-MacOSX-x86_64.sh"
  elif [ "$ARCH" = "arm64" ]; then
    MINIFORGE_FILENAME="Miniforge3-MacOSX-arm64.sh"
  else
    echo "Error: Unsupported macOS architecture: $ARCH"
    exit 1
  fi
else
  echo "Error: Unsupported OS: $OS"
  exit 1
fi

MINIFORGE_URL="https://github.com/conda-forge/miniforge/releases/download/${MINIFORGE_VERSION}/${MINIFORGE_FILENAME}"

# On macOS, libmamba may fail to detect the OS version via sw_vers in some
# shell environments, which causes a codesigning error during installation.
# Export CONDA_OVERRIDE_OSX before running the installer so it takes effect.
if [ "$OS" = "Darwin" ]; then
  MACOS_VERSION=$(sw_vers -productVersion 2>/dev/null \
    || defaults read loginwindow SystemVersionStampAsString 2>/dev/null \
    || echo "15.0")
  export CONDA_OVERRIDE_OSX="$MACOS_VERSION"
  echo "macOS version override: $CONDA_OVERRIDE_OSX"
fi

# Download Miniforge if not present
if [ ! -d "$MINIFORGE_DIR" ]; then
  echo "Downloading Miniforge $MINIFORGE_VERSION for $OS $ARCH..."
  curl -L -o "$MINIFORGE_INSTALLER" "$MINIFORGE_URL"
  chmod +x "$MINIFORGE_INSTALLER"
  echo "Installing Miniforge to $MINIFORGE_DIR..."
  # Allow a non-zero exit: on macOS arm64 with pre-release OS versions,
  # libmamba may fail the post-install codesign step even though all files
  # are correctly extracted. Conda-forge packages come pre-signed from the
  # channel, so this step is not required for the environment to work.
  "$MINIFORGE_INSTALLER" -b -p "$MINIFORGE_DIR" || true
  rm -f "$MINIFORGE_INSTALLER"
  if [ ! -f "${MINIFORGE_DIR}/etc/profile.d/conda.sh" ]; then
    echo "Error: Miniforge installer did not produce a working base environment."
    echo "Check the output above for fatal errors."
    exit 1
  fi
else
  echo "Miniforge already present at $MINIFORGE_DIR"
fi

# Source Miniforge
echo "Activating Miniforge..."
source "${MINIFORGE_DIR}/etc/profile.d/conda.sh"

# Keep conda itself current before touching any environments.
# conda lives in the base env, not in cosmic_foundry.
echo "Updating conda..."
conda update -n base conda --yes || true

# Create or update the conda environment from YAML.
# Using update --prune on an existing env avoids the "already exists" error
# from create and removes packages that have been dropped from the spec.
if conda env list | grep -qE "^cosmic_foundry[[:space:]]"; then
  echo "Updating existing cosmic_foundry environment..."
  conda env update -f "$ENV_FILE" --prune
else
  echo "Creating cosmic_foundry environment..."
  conda env create -f "$ENV_FILE" --yes
fi

conda activate cosmic_foundry

# Install the cosmic_foundry package in editable mode with dev and docs
# extras. Editable install lets source changes take effect without
# reinstalling; the [dev,docs] extras are the superset CI uses, so local
# setup matches CI by default. After a git pull that changes extras or
# entry points, re-run 'pip install -e .[dev,docs]' inside the activated
# env — no need to re-run this whole script.
echo "Installing cosmic_foundry package (editable, dev+docs extras)..."
pip install -e ".[dev,docs]"

# Install agent CLIs via the conda env's npm so the installs land in the
# conda prefix — no system-wide writes, no admin privileges required.
echo "Installing agent CLIs (Claude, Codex, Gemini)..."
npm install -g @anthropic-ai/claude-code
npm install -g @openai/codex
npm install -g @google/gemini-cli

# Install pre-commit git hook so local commits run the same checks as CI.
# pre-commit refuses to install while core.hooksPath is set. Clear any
# stale local override (e.g. inherited from a git init template) so the
# hook can land in the default .git/hooks/ location. If a global or
# system value is still in effect, surface it instead of silently
# overriding the user's dotfiles.
echo "Installing pre-commit hooks..."
if [ -n "$(git -C "$REPO_ROOT" config --local --get core.hooksPath 2>/dev/null)" ]; then
  echo "  Clearing stale local core.hooksPath override"
  git -C "$REPO_ROOT" config --local --unset-all core.hooksPath
fi
remaining_hooks_path=$(git -C "$REPO_ROOT" config --get core.hooksPath 2>/dev/null || true)
if [ -n "$remaining_hooks_path" ]; then
  echo "Error: core.hooksPath is set to '$remaining_hooks_path' outside this repo's local config." >&2
  echo "       pre-commit refuses to install while it is set. Unset it (for example," >&2
  echo "       'git config --global --unset core.hooksPath') and re-run this script." >&2
  exit 1
fi
pre-commit install

# Install gitleaks for local secret scanning.
# On macOS, use Homebrew (the only supported binary distribution).
# On Linux in CI, gitleaks is provided by gitleaks/gitleaks-action; developers
# on Linux should install it manually: https://github.com/gitleaks/gitleaks
echo "Installing gitleaks..."
if [ "$OS" = "Darwin" ]; then
  if command -v brew &>/dev/null; then
    brew install gitleaks 2>/dev/null || brew upgrade gitleaks 2>/dev/null || true
    if command -v gitleaks &>/dev/null; then
      echo "  gitleaks $(gitleaks version) installed"
    else
      echo "  Warning: gitleaks install via Homebrew failed — install manually with 'brew install gitleaks'"
    fi
  else
    echo "  Warning: Homebrew not found — install gitleaks manually: https://github.com/gitleaks/gitleaks"
  fi
else
  echo "  Skipping gitleaks install on $OS — install manually if needed: https://github.com/gitleaks/gitleaks"
fi

# Configure a global gitignore to catch machine-specific and credential files
# across every repository on this machine, not just this one.
GLOBAL_GITIGNORE="${HOME}/.config/git/ignore"
if [ -z "$(git config --global core.excludesfile 2>/dev/null)" ]; then
  echo "Configuring global gitignore at ${GLOBAL_GITIGNORE}..."
  mkdir -p "${HOME}/.config/git"
  if [ ! -f "$GLOBAL_GITIGNORE" ]; then
    cat > "$GLOBAL_GITIGNORE" << 'GITIGNORE_EOF'
# Global gitignore — machine-specific files that should never be committed
# in any repository on this machine.

# macOS
.DS_Store
.AppleDouble
.LSOverride
._*
.Spotlight-V100
.Trashes
Icon?

# Credential files / private keys
*.pem
*.key
*.p12
*.pfx
.netrc
.npmrc-private
.pypirc

# Local environment overrides
.envrc.local
*.local
.env.local
.env.*.local

# Editor artefacts
*.swp
*~
.vscode/*.log
*.sublime-workspace

# Cloud provider config (if ever checked out into a project tree)
.aws/
.azure/
.gcloud/
GITIGNORE_EOF
  fi
  git config --global core.excludesfile "$GLOBAL_GITIGNORE"
  echo "  Global gitignore configured: ${GLOBAL_GITIGNORE}"
else
  echo "  Global gitignore already configured: $(git config --global core.excludesfile)"
fi

# Record that the environment is in sync with the current spec files.
# start_agent.sh compares this sentinel against the mtimes of
# cosmic_foundry.yml and setup_environment.sh to decide whether a
# sync is needed before launching an agent session.
touch "${MINIFORGE_DIR}/envs/cosmic_foundry/.env_last_updated"

conda deactivate

echo "Setup complete"
