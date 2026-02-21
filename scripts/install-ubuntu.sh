#!/usr/bin/env bash
set -euo pipefail

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "This installer supports Linux (Ubuntu) only."
  exit 1
fi

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 is required (3.9+)."
  exit 1
fi

if ! command -v pipx >/dev/null 2>&1; then
  echo "pipx not found. Installing with apt..."
  sudo apt-get update
  sudo apt-get install -y pipx
  pipx ensurepath || true
fi

SRC="${1:-.}"
echo "Installing aishell from: ${SRC}"
pipx install --force "${SRC}"

# Improve GUI capture support (best effort)
if command -v apt-get >/dev/null 2>&1; then
  sudo apt-get update
  sudo apt-get install -y gnome-screenshot || true
fi

echo "Done. Run: aishell  or  aishell-gui"
echo "Then configure providers with: /setup and /auth"
echo "Optional: install Ollama for local/offline DeepSeek with: /ollama install"
