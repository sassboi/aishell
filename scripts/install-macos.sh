#!/usr/bin/env bash
set -euo pipefail

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "This installer supports macOS only."
  exit 1
fi

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 is required. Install Python 3.9+ and retry."
  exit 1
fi

SRC="${1:-.}"

if command -v pipx >/dev/null 2>&1; then
  echo "Installing aishell with pipx from: ${SRC}"
  pipx install --force "${SRC}"
  echo "Done. Run: aishell"
  echo "Then configure providers with: /setup and /auth"
  echo "Optional: install Ollama for local/offline DeepSeek with: /ollama install"
  exit 0
fi

echo "pipx not found. Falling back to user install with pip."
python3 -m pip install --user --upgrade pip
python3 -m pip install --user --upgrade "${SRC}"
echo "Done. Ensure ~/Library/Python/*/bin is in PATH, then run: aishell"
echo "Then configure providers with: /setup and /auth"
echo "Optional: install Ollama for local/offline DeepSeek with: /ollama install"
