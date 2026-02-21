#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="${1:-$ROOT_DIR}"

echo "aishell installer"
echo "Note: Ollama is optional (used for local/offline DeepSeek), large, and is NOT auto-installed."
echo "Users must explicitly approve/install it later (for example: /ollama install)."
echo

case "$(uname -s)" in
  Darwin)
    exec "$ROOT_DIR/scripts/install-macos.sh" "$SRC"
    ;;
  Linux)
    exec "$ROOT_DIR/scripts/install-ubuntu.sh" "$SRC"
    ;;
  *)
    echo "Unsupported OS: $(uname -s)"
    echo "aishell currently supports macOS and Ubuntu Linux."
    exit 1
    ;;
esac
