#!/usr/bin/env bash
set -euo pipefail

ask_yes_no() {
  local prompt="$1"
  local default="${2:-N}"
  local suffix="[y/N]"
  if [[ "$default" == "Y" ]]; then
    suffix="[Y/n]"
  fi
  read -r -p "${prompt} ${suffix} " ans
  ans="${ans:-$default}"
  [[ "$ans" =~ ^[Yy]([Ee][Ss])?$ ]]
}

ensure_npm() {
  if command -v npm >/dev/null 2>&1; then
    return 0
  fi
  echo "npm not found. Installing nodejs + npm..."
  sudo apt-get update
  sudo apt-get install -y nodejs npm
}

install_node_cli() {
  local label="$1"
  local package="$2"
  local binary="$3"
  if command -v "$binary" >/dev/null 2>&1; then
    echo "${label} already installed (${binary})."
    return 0
  fi
  ensure_npm
  echo "Installing ${label}..."
  npm install -g "$package" || {
    echo "Failed to install ${label} (${package})."
    return 1
  }
}

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

if ask_yes_no "Install optional AI provider CLIs now?" "N"; then
  if ask_yes_no "Install OpenAI Codex CLI (gpt model)?" "Y"; then
    install_node_cli "OpenAI Codex CLI" "@openai/codex" "codex" || true
  fi
  if ask_yes_no "Install Claude CLI?" "Y"; then
    install_node_cli "Claude CLI" "@anthropic-ai/claude-code" "claude" || true
  fi
  if ask_yes_no "Install Gemini CLI?" "Y"; then
    install_node_cli "Gemini CLI" "@google/gemini-cli" "gemini" || true
  fi
  if ask_yes_no "Install Ollama (for local/offline DeepSeek)?" "N"; then
    if command -v ollama >/dev/null 2>&1; then
      echo "Ollama already installed."
    else
      curl -fsSL https://ollama.com/install.sh | sh || echo "Failed to install Ollama."
    fi
  fi
fi
