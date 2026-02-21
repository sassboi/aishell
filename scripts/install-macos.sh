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
  if command -v brew >/dev/null 2>&1; then
    echo "npm not found. Installing node..."
    brew install node
    return 0
  fi
  echo "npm not found and Homebrew unavailable. Install Node.js manually."
  return 1
}

install_node_cli() {
  local label="$1"
  local package="$2"
  local binary="$3"
  if command -v "$binary" >/dev/null 2>&1; then
    echo "${label} already installed (${binary})."
    return 0
  fi
  ensure_npm || return 1
  echo "Installing ${label}..."
  if npm install -g "$package"; then
    return 0
  fi

  echo "Initial install failed; trying npm EACCES permission fix..."
  mkdir -p "$HOME/.npm-global"
  npm config set prefix "$HOME/.npm-global" || true
  if ! grep -q 'npm-global/bin' "$HOME/.zshrc" 2>/dev/null; then
    echo 'export PATH="$HOME/.npm-global/bin:$PATH"' >> "$HOME/.zshrc"
  fi
  export PATH="$HOME/.npm-global/bin:$PATH"

  npm install -g "$package" || {
    echo "Failed to install ${label} (${package}) after npm permission fix."
    return 1
  }
}

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "This installer supports macOS only."
  exit 1
fi

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 is required. Install Python 3.9+ and retry."
  exit 1
fi

SRC="${1:-.}"
AUTO_INSTALL_CLIS="${AI_INSTALL_CLIS:-1}"

post_install_wizard() {
  if [[ "$AUTO_INSTALL_CLIS" == "1" ]]; then
    echo "Auto-installing provider CLIs: codex, claude, gemini"
    install_node_cli "OpenAI Codex CLI" "@openai/codex" "codex" || true
    install_node_cli "Claude CLI" "@anthropic-ai/claude-code" "claude" || true
    install_node_cli "Gemini CLI" "@google/gemini-cli" "gemini" || true
  else
    echo "Skipping provider CLI auto-install (AI_INSTALL_CLIS=${AUTO_INSTALL_CLIS})."
  fi
  if ask_yes_no "Install Ollama (for local/offline DeepSeek)?" "N"; then
    if command -v ollama >/dev/null 2>&1; then
      echo "Ollama already installed."
    elif command -v brew >/dev/null 2>&1; then
      brew install ollama || echo "Failed to install Ollama."
    else
      echo "Homebrew not found. Install Ollama manually."
    fi
  fi
}

if command -v pipx >/dev/null 2>&1; then
  echo "Installing aishell with pipx from: ${SRC}"
  pipx install --force "${SRC}"
  echo "Done. Run: aishell"
  echo "Then configure providers with: /setup and /auth"
  echo "Optional: install Ollama for local/offline DeepSeek with: /ollama install"
  post_install_wizard
  exit 0
fi

echo "pipx not found. Falling back to user install with pip."
python3 -m pip install --user --upgrade pip
python3 -m pip install --user --upgrade "${SRC}"
echo "Done. Ensure ~/Library/Python/*/bin is in PATH, then run: aishell"
echo "Then configure providers with: /setup and /auth"
echo "Optional: install Ollama for local/offline DeepSeek with: /ollama install"
post_install_wizard
