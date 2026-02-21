# aishell (macOS + Ubuntu)

`aishell` is an AI-assisted shell that routes natural-language prompts to Claude, GPT (Codex CLI), Gemini, or local DeepSeek via Ollama, and runs normal shell commands directly.

## Features
- CLI mode (`aishell`) and desktop GUI mode (`aishell-gui`).
- Model/account/usage metadata in CLI prompt and `/usage`.
- Auth management from CLI via `/auth`.
- Smart suggestions, Tab completion, response-time output.

## Requirements
- Python 3.9+
- At least one model CLI installed and authenticated:
  - `claude`
  - `codex`
  - `gemini`
- For DeepSeek local/offline mode: `ollama` (model default: `deepseek-r1:8b`)

Notes:
- Ollama is optional. Install it only if you want DeepSeek local mode.
- Ollama is useful when you have no internet for inference (after the model is already pulled locally).

## Auto install (recommended)
One script for both macOS and Ubuntu:
```bash
./scripts/autoinstall.sh .
```

You can also pass a wheel path:
```bash
./scripts/autoinstall.sh /path/to/aishell_terminal-0.1.0-py3-none-any.whl
```

The installer can optionally prompt to install provider CLIs (`codex`, `claude`, `gemini`, `ollama`) and will only install each one with explicit user approval.

## Run
```bash
aishell
```

GUI mode:
```bash
aishell-gui
```

## Core Commands (CLI)
- `/help`
- `/model [claude|gpt|gemini|deepseek]`
- `/usage`
- `/auth [model] [status|login|logout]`
- `/ollama [status|install|uninstall|pull [model]]`
- `/update`
- `/setup`
- `/clear`
- `/history`
- `/save`
- `/smart [on|off]`
- `/dryrun [on|off]`
- `/exit`

Force shell execution with `!`:
```bash
! ls -la
```

## Build and share
```bash
python3 -m pip wheel . --no-deps --no-build-isolation -w dist
```
Share the wheel in `dist/`.

Install from wheel directly:
```bash
pipx install /path/to/aishell_terminal-0.1.0-py3-none-any.whl
```

Important:
- Wheel install does not force-install Ollama.
- Users must explicitly approve/install Ollama themselves (for example with `/ollama install` or `brew install ollama`), because it is large and optional.
- `aishell` checks for package updates and can be checked manually with `/update`.
