# macOS Setup Guide

## 1. Install model CLIs
Install whichever model providers you plan to use, and ensure the binaries are on your `PATH`:
- `claude`
- `codex`
- `gemini`

## 2. Install aishell
From source:
```bash
pipx install .
```

Or run:
```bash
./scripts/install-macos.sh .
```

## 3. First run
```bash
aishell
```

The setup wizard will:
- Detect installed model CLIs.
- Ask which models to enable.
- Ask which model should be default.
- Offer to run authentication commands.

## 4. Re-run setup anytime
Inside `aishell`, run:
```text
/setup
```

## 5. Troubleshooting
- If `aishell` says a model binary is missing, install that model CLI and run `/setup` again.
- If authentication fails, run the provider login command manually and then rerun `/setup`.
- If `aishell` is not found after pip install, ensure your user bin directory is in PATH.
