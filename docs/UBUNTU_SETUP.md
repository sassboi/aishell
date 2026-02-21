# Ubuntu Setup Guide

## 1. Install model CLIs
Install whichever providers you want to use and ensure binaries are in `PATH`:
- `claude`
- `codex`
- `gemini`

## 2. Install aishell
```bash
./scripts/install-ubuntu.sh .
```

Or:
```bash
pipx install .
```

## 3. First run
```bash
aishell
```
Use `/setup` and `/auth` commands as needed.

## 4. GUI mode
```bash
aishell-gui
```
For screen-context capture in GUI, install one screenshot tool:
```bash
sudo apt-get install -y gnome-screenshot
# or
sudo apt-get install -y scrot
# or
sudo apt-get install -y imagemagick
```
