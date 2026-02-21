#!/usr/bin/env python3
import base64
import importlib.metadata as importlib_metadata
import json
import os
import re
import shlex
import subprocess
import shutil
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from functools import lru_cache
from html import escape
from pathlib import Path
atexit = None
readline = None

try:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.auto_suggest import AutoSuggest, AutoSuggestFromHistory, Suggestion
    from prompt_toolkit.completion import Completer, Completion
    from prompt_toolkit.formatted_text import HTML
    from prompt_toolkit.history import FileHistory
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.styles import Style
    HAVE_PROMPT_TOOLKIT = True
except Exception:
    HAVE_PROMPT_TOOLKIT = False

# -------- Settings --------
HOME = Path.home()
STATE_DIR = HOME / ".ai_shell"
STATE_DIR.mkdir(exist_ok=True)
HIST_FILE = str(STATE_DIR / "history.txt")
CONFIG_FILE = STATE_DIR / "config.json"
UPDATE_FILE = STATE_DIR / "update.json"

DEFAULT_MODEL = os.environ.get("AI_DEFAULT", "claude")  # claude | gpt | gemini | deepseek
MAX_TURNS = int(os.environ.get("AI_MAX_TURNS", "12"))
FAST_TURNS = int(os.environ.get("AI_FAST_TURNS", "4"))
MODEL_TIMEOUT = {
    "gpt": int(os.environ.get("AI_TIMEOUT_GPT", "45")),
    "claude": int(os.environ.get("AI_TIMEOUT_CLAUDE", "35")),
    "gemini": int(os.environ.get("AI_TIMEOUT_GEMINI", "35")),
    "deepseek": int(os.environ.get("AI_TIMEOUT_DEEPSEEK", "90")),
}
SLASH_COMMANDS = ["/help", "/model", "/usage", "/auth", "/ollama", "/update", "/speed", "/fixnpm", "/setup", "/clear", "/history", "/save", "/smart", "/dryrun", "/exit"]
SHELL_BUILTINS = {"cd", "pwd"}
SUBCOMMANDS = {
    "git": [
        "add", "bisect", "branch", "checkout", "cherry-pick", "clone", "commit",
        "diff", "fetch", "grep", "init", "log", "merge", "mv", "pull", "push",
        "rebase", "reset", "restore", "revert", "show", "stash", "status", "switch",
        "tag", "worktree",
    ],
    "docker": [
        "build", "compose", "cp", "exec", "images", "inspect", "kill", "logs",
        "network", "ps", "pull", "push", "restart", "rm", "rmi", "run", "start",
        "stats", "stop", "system", "volume",
    ],
    "npm": ["audit", "ci", "install", "run", "start", "test", "update"],
    "pip": ["install", "list", "show", "uninstall", "freeze"],
    "brew": ["doctor", "info", "install", "list", "search", "services", "uninstall", "update", "upgrade"],
    "ollama": ["list", "run", "pull", "show", "ps", "stop", "rm", "serve"],
}

# One-shot commands (adjust if your CLIs need flags)
MODEL_CMD = {
    # Force non-interactive text mode so Claude does not enter internal
    # tool approval flows that cannot be answered through this wrapper.
    "claude": [
        "claude",
        "-p",
        "--permission-mode", "dontAsk",
        "--output-format", "text",
        "--no-session-persistence",
    ],
    # Non-interactive mode for scripting/wrappers:
    # - read-only sandbox by default
    # - never pause for approvals
    "gpt":    ["codex", "exec", "--sandbox", "read-only", "--skip-git-repo-check"],
    "gemini": ["gemini"],
    "deepseek": ["ollama", "run"],
}

MODEL_SETUP = {
    "claude": {
        "binary": "claude",
        "auth_cmd": ["claude", "login"],
        "install_hint": "Install Claude Code CLI and ensure `claude` is on PATH.",
    },
    "gpt": {
        "binary": "codex",
        "auth_cmd": ["codex", "login"],
        "install_hint": "Install OpenAI Codex CLI and ensure `codex` is on PATH.",
    },
    "gemini": {
        "binary": "gemini",
        "auth_cmd": ["gemini", "auth", "login"],
        "install_hint": "Install Gemini CLI and ensure `gemini` is on PATH.",
    },
    "deepseek": {
        "binary": "ollama",
        "install_hint": "Install Ollama and ensure `ollama` is on PATH.",
    },
}

def normalize_model(m: str) -> str:
    m = (m or "").strip().lower()
    if m in ("codex", "chatgpt", "openai"):
        return "gpt"
    if m in ("ds", "deepseek"):
        return "deepseek"
    if m in ("claude", "gpt", "gemini", "deepseek"):
        return m
    return "claude"

model = normalize_model(DEFAULT_MODEL)
history = []  # list of (role, text)
USE_COLOR = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None
PT_STYLE = None
if HAVE_PROMPT_TOOLKIT:
    PT_STYLE = Style.from_dict({
        "prompt.model": "bold ansiblack bg:ansicyan",
        "prompt.path": "bold ansiwhite",
        "prompt.sep": "ansicyan",
    })

MODEL_COLOR = {
    "claude": "35",
    "gpt": "36",
    "gemini": "33",
    "deepseek": "32",
}
MODEL_ICON = {
    "claude": "[CLD]",
    "gpt": "[GPT]",
    "gemini": "[GEM]",
    "deepseek": "[DSK]",
}

def color(text: str, code: str):
    if not USE_COLOR:
        return text
    return f"\033[{code}m{text}\033[0m"

def model_badge(m: str):
    code = MODEL_COLOR.get(m, "37")
    icon = MODEL_ICON.get(m, f"[{m.upper()}]")
    exact = get_model_metadata(m).get("exact", m)
    return color(f"{icon} {exact}", f"1;{code}")

def prompt_text_for(model_name: str, cwd: str):
    return f"{model_badge(model_name)} {color(cwd, '1;37')} {color('>>', '36')} "

def prompt_html_for(model_name: str, cwd: str):
    icon = MODEL_ICON.get(model_name, f"[{model_name.upper()}]")
    exact = escape(get_model_metadata(model_name).get("exact", model_name))
    return (
        f'<prompt.model>{icon} {exact}</prompt.model> '
        f'<prompt.path>{escape(cwd)}</prompt.path> '
        f'<prompt.sep>&gt;&gt;</prompt.sep> '
    )

def print_banner(model_name: str, smart_mode: bool):
    bar = "=" * 62
    print(color(bar, "2;37"))
    title = color("AI Shell", "1;97")
    meta = get_model_metadata(model_name)
    mode = color(f"model={meta['exact']}", f"1;{MODEL_COLOR.get(model_name, '37')}")
    smart = color(f"smart={'on' if smart_mode else 'off'}", "1;32" if smart_mode else "1;31")
    print(f"{title}  {mode}  {smart}")
    print(color(f"identity={meta.get('user', 'unknown')}  usage={meta.get('usage', 'unknown')}", "2;37"))
    print(color("Type /help for commands", "2;37"))
    print(color(bar, "2;37"))

def print_status(tag: str, msg: str, code: str = "36"):
    print(f"{color(f'[{tag}]', f'1;{code}')} {msg}")

def print_help():
    print(color("""
AI Shell commands:
  /model [claude|gpt|gemini|deepseek]   Switch model
  /usage                      Show model/identity/usage details
  /auth [model] [action]      Auth actions: status|login|logout
  /ollama [action]            Ollama actions: status|install|uninstall|pull [model]
  /update                     Check if a newer aishell version is available
  /speed [fast|balanced]      Prefer lower latency vs richer context
  /fixnpm                     Fix npm global install EACCES permissions
  /setup                      Re-run model setup/auth wizard
  /clear                      Clear chat context
  /history                    Show current context (truncated)
  /save                       Save transcript to ~/.ai_shell/
  /exit                       Exit AI Shell
  /smart [on|off]             Toggle smart command suggestions
  /dryrun [on|off]            If on, never execute commands (even with y)

Usage:
  - Normal shell commands run as shell.
  - Non-commands are treated as AI (with persistent context).
  - If AI suggests a command, you can press y to run it.
  - Press Tab to autocomplete commands and paths.

Force shell: prefix with !
  ! echo hello
""".strip(), "37"))

def _decode_jwt_payload(token: str):
    parts = (token or "").split(".")
    if len(parts) < 2:
        return {}
    payload = parts[1]
    payload += "=" * (-len(payload) % 4)
    try:
        return json.loads(base64.urlsafe_b64decode(payload).decode("utf-8"))
    except Exception:
        return {}

def _format_ttl(seconds: int):
    if seconds <= 0:
        return "expired"
    hours, rem = divmod(seconds, 3600)
    mins, _ = divmod(rem, 60)
    if hours:
        return f"{hours}h {mins}m"
    return f"{mins}m"

def _normalize_email(value: str):
    return (value or "").strip()

def deepseek_ollama_model() -> str:
    return (os.environ.get("AI_DEEPSEEK_MODEL") or "deepseek-r1:8b").strip()

def _current_version() -> str:
    try:
        return importlib_metadata.version("aishell-terminal")
    except Exception:
        return "0.1.0"

def _version_tuple(v: str):
    nums = [int(x) for x in re.findall(r"\d+", v or "")]
    return tuple(nums[:4]) if nums else (0,)

def check_for_updates(force: bool = False):
    now = int(time.time())
    state = _read_json(UPDATE_FILE)
    last = int(state.get("last_checked") or 0)
    if not force and (now - last) < 24 * 3600:
        return False

    current = _current_version()
    latest = ""
    try:
        with urllib.request.urlopen("https://pypi.org/pypi/aishell-terminal/json", timeout=2) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
            latest = str(((payload.get("info") or {}).get("version") or "")).strip()
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError):
        state["last_checked"] = now
        try:
            UPDATE_FILE.write_text(json.dumps(state), encoding="utf-8")
        except Exception:
            pass
        if force:
            print_status("update", "Could not reach update server.", "33")
        return False

    state["last_checked"] = now
    state["latest_seen"] = latest
    try:
        UPDATE_FILE.write_text(json.dumps(state), encoding="utf-8")
    except Exception:
        pass

    if latest and _version_tuple(latest) > _version_tuple(current):
        print_status("update", f"New aishell version available: {latest} (current {current})", "33")
        print_status("update", "Upgrade with: pipx upgrade aishell-terminal", "33")
        return True
    if force:
        print_status("update", f"You are on the latest version ({current}).", "32")
    return False

def looks_like_usage_exhausted(text: str) -> bool:
    s = (text or "").lower()
    markers = (
        "insufficient_quota",
        "usage limit",
        "quota exceeded",
        "rate limit",
        "too many requests",
        "out of credits",
        "billing",
        "exhausted",
        "free tier",
        "pro plan",
        "upgrade plan",
    )
    return any(m in s for m in markers)

AUTH_ACTIONS = {"status", "login", "logout"}
OLLAMA_ACTIONS = {"status", "install", "uninstall", "pull"}

def _ollama_install_plan():
    if sys.platform == "darwin" and shutil.which("brew"):
        return (["brew", "install", "ollama"], False, "brew install ollama")
    if sys.platform.startswith("linux"):
        if shutil.which("snap"):
            return (["sudo", "snap", "install", "ollama"], False, "sudo snap install ollama")
        if shutil.which("curl"):
            return ("curl -fsSL https://ollama.com/install.sh | sh", True, "curl -fsSL https://ollama.com/install.sh | sh")
    return (None, False, "")

def _ollama_uninstall_plan():
    if sys.platform == "darwin" and shutil.which("brew"):
        return (["brew", "uninstall", "ollama"], False, "brew uninstall ollama")
    if sys.platform.startswith("linux"):
        if shutil.which("snap"):
            return (["sudo", "snap", "remove", "ollama"], False, "sudo snap remove ollama")
        if shutil.which("apt-get"):
            return (["sudo", "apt-get", "remove", "-y", "ollama"], False, "sudo apt-get remove -y ollama")
    return (None, False, "")

def _run_live(cmd, shell_mode: bool = False):
    if shell_mode:
        return subprocess.run(cmd, shell=True, cwd=os.getcwd()).returncode
    return subprocess.run(cmd, cwd=os.getcwd()).returncode

def run_ollama_command(action: str, arg: str = ""):
    action = (action or "status").strip().lower()
    if action not in OLLAMA_ACTIONS:
        print("Usage: /ollama [status|install|uninstall|pull [model]]")
        return

    if action == "status":
        ov = _ollama_version()
        if not ov:
            print_status("ollama", "not installed", "31")
            _, _, install_text = _ollama_install_plan()
            if install_text:
                print_status("ollama", f"Install with: {install_text}", "33")
            return
        models = _ollama_list_models()
        print_status("ollama", ov, "32")
        if models:
            shown = ", ".join(models[:6])
            suffix = " ..." if len(models) > 6 else ""
            print_status("models", f"{len(models)} installed: {shown}{suffix}", "34")
        else:
            print_status("models", "none installed", "33")
        return

    if action == "install":
        if shutil.which("ollama"):
            print_status("ollama", "already installed", "32")
            return
        plan_cmd, shell_mode, display = _ollama_install_plan()
        if not plan_cmd:
            print_status("ollama", "no supported installer found on this OS.", "31")
            return
        if yes_no_prompt(f"Run now: {display}?", default_yes=False):
            rc = _run_live(plan_cmd, shell_mode=shell_mode)
            print_status("ollama", "install complete" if rc == 0 else f"install failed (exit {rc})", "32" if rc == 0 else "31")
        return

    if action == "uninstall":
        if not shutil.which("ollama"):
            print_status("ollama", "already not installed", "34")
            return
        plan_cmd, shell_mode, display = _ollama_uninstall_plan()
        if not plan_cmd:
            print_status("ollama", "no supported uninstall command found on this OS.", "31")
            return
        if yes_no_prompt(f"This will remove Ollama. Run: {display}?", default_yes=False):
            rc = _run_live(plan_cmd, shell_mode=shell_mode)
            print_status("ollama", "uninstall complete" if rc == 0 else f"uninstall failed (exit {rc})", "32" if rc == 0 else "31")
        return

    if action == "pull":
        target = (arg or deepseek_ollama_model()).strip()
        if not target:
            target = deepseek_ollama_model()
        if shutil.which("ollama") is None:
            print_status("ollama", "not installed", "31")
            return
        print_status("ollama", f"pulling {target}", "34")
        rc = _run_live(["ollama", "pull", target], shell_mode=False)
        print_status("ollama", "pull complete" if rc == 0 else f"pull failed (exit {rc})", "32" if rc == 0 else "31")
        return

def fix_npm_permissions():
    if shutil.which("npm") is None:
        print_status("fixnpm", "npm not found on PATH.", "31")
        return

    try:
        p = subprocess.run(["npm", "config", "get", "prefix"], text=True, capture_output=True, timeout=5)
        prefix = (p.stdout or "").strip()
    except Exception:
        prefix = ""

    if prefix and os.path.isdir(prefix) and os.access(prefix, os.W_OK):
        print_status("fixnpm", f"npm prefix already writable: {prefix}", "32")
        return

    target = str(Path.home() / ".npm-global")
    try:
        Path(target).mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print_status("fixnpm", f"failed to create {target}: {e}", "31")
        return

    rc = subprocess.run(["npm", "config", "set", "prefix", target], cwd=os.getcwd()).returncode
    if rc != 0:
        print_status("fixnpm", "failed to set npm prefix", "31")
        return

    shell_name = Path(os.environ.get("SHELL", "")).name
    if shell_name == "zsh":
        rc_file = Path.home() / ".zshrc"
    elif shell_name == "bash":
        rc_file = Path.home() / ".bashrc"
    else:
        rc_file = Path.home() / ".profile"

    export_line = 'export PATH="$HOME/.npm-global/bin:$PATH"'
    try:
        existing = rc_file.read_text(encoding="utf-8") if rc_file.exists() else ""
        if export_line not in existing:
            with rc_file.open("a", encoding="utf-8") as f:
                if existing and not existing.endswith("\n"):
                    f.write("\n")
                f.write(export_line + "\n")
    except Exception as e:
        print_status("fixnpm", f"prefix set, but failed to update {rc_file}: {e}", "33")
        print_status("fixnpm", f"manually add: {export_line}", "33")
        return

    print_status("fixnpm", f"npm global prefix set to {target}", "32")
    print_status("fixnpm", f"added PATH export to {rc_file}", "32")
    print_status("fixnpm", "open a new terminal (or source your shell rc) and retry npm install -g", "34")

def run_auth_command(model_name: str, action: str):
    if model_name == "deepseek":
        if action == "status":
            m = deepseek_ollama_model()
            ov = _ollama_version() or "unknown"
            if shutil.which("ollama") is None:
                print_status("auth", "ollama is not installed. Install Ollama first.", "31")
                return
            if _ollama_has_model(m):
                print_status("auth", f"deepseek ready via ollama; model={m}; ollama={ov}", "32")
            else:
                print_status("auth", f"ollama installed but model not pulled: {m}", "33")
                print_status("auth", f"Run: ollama pull {m}", "33")
            return
        if action == "login":
            print_status("auth", "DeepSeek via Ollama does not require account login.", "34")
            _ensure_deepseek_ollama_ready(interactive=True)
            return
        if action == "logout":
            print_status("auth", "DeepSeek via Ollama has no auth session to log out.", "34")
            return
    commands = {
        "gpt": {
            "status": ["codex", "login", "status"],
            "login": ["codex", "login"],
            "logout": ["codex", "logout"],
        },
        "claude": {
            "status": ["claude", "auth", "status"],
            "login": ["claude", "auth", "login"],
            "logout": ["claude", "auth", "logout"],
        },
        "gemini": {
            "status": ["gemini", "auth", "status"],
            "login": ["gemini", "auth", "login"],
            "logout": ["gemini", "auth", "logout"],
        },
        "deepseek": {},
    }
    cmd = ((commands.get(model_name) or {}).get(action) or [])
    if not cmd:
        print_status("auth", f"unsupported auth action for {model_name}", "31")
        return
    print_status("auth", f"running: {' '.join(cmd)}", "34")
    try:
        rc = subprocess.run(cmd, cwd=os.getcwd()).returncode
        if rc == 0:
            print_status("auth", f"{model_name} {action}: ok", "32")
        else:
            print_status("auth", f"{model_name} {action}: exit {rc}", "31")
    except FileNotFoundError:
        print_status("auth", f"command not found: {cmd[0]}", "31")

def _read_json(path: Path):
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}

@lru_cache(maxsize=1)
def _ollama_version():
    for args in (["ollama", "--version"], ["ollama", "-v"]):
        try:
            p = subprocess.run(
                args,
                text=True,
                capture_output=True,
                timeout=4,
            )
            text = ((p.stdout or "") + "\n" + (p.stderr or "")).strip()
            if p.returncode == 0 and text:
                first = text.splitlines()[0].strip()
                return first or None
        except Exception:
            continue
    return None

def _ollama_list_models():
    try:
        p = subprocess.run(
            ["ollama", "list"],
            text=True,
            capture_output=True,
            timeout=8,
        )
    except Exception:
        return []
    if p.returncode != 0:
        return []
    models = []
    for ln in (p.stdout or "").splitlines()[1:]:
        cols = ln.split()
        if cols:
            models.append(cols[0].strip())
    return models

def _ollama_has_model(model_name: str) -> bool:
    names = set(_ollama_list_models())
    if model_name in names:
        return True
    base = model_name.split(":", 1)[0]
    return any(n.split(":", 1)[0] == base for n in names)

def _ensure_deepseek_ollama_ready(interactive: bool = True) -> bool:
    m = deepseek_ollama_model()
    if shutil.which("ollama") is None:
        print_status("auth", "ollama is not installed. Install Ollama and retry.", "31")
        return False
    if _ollama_has_model(m):
        return True
    print_status("auth", f"DeepSeek model not found locally: {m}", "33")
    if interactive and yes_no_prompt(f"Pull {m} with Ollama now?", default_yes=True):
        print_status("auth", f"running: ollama pull {m}", "34")
        rc = subprocess.run(["ollama", "pull", m], cwd=os.getcwd()).returncode
        if rc != 0:
            print_status("auth", f"ollama pull failed with exit {rc}", "31")
            return False
        return _ollama_has_model(m)
    return False

def _latest_gemini_session():
    base = Path.home() / ".gemini" / "tmp"
    if not base.exists():
        return {}
    try:
        files = sorted(base.glob("*/*/session-*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    except Exception:
        files = []
    if not files:
        return {}
    data = _read_json(files[0])
    messages = data.get("messages") or []
    for msg in reversed(messages):
        if isinstance(msg, dict) and msg.get("type") == "gemini":
            return msg
    return {}

def get_model_metadata(model_name: str):
    meta = {
        "exact": model_name,
        "user": "unknown",
        "usage": "quota unavailable",
    }
    if model_name == "gpt":
        cfg = Path.home() / ".codex" / "config.toml"
        auth = Path.home() / ".codex" / "auth.json"
        try:
            if cfg.exists():
                for ln in cfg.read_text(encoding="utf-8").splitlines():
                    s = ln.strip()
                    if s.startswith("model"):
                        _, rhs = s.split("=", 1)
                        exact = rhs.strip().strip('"').strip("'")
                        if exact:
                            meta["exact"] = exact
                        break
        except Exception:
            pass
        try:
            if auth.exists():
                data = _read_json(auth)
                claims = _decode_jwt_payload(((data.get("tokens") or {}).get("id_token") or ""))
                email = _normalize_email(claims.get("email") or "")
                if email:
                    meta["user"] = email
                exp = int(claims.get("exp") or 0)
                if exp:
                    ttl = exp - int(time.time())
                    meta["usage"] = f"quota unavailable; auth ttl {_format_ttl(ttl)}"
        except Exception:
            pass
    elif model_name == "claude":
        settings = _read_json(Path.home() / ".claude" / "settings.json")
        state = _read_json(Path.home() / ".claude.json")
        oauth = state.get("oauthAccount") if isinstance(state, dict) else {}
        meta["exact"] = settings.get("model") or "claude (cli default)"
        email = _normalize_email((oauth or {}).get("emailAddress") or "")
        if email:
            meta["user"] = email
        billing = (oauth or {}).get("billingType") or "unknown"
        extra = (oauth or {}).get("hasExtraUsageEnabled")
        usage_bits = [f"billing={billing}", "quota unavailable"]
        if isinstance(extra, bool):
            usage_bits.insert(1, f"extra_usage={'on' if extra else 'off'}")
        meta["usage"] = "; ".join(usage_bits)
    elif model_name == "gemini":
        settings = _read_json(Path.home() / ".gemini" / "settings.json")
        accounts = _read_json(Path.home() / ".gemini" / "google_accounts.json")
        creds = _read_json(Path.home() / ".gemini" / "oauth_creds.json")
        latest = _latest_gemini_session()

        meta["exact"] = settings.get("model") or latest.get("model") or "gemini (cli default)"
        email = _normalize_email(accounts.get("active") or "")
        if email:
            meta["user"] = email

        usage_parts = ["quota unavailable"]
        exp_ms = int(creds.get("expiry_date") or 0)
        if exp_ms:
            ttl = int(exp_ms / 1000) - int(time.time())
            usage_parts.append(f"auth ttl {_format_ttl(ttl)}")
        tokens = latest.get("tokens") if isinstance(latest, dict) else {}
        total = (tokens or {}).get("total")
        if isinstance(total, int):
            usage_parts.append(f"last_turn_tokens={total}")
        meta["usage"] = "; ".join(usage_parts)
    elif model_name == "deepseek":
        selected = deepseek_ollama_model()
        ov = _ollama_version()
        meta["exact"] = f"{selected} (ollama local; {ov})" if ov else f"{selected} (ollama local)"
        meta["user"] = (os.environ.get("USER") or "local")
        if shutil.which("ollama") is None:
            meta["usage"] = "ollama missing; install required; quota n/a"
        elif _ollama_has_model(selected):
            meta["usage"] = "local model ready; quota n/a"
        else:
            meta["usage"] = f"model not pulled: {selected}; quota n/a"
    return meta

def print_model_info(model_name: str):
    meta = get_model_metadata(model_name)
    code = MODEL_COLOR.get(model_name, "37")
    icon = MODEL_ICON.get(model_name, f"[{model_name.upper()}]")
    print_status("model", f"{icon} {meta['exact']}", code)
    print_status("identity", meta.get("user", "unknown"), "90")
    print_status("usage", meta.get("usage", "unknown"), "90")

def load_config():
    if not CONFIG_FILE.exists():
        return {}
    try:
        with CONFIG_FILE.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except (OSError, json.JSONDecodeError):
        pass
    return {}

def save_config(cfg):
    with CONFIG_FILE.open("w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, sort_keys=True)
        f.write("\n")

def model_binary_available(model_name: str):
    setup = MODEL_SETUP.get(model_name, {})
    binary = setup.get("binary")
    if not binary:
        return False, None
    resolved = shutil.which(binary)
    return resolved is not None, resolved

def pick_best_available_model(enabled_models, preferred=None):
    ordered = []
    preferred = normalize_model(preferred or "")
    if preferred and preferred in enabled_models:
        ordered.append(preferred)
    for m in enabled_models:
        if m not in ordered:
            ordered.append(m)
    for m in ordered:
        ok, _ = model_binary_available(m)
        if ok:
            return m
    return ordered[0] if ordered else normalize_model(DEFAULT_MODEL)

def yes_no_prompt(message: str, default_yes: bool = True) -> bool:
    default = "Y/n" if default_yes else "y/N"
    raw = input(f"{message} [{default}] ").strip().lower()
    if not raw:
        return default_yes
    return raw in ("y", "yes")

def run_auth(model_name: str):
    auth_cmd = MODEL_SETUP.get(model_name, {}).get("auth_cmd")
    if not auth_cmd:
        return
    try:
        print(f"[auth] Running: {' '.join(auth_cmd)}")
        rc = subprocess.run(auth_cmd).returncode
        if rc == 0:
            print(f"[auth] {model_name}: success")
        else:
            print(f"[auth] {model_name}: command exited with code {rc}")
    except FileNotFoundError:
        print(f"[auth] {model_name}: command not found")

def run_setup_wizard(existing_cfg=None):
    existing_cfg = existing_cfg or {}
    print(f"AI Shell setup ({sys.platform})")
    print("Select models to enable and optionally authenticate them now.\n")

    enabled = []
    model_order = ["claude", "gpt", "gemini", "deepseek"]
    prev_enabled = set(existing_cfg.get("enabled_models", []))

    for name in model_order:
        available, path = model_binary_available(name)
        status = f"found at {path}" if available else "not found"
        print(f"- {name}: {status}")
        should_enable = yes_no_prompt(
            f"Enable {name}?",
            default_yes=(name in prev_enabled) if prev_enabled else available,
        )
        if not should_enable:
            continue
        enabled.append(name)
        if not available:
            print(f"  [setup] {MODEL_SETUP[name]['install_hint']}")

    if not enabled:
        print("[setup] No models enabled; defaulting to claude.")
        enabled = ["claude"]

    prev_default = normalize_model(existing_cfg.get("default_model", DEFAULT_MODEL))
    if prev_default not in enabled:
        prev_default = enabled[0]

    print("\nChoose default model:")
    for idx, name in enumerate(enabled, start=1):
        marker = "*" if name == prev_default else " "
        print(f"  {idx}. [{marker}] {name}")
    raw = input(f"Default model number [1-{len(enabled)}] (Enter for {prev_default}): ").strip()
    if raw.isdigit():
        picked = int(raw)
        if 1 <= picked <= len(enabled):
            default_model = enabled[picked - 1]
        else:
            default_model = prev_default
    else:
        default_model = prev_default

    for name in enabled:
        available, _ = model_binary_available(name)
        auth_cmd = MODEL_SETUP.get(name, {}).get("auth_cmd")
        if not available or not auth_cmd:
            continue
        if yes_no_prompt(f"Run auth for {name} now?", default_yes=False):
            run_auth(name)

    cfg = {
        "setup_complete": True,
        "enabled_models": enabled,
        "default_model": default_model,
    }
    save_config(cfg)
    print(f"\n[setup] Saved config to {CONFIG_FILE}")
    return cfg

@lru_cache(maxsize=1)
def get_path_commands(path_env: str):
    cmds = set()
    for d in path_env.split(os.pathsep):
        if not d:
            continue
        try:
            for name in os.listdir(d):
                full = os.path.join(d, name)
                if os.path.isfile(full) and os.access(full, os.X_OK):
                    cmds.add(name)
        except OSError:
            continue
    return sorted(cmds)

def complete_path(text: str):
    raw = text or ""
    raw_dir = os.path.dirname(raw)
    prefix = os.path.basename(raw)
    search_dir = os.path.expanduser(raw_dir) if raw_dir else "."

    try:
        entries = os.listdir(search_dir)
    except OSError:
        return []

    out = []
    for entry in entries:
        if not entry.startswith(prefix):
            continue
        full = os.path.join(search_dir, entry)
        candidate = os.path.join(raw_dir, entry) if raw_dir else entry
        if os.path.isdir(full):
            candidate += "/"
        out.append(candidate.replace(" ", r"\ "))
    return sorted(out)

def build_completions(line: str, text: str, begidx: int):
    # Slash command completion at start of line
    if line.startswith("/") and begidx == 0:
        return [c for c in SLASH_COMMANDS if c.startswith(text)]

    # First token: complete executable names
    if begidx == 0 or (line.startswith("!") and begidx == 1):
        cmds = get_path_commands(os.environ.get("PATH", ""))
        return [c for c in cmds if c.startswith(text)]

    # Other tokens: complete paths/files
    return complete_path(text)

def readline_completer(text, state):
    if readline is None:
        return None
    line = readline.get_line_buffer()
    begidx = readline.get_begidx()
    options = build_completions(line, text, begidx)
    if state < len(options):
        return options[state]
    return None

if HAVE_PROMPT_TOOLKIT:
    class AIShellCompleter(Completer):
        def __init__(self, live_enabled_getter=None):
            self.live_enabled_getter = live_enabled_getter or (lambda: True)

        def get_completions(self, document, complete_event):
            # Keep Tab completion available even when live suggestions are off.
            if not self.live_enabled_getter() and not complete_event.completion_requested:
                return
            line = document.text_before_cursor
            word = document.get_word_before_cursor(WORD=True)
            begidx = document.cursor_position - len(word)
            for option in build_completions(line, word, begidx):
                yield Completion(option, start_position=-len(word))

    class AIShellAutoSuggest(AutoSuggest):
        def __init__(self, enabled_getter=None):
            self.history_suggest = AutoSuggestFromHistory()
            self.enabled_getter = enabled_getter or (lambda: True)

        def get_suggestion(self, buffer, document):
            if not self.enabled_getter():
                return None
            hist = self.history_suggest.get_suggestion(buffer, document)
            if hist:
                return hist

            text = document.text
            if not text.strip():
                return None

            if text.startswith("/"):
                matches = [c for c in SLASH_COMMANDS if c.startswith(text) and c != text]
                if matches:
                    return Suggestion(matches[0][len(text):])
                return None

            try:
                parts = shlex.split(text)
            except ValueError:
                return None
            if not parts:
                return None

            first = parts[0]

            if len(parts) == 1 and not text.endswith(" ") and not command_exists(first):
                matches = suggest_commands(first, limit=1)
                if matches:
                    return Suggestion(matches[0][len(first):])
                return None

            if first in SUBCOMMANDS and len(parts) >= 2 and not text.endswith(" "):
                sub = parts[1]
                if sub and not sub.startswith("-"):
                    matches = [s for s in SUBCOMMANDS[first] if s.startswith(sub)]
                    if matches:
                        return Suggestion(matches[0][len(sub):])

            if first == "cd" and len(parts) == 2 and not text.endswith(" "):
                target = parts[1]
                if not os.path.isdir(os.path.expanduser(target)):
                    matches = suggest_cd_targets(target, limit=1)
                    if matches:
                        return Suggestion(matches[0][len(target):])

            return None

def command_exists(cmd: str) -> bool:
    return cmd in SHELL_BUILTINS or shutil.which(cmd) is not None

def suggest_commands(prefix: str, limit: int = 8):
    if not prefix:
        return []
    cmds = get_path_commands(os.environ.get("PATH", ""))
    return [c for c in cmds if c.startswith(prefix)][:limit]

def suggest_cd_targets(prefix: str, limit: int = 8):
    dirs = []
    raw = prefix or ""
    raw_dir = os.path.dirname(raw)
    name_prefix = os.path.basename(raw)
    search_dir = os.path.expanduser(raw_dir) if raw_dir else "."
    try:
        for entry in os.listdir(search_dir):
            if not entry.startswith(name_prefix):
                continue
            full = os.path.join(search_dir, entry)
            if os.path.isdir(full):
                candidate = os.path.join(raw_dir, entry) if raw_dir else entry
                dirs.append(candidate)
    except OSError:
        return []
    return sorted(dirs)[:limit]

def shell_partial_suggestions(line: str):
    """
    Return a list of suggestions if the input appears to be an incomplete shell command.
    Otherwise return an empty list.
    """
    try:
        parts = shlex.split(line)
    except ValueError:
        return []
    if not parts:
        return []

    first = parts[0]

    # Incomplete executable name
    if not command_exists(first):
        return suggest_commands(first)

    # Builtin-specific suggestions
    if first == "cd" and len(parts) == 2:
        target = os.path.expanduser(parts[1])
        if not os.path.isdir(target):
            return suggest_cd_targets(parts[1])

    # Common CLI subcommand suggestions (e.g., git che -> checkout/cherry-pick)
    if first in SUBCOMMANDS and len(parts) >= 2:
        sub = parts[1]
        if sub.startswith("-"):
            return []
        options = SUBCOMMANDS[first]
        if sub not in options:
            return [f"{first} {s}" for s in options if s.startswith(sub)][:8]

    return []

def save_transcript():
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = STATE_DIR / f"transcript.{ts}.{model}.txt"
    with out.open("w", encoding="utf-8") as f:
        for r, t in history:
            f.write(f"{r.upper()}: {t}\n\n")
    print(f"[saved] {out}")

def run_shell(line: str):
    line = line.strip()
    if not line:
        return

    # Builtin: cd (persist within this Python process)
    if line == "cd" or line.startswith("cd "):
        parts = shlex.split(line)
        # cd -> home
        target = os.path.expanduser("~") if len(parts) == 1 else os.path.expanduser(parts[1])
        try:
            os.chdir(target)
        except FileNotFoundError:
            print(f"cd: no such file or directory: {target}")
        except NotADirectoryError:
            print(f"cd: not a directory: {target}")
        return

    # Builtin: pwd (use current working directory)
    if line == "pwd":
        print(os.getcwd())
        return

    # Run everything else in a login shell so PATH/aliases work reasonably well
    subprocess.run(["bash", "-lc", line], cwd=os.getcwd())

def looks_like_natural_language(line: str) -> bool:
    s = line.strip().lower()
    if not s:
        return False

    # Strong NL signals
    if "?" in s:
        return True

    # Common NL starters
    nl_starters = (
        "what", "why", "how", "explain", "summarise", "summarize",
        "help", "can you", "could you", "please", "tell me", "show me",
        "where", "which", "who"
    )
    if s.startswith(nl_starters):
        return True

    # If it has multiple words and none look like paths/flags, often NL.
    # (This catches: "find files here", "list files sorted by size", etc.)
    tokens = s.split()
    if len(tokens) >= 3:
        has_flag = any(t.startswith("-") for t in tokens[1:])
        has_pathy = any(("/" in t) or (t.startswith(".")) for t in tokens)
        if not has_flag and not has_pathy:
            return True

    return False

def should_prefer_explanation(line: str) -> bool:
    """
    Return True for conceptual prompts where a prose explanation is likely
    better than command suggestions.
    """
    s = (line or "").strip().lower()
    if not s:
        return False

    # Keep smart CMD mode on for concrete file/task requests even if phrased
    # as a question, e.g. "what is the shape of cdf_grid.mat?".
    task_markers = (
        ".mat", ".csv", ".json", ".txt", ".py", "/", "./", "../",
        "shape", "size", "count", "list", "find", "grep", "check", "file",
    )
    if any(marker in s for marker in task_markers):
        return False

    starters = (
        "why", "how does", "what is", "who is", "where is", "which is",
        "explain", "describe", "tell me", "summarize", "summarise",
    )
    return s.startswith(starters)

def is_shell_command(line: str) -> bool:
    # Force shell if prefixed with !
    if line.startswith("!"):
        return True

    # If it looks like natural language, prefer AI even if first token is a real command
    if looks_like_natural_language(line):
        return False

    try:
        parts = shlex.split(line)
    except ValueError:
        return False
    if not parts:
        return True
    first = parts[0]
    shell_ops = ["|", ">", ">>", "<", "&&", "||", ";"]
    if any(op in line for op in shell_ops):
        return True
    return command_exists(first)

def build_prompt(user_text: str, smart_mode: bool, max_turns: int = MAX_TURNS, fast_mode: bool = False) -> str:
    ctx = history[-2 * max_turns:]

    system = [
        "You are an assistant inside a terminal for a PhD workflow.",
        "Be concise and practical.",
        "If you propose terminal commands, prefer safe, non-destructive commands.",
        "Never include secrets or tokens.",
    ]
    if fast_mode:
        system.append("Latency mode is FAST: answer briefly and skip long explanations.")

    if smart_mode:
        system += [
            "",
            "IMPORTANT OUTPUT FORMAT:",
            "If a terminal command would help, include EXACTLY ONE command suggestion in this format:",
            "CMD: <single-line command here>",
            "Then provide a short explanation.",
            "If no command is needed, do NOT output CMD: at all.",
            "",
            "Rules for CMD:",
            "- single line only (no newlines)",
            "- avoid destructive actions (rm, sudo, mv overwriting, etc.) unless explicitly requested",
            "- if potentially destructive, suggest a safer inspection command first (ls, find, rg, git diff, etc.)",
            "- for file/folder search, prefer fast commands (fd/rg) or limit traversal depth (e.g. find . -maxdepth N)",
        ]

    lines = []
    lines.extend(system)
    lines.append("")
    for r, t in ctx:
        lines.append(("User: " if r == "user" else "Assistant: ") + t)
    lines.append(f"User: {user_text}")
    lines.append("Assistant:")
    return "\n".join(lines)

def call_model(prompt: str) -> str:
    if model == "deepseek":
        cmd = ["ollama", "run", deepseek_ollama_model()]
    else:
        cmd = MODEL_CMD.get(model)
    if not cmd:
        return f"[error] unknown model: {model}"

    try:
        timeout_s = MODEL_TIMEOUT.get(model, 45)
        if model == "gpt":
            # Run in non-interactive mode and explicitly set working directory
            p = subprocess.run(
                cmd + ["--cd", os.getcwd(), prompt],
                text=True,
                capture_output=True,
                timeout=timeout_s,
            )
        elif model == "deepseek":
            p = subprocess.run(cmd + [prompt], text=True, capture_output=True, timeout=timeout_s)
        else:
            p = subprocess.run(cmd + [prompt], text=True, capture_output=True, timeout=timeout_s)
    except FileNotFoundError:
        return f"[error] '{cmd[0]}' not found on PATH."
    except subprocess.TimeoutExpired:
        return f"[error] {model} command timed out after {MODEL_TIMEOUT.get(model, 45)} seconds."

    out = (p.stdout or "").strip()
    err = (p.stderr or "").strip()

    # In codex exec: progress streams to stderr; final answer goes to stdout.
    if not out and err:
        out = err
    return out if out else "[no output returned]"

def extract_cmd(text: str):
    """
    Find a single-line command in the form:
      CMD: <...>
    Returns (cmd, cleaned_text_without_cmdline)
    """
    lines = text.splitlines()
    cmd_line_idx = None
    cmd = None
    for i, ln in enumerate(lines):
        if ln.strip().startswith("CMD:"):
            cmd_line_idx = i
            cmd = ln.split("CMD:", 1)[1].strip()
            break

    if cmd_line_idx is None or not cmd:
        return None, text

    # Remove the CMD line from displayed response
    cleaned = "\n".join([ln for j, ln in enumerate(lines) if j != cmd_line_idx]).strip()
    return cmd, cleaned

def confirm_run(cmd: str) -> bool:
    ans = input(f"\nRun this command? [y/N]  {cmd}\n> ").strip().lower()
    return ans == "y" or ans == "yes"

def ensure_deepseek_ready() -> bool:
    if model != "deepseek":
        return True
    return _ensure_deepseek_ollama_ready(interactive=True)

def main():
    global atexit, model, history, readline
    cfg = load_config()
    if not cfg.get("setup_complete"):
        cfg = run_setup_wizard(cfg)

    enabled_models = [normalize_model(m) for m in cfg.get("enabled_models", [])]
    enabled_models = [m for m in enabled_models if m in MODEL_CMD]
    if not enabled_models:
        enabled_models = [normalize_model(DEFAULT_MODEL)]

    model = normalize_model(cfg.get("default_model", DEFAULT_MODEL))
    if model not in enabled_models:
        model = enabled_models[0]
    if model == "deepseek" and shutil.which("ollama") is None:
        fallback = pick_best_available_model([m for m in enabled_models if m != "deepseek"], preferred=cfg.get("default_model"))
        if fallback and fallback != "deepseek":
            model = fallback
            cfg["default_model"] = model
            save_config(cfg)
            print_status("model", f"deepseek unavailable (ollama missing), switched to {model}", "33")

    smart_mode = True
    dryrun = False
    speed_mode = str(cfg.get("speed_mode", "balanced")).strip().lower()
    if speed_mode not in ("fast", "balanced"):
        speed_mode = "balanced"

    session = None
    if HAVE_PROMPT_TOOLKIT:
        suggestion_enabled = [True]

        kb = KeyBindings()

        @kb.add("tab")
        def _(event):
            buf = event.current_buffer
            if buf.suggestion:
                buf.insert_text(buf.suggestion.text)
            else:
                buf.start_completion(select_first=False)

        @kb.add("escape")
        def _(event):
            suggestion_enabled[0] = not suggestion_enabled[0]
            status = "on" if suggestion_enabled[0] else "off"
            print_status("suggest", status, "34")

        session = PromptSession(
            history=FileHistory(HIST_FILE),
            completer=AIShellCompleter(live_enabled_getter=lambda: suggestion_enabled[0]),
            auto_suggest=AIShellAutoSuggest(enabled_getter=lambda: suggestion_enabled[0]),
            complete_while_typing=True,
            key_bindings=kb,
            style=PT_STYLE,
        )
    else:
        try:
            import atexit as _atexit
            import readline as _readline

            atexit = _atexit
            readline = _readline
            try:
                readline.read_history_file(HIST_FILE)
            except FileNotFoundError:
                pass
            readline.set_history_length(2000)
            atexit.register(readline.write_history_file, HIST_FILE)
            readline.set_completer_delims(" \t\n\"'`@$><=;|&{(")
            readline.parse_and_bind("tab: complete")
            readline.set_completer(readline_completer)
        except Exception:
            readline = None

    print_banner(model, smart_mode)
    check_for_updates(force=False)
    while True:
        cwd = os.getcwd()
        prompt_text = prompt_text_for(model, cwd)
        try:
            if session is not None:
                line = session.prompt(HTML(prompt_html_for(model, cwd))).strip()
            else:
                line = input(prompt_text).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not line:
            continue

        started = time.perf_counter()
        should_exit = False

        if line.startswith("/"):
            parts = line.split()
            cmd = parts[0]

            if cmd == "/help":
                print_help()
            elif cmd == "/model":
                if len(parts) != 2:
                    print(f"Usage: /model [{'|'.join(enabled_models)}]")
                else:
                    requested = normalize_model(parts[1])
                    if requested not in enabled_models:
                        if requested in MODEL_CMD:
                            enabled_models.append(requested)
                            cfg["enabled_models"] = enabled_models
                            cfg["default_model"] = requested
                            save_config(cfg)
                            model = requested
                            print_status("model", f"'{requested}' was auto-enabled", "33")
                            print_model_info(model)
                        else:
                            print_status("model", f"'{requested}' is not enabled. Run /setup to change enabled models.", "31")
                    else:
                        model = requested
                        if model == "deepseek" and shutil.which("ollama") is None:
                            fallback = pick_best_available_model([m for m in enabled_models if m != "deepseek"], preferred=cfg.get("default_model"))
                            if fallback and fallback != "deepseek":
                                print_status("model", "deepseek unavailable (ollama missing)", "33")
                                model = fallback
                                print_status("model", f"switched to {model}", "33")
                        cfg["default_model"] = model
                        save_config(cfg)
                        print_model_info(model)
            elif cmd == "/usage":
                print_model_info(model)
            elif cmd == "/auth":
                target = model
                action = "status"
                args = parts[1:]
                if len(args) == 1:
                    one = args[0].lower()
                    if one in AUTH_ACTIONS:
                        action = one
                    elif normalize_model(one) in ("claude", "gpt", "gemini", "deepseek"):
                        target = normalize_model(one)
                    else:
                        print("Usage: /auth [model] [status|login|logout]")
                        continue
                elif len(args) >= 2:
                    a0 = args[0].lower()
                    a1 = args[1].lower()
                    if normalize_model(a0) in ("claude", "gpt", "gemini", "deepseek") and a1 in AUTH_ACTIONS:
                        target = normalize_model(a0)
                        action = a1
                    elif a0 in AUTH_ACTIONS and normalize_model(a1) in ("claude", "gpt", "gemini", "deepseek"):
                        action = a0
                        target = normalize_model(a1)
                    else:
                        print("Usage: /auth [model] [status|login|logout]")
                        continue
                run_auth_command(target, action)
                print_model_info(target)
            elif cmd == "/ollama":
                action = "status"
                arg = ""
                if len(parts) >= 2:
                    action = parts[1].lower()
                if len(parts) >= 3:
                    arg = " ".join(parts[2:]).strip()
                run_ollama_command(action, arg)
            elif cmd == "/update":
                check_for_updates(force=True)
            elif cmd == "/speed":
                if len(parts) != 2 or parts[1] not in ("fast", "balanced"):
                    print("Usage: /speed [fast|balanced]")
                else:
                    speed_mode = parts[1]
                    cfg["speed_mode"] = speed_mode
                    save_config(cfg)
                    print_status("speed", speed_mode, "32" if speed_mode == "fast" else "34")
            elif cmd == "/fixnpm":
                fix_npm_permissions()
            elif cmd == "/setup":
                cfg = run_setup_wizard(cfg)
                enabled_models = [normalize_model(m) for m in cfg.get("enabled_models", [])]
                enabled_models = [m for m in enabled_models if m in MODEL_CMD]
                if not enabled_models:
                    enabled_models = [normalize_model(DEFAULT_MODEL)]
                if model not in enabled_models:
                    model = normalize_model(cfg.get("default_model", enabled_models[0]))
                    if model not in enabled_models:
                        model = enabled_models[0]
                if model == "deepseek" and shutil.which("ollama") is None:
                    fallback = pick_best_available_model([m for m in enabled_models if m != "deepseek"], preferred=cfg.get("default_model"))
                    if fallback and fallback != "deepseek":
                        model = fallback
                        cfg["default_model"] = model
                        save_config(cfg)
                        print_status("model", f"deepseek unavailable (ollama missing), switched to {model}", "33")
                speed_mode = str(cfg.get("speed_mode", speed_mode)).strip().lower()
                if speed_mode not in ("fast", "balanced"):
                    speed_mode = "balanced"
            elif cmd == "/clear":
                history = []
                print_status("cleared", "chat context", "34")
            elif cmd == "/history":
                turns = FAST_TURNS if speed_mode == "fast" else MAX_TURNS
                ctx = history[-2 * turns:]
                if not ctx:
                    print("[empty]")
                else:
                    for r, t in ctx:
                        t2 = t[:400] + ("..." if len(t) > 400 else "")
                        print(f"{r}: {t2}\n")
            elif cmd == "/save":
                save_transcript()
            elif cmd == "/smart":
                if len(parts) != 2 or parts[1] not in ("on", "off"):
                    print("Usage: /smart [on|off]")
                else:
                    smart_mode = (parts[1] == "on")
                    print_status("smart", "on" if smart_mode else "off", "32" if smart_mode else "31")
            elif cmd == "/dryrun":
                if len(parts) != 2 or parts[1] not in ("on", "off"):
                    print("Usage: /dryrun [on|off]")
                else:
                    dryrun = (parts[1] == "on")
                    print_status("dryrun", "on" if dryrun else "off", "33")
            elif cmd == "/exit":
                should_exit = True
            else:
                print("Unknown command. Try /help")

        elif line.startswith("!"):
            # Force shell if prefixed with !
            run_shell(line[1:].lstrip())

        elif is_shell_command(line):
            # Decide shell vs AI
            partial = shell_partial_suggestions(line)
            if partial:
                print_status("suggest", ", ".join(partial), "34")
            else:
                run_shell(line)
        else:
            # AI path (persistent chat)
            if model == "deepseek" and not ensure_deepseek_ready():
                print_status("auth", "DeepSeek is not ready. Skipping request.", "31")
                elapsed = time.perf_counter() - started
                print_status("response-time", f"{elapsed:.3f}s", "90")
                continue
            explanation_first = should_prefer_explanation(line)
            turns = FAST_TURNS if speed_mode == "fast" else MAX_TURNS
            prompt = build_prompt(
                line,
                smart_mode=(smart_mode and not explanation_first),
                max_turns=turns,
                fast_mode=(speed_mode == "fast"),
            )
            raw = call_model(prompt)
            if looks_like_usage_exhausted(raw):
                print_status("usage", f"{model} may have exhausted plan quota or hit a rate limit.", "31")
            suggested_cmd, cleaned = extract_cmd(raw)

            # Print AI response (without CMD line)
            if cleaned:
                print(color(cleaned, "37"))

            # Store history
            history.append(("user", line))
            history.append(("assistant", raw))

            # If AI suggested a command, run policy by model.
            if suggested_cmd and not explanation_first:
                # basic safety: refuse multiline (shouldn't happen, but guard)
                if "\n" in suggested_cmd or "\r" in suggested_cmd:
                    print_status("blocked", "Suggested command contained newlines. Not running.", "31")
                elif dryrun:
                    print_status("dryrun", f"Command suggested (not executed): {suggested_cmd}", "33")
                elif model in ("claude", "gemini"):
                    run_shell(suggested_cmd)
                elif confirm_run(suggested_cmd):
                    run_shell(suggested_cmd)
                else:
                    print_status("ok", "Not executed.", "34")

        elapsed = time.perf_counter() - started
        print_status("response-time", f"{elapsed:.3f}s", "90")
        if should_exit:
            break

if __name__ == "__main__":
    main()
