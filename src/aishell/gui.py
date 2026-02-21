#!/usr/bin/env python3
import os
import shlex
import subprocess
import shutil
import threading
import time
import base64
import json
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional
from tkinter import BOTH, END, LEFT, RIGHT, VERTICAL, X, Y, BooleanVar, StringVar, Tk, ttk, filedialog, messagebox, PhotoImage
from tkinter.scrolledtext import ScrolledText

from . import cli


class AIShellGUI:
    def __init__(self, root: Tk):
        self.root = root
        self.root.title("AI Shell Desktop")
        self.root.geometry("1280x820")
        self.root.minsize(1000, 640)
        self.icon_image = None
        self._set_window_icon()

        self.cfg = cli.load_config()
        if not self.cfg.get("setup_complete"):
            self.cfg = {
                "setup_complete": True,
                "enabled_models": ["gpt", "claude", "gemini", "deepseek"],
                "default_model": cli.normalize_model(os.environ.get("AI_DEFAULT", "gpt")),
            }
            cli.save_config(self.cfg)

        self.enabled_models = [cli.normalize_model(m) for m in self.cfg.get("enabled_models", []) if m in cli.MODEL_CMD]
        if not self.enabled_models:
            self.enabled_models = ["gpt", "claude", "gemini", "deepseek"]

        self.model_var = StringVar(value=cli.normalize_model(self.cfg.get("default_model", self.enabled_models[0])))
        if self.model_var.get() not in self.enabled_models:
            self.model_var.set(self.enabled_models[0])

        self.smart_var = BooleanVar(value=True)
        self.dryrun_var = BooleanVar(value=False)
        self.autorun_cmd_var = BooleanVar(value=True)
        self.screen_ctx_var = BooleanVar(value=False)

        self.cwd = os.getcwd()
        self.tree_root = self.cwd
        self.history = []  # list[(role, text)]
        self.backend_meta = self._detect_backend_metadata()
        self.input_start_index = "1.0"
        self.input_locked = False
        self.tab_state = None

        self._build_ui()
        self._load_tree_root(self.tree_root)
        self._append_status("ready", f"GUI started at {self.cwd}")
        self._print_prompt()

    def _build_ui(self):
        top = ttk.Frame(self.root, padding=(10, 8))
        top.pack(fill=X)

        ttk.Label(top, text="Model").pack(side=LEFT)
        self.model_combo = ttk.Combobox(top, values=self.enabled_models, width=10, textvariable=self.model_var, state="readonly")
        self.model_combo.pack(side=LEFT, padx=(6, 12))
        self.model_combo.bind("<<ComboboxSelected>>", lambda _e: self._on_model_change())

        ttk.Checkbutton(top, text="Smart", variable=self.smart_var).pack(side=LEFT, padx=(0, 10))
        ttk.Checkbutton(top, text="Dry Run", variable=self.dryrun_var).pack(side=LEFT, padx=(0, 10))
        ttk.Checkbutton(top, text="Auto-run CMD", variable=self.autorun_cmd_var).pack(side=LEFT, padx=(0, 16))
        ttk.Checkbutton(top, text="Use Screen Context", variable=self.screen_ctx_var).pack(side=LEFT, padx=(0, 10))
        ttk.Button(top, text="Capture Now", command=self._capture_now).pack(side=LEFT, padx=(0, 10))

        ttk.Button(top, text="Choose Root", command=self._choose_root).pack(side=LEFT, padx=(0, 8))
        ttk.Button(top, text="Refresh Tree", command=lambda: self._load_tree_root(self.tree_root)).pack(side=LEFT, padx=(0, 8))
        ttk.Button(top, text="Save Transcript", command=self._save_transcript).pack(side=LEFT, padx=(0, 8))
        ttk.Button(top, text="Clear Chat", command=self._clear_chat).pack(side=LEFT)

        self.cwd_label = ttk.Label(top, text=f"CWD: {self.cwd}")
        self.cwd_label.pack(side=RIGHT)
        self.identity_label = ttk.Label(top, text=self._identity_text())
        self.identity_label.pack(side=RIGHT, padx=(0, 16))

        split = ttk.Panedwindow(self.root, orient="horizontal")
        split.pack(fill=BOTH, expand=True, padx=10, pady=(0, 10))

        left = ttk.Frame(split, width=330)
        right = ttk.Frame(split)
        split.add(left, weight=1)
        split.add(right, weight=3)

        ttk.Label(left, text="Files", font=("SF Pro Text", 12, "bold")).pack(anchor="w", pady=(4, 6))
        tree_wrap = ttk.Frame(left)
        tree_wrap.pack(fill=BOTH, expand=True)

        self.tree = ttk.Treeview(tree_wrap, show="tree")
        yscroll = ttk.Scrollbar(tree_wrap, orient=VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=yscroll.set)
        self.tree.pack(side=LEFT, fill=BOTH, expand=True)
        yscroll.pack(side=RIGHT, fill=Y)
        self.tree.bind("<<TreeviewOpen>>", self._on_tree_open)
        self.tree.bind("<Double-1>", self._on_tree_double_click)

        ttk.Label(right, text="Conversation", font=("SF Pro Text", 12, "bold")).pack(anchor="w", pady=(4, 6))
        self.suggest_label = ttk.Label(right, text="Suggestion: ", foreground="#64748B")
        self.suggest_label.pack(anchor="w", pady=(0, 4))
        self.output = ScrolledText(right, wrap="word", height=24)
        self.output.pack(fill=BOTH, expand=True)
        # Fixed high-contrast colors so text remains readable regardless of system theme.
        self.output.configure(background="#0F172A", foreground="#E2E8F0", insertbackground="#E2E8F0")
        self.output.tag_configure("user", foreground="#93C5FD", font=("Menlo", 11, "bold"))
        self.output.tag_configure("assistant", foreground="#E2E8F0", font=("Menlo", 11))
        self.output.tag_configure("status", foreground="#94A3B8", font=("Menlo", 10, "italic"))
        self.output.tag_configure("error", foreground="#FCA5A5", font=("Menlo", 10, "bold"))
        self.output.tag_configure("cmd", foreground="#86EFAC", font=("Menlo", 10, "bold"))
        self.output.bind("<Return>", self._on_enter)
        self.output.bind("<BackSpace>", self._on_backspace)
        self.output.bind("<Left>", self._on_left)
        self.output.bind("<Home>", self._on_home)
        self.output.bind("<Button-1>", self._on_click)
        self.output.bind("<Tab>", self._on_tab_complete)
        self.output.bind("<KeyRelease>", self._on_key_release)
        self.output.focus_set()

    def _append(self, text: str, tag: str):
        self.output.insert(END, text + "\n", tag)
        self.output.see(END)

    def _append_status(self, tag: str, msg: str):
        self._append(f"[{tag}] {msg}", "status")

    def _append_error(self, msg: str):
        self._append(f"[error] {msg}", "error")

    def _append_cmd(self, cmd: str):
        self._append(f"$ {cmd}", "cmd")

    def _set_window_icon(self):
        # Draw a small terminal-like icon directly in Tk so packaging stays simple.
        img = PhotoImage(width=64, height=64)
        img.put("#0B1220", to=(0, 0, 64, 64))
        img.put("#14B8A6", to=(0, 0, 64, 10))
        img.put("#F8FAFC", to=(8, 4, 14, 8))
        img.put("#F8FAFC", to=(18, 4, 24, 8))
        img.put("#F8FAFC", to=(28, 4, 34, 8))
        # Prompt chevron
        img.put("#60A5FA", to=(16, 22, 20, 26))
        img.put("#60A5FA", to=(20, 26, 24, 30))
        img.put("#60A5FA", to=(16, 30, 20, 34))
        # Cursor block
        img.put("#A3E635", to=(30, 28, 46, 34))
        self.icon_image = img
        try:
            self.root.iconphoto(True, self.icon_image)
        except Exception:
            pass

    def _prompt_prefix(self):
        return f"[{self.model_var.get()}] {self.cwd} » "

    def _print_prompt(self):
        self.output.insert("end-1c", self._prompt_prefix(), "status")
        self.input_start_index = self.output.index("end-1c")
        self.output.mark_set("insert", "end-1c")
        self.output.see(END)

    def _current_input(self):
        return self.output.get(self.input_start_index, "end-1c")

    def _set_locked(self, locked: bool):
        self.input_locked = locked
        if not locked:
            self.tab_state = None

    def _on_click(self, _event):
        self.root.after(1, lambda: self.output.mark_set("insert", "end-1c"))
        return None

    def _on_home(self, _event):
        self.output.mark_set("insert", self.input_start_index)
        return "break"

    def _on_left(self, _event):
        if self.output.compare("insert", "<=", self.input_start_index):
            return "break"
        return None

    def _on_backspace(self, _event):
        if self.input_locked or self.output.compare("insert", "<=", self.input_start_index):
            return "break"
        return None

    def _on_enter(self, _event):
        if self.input_locked:
            return "break"
        line = self._current_input().strip()
        self.output.insert(END, "\n")
        if not line:
            self._print_prompt()
            return "break"
        self._set_locked(True)
        threading.Thread(target=self._process_line, args=(line,), daemon=True).start()
        return "break"

    def _set_current_input(self, new_line: str):
        self.output.delete(self.input_start_index, "end-1c")
        self.output.insert("end-1c", new_line)
        self.output.mark_set("insert", "end-1c")
        self.output.see(END)

    def _split_completion_target(self, line: str):
        if not line:
            return "", 0
        if line.endswith(" "):
            return "", len(line)
        begidx = line.rfind(" ") + 1
        return line[begidx:], begidx

    def _update_suggestion_label(self):
        line = self._current_input()
        text, begidx = self._split_completion_target(line)
        options = cli.build_completions(line, text, begidx)[:5]
        if options:
            self.suggest_label.configure(text=f"Suggestion: {', '.join(options)}")
        else:
            self.suggest_label.configure(text="Suggestion: ")

    def _on_key_release(self, event):
        if event.keysym in ("Tab", "Return", "Shift_L", "Shift_R", "Control_L", "Control_R", "Alt_L", "Alt_R"):
            return
        self.tab_state = None
        self._update_suggestion_label()

    def _on_tab_complete(self, _event):
        if self.input_locked:
            return "break"
        line = self._current_input()
        text, begidx = self._split_completion_target(line)
        options = cli.build_completions(line, text, begidx)
        if not options:
            self.root.bell()
            return "break"

        if (
            self.tab_state
            and self.tab_state.get("line") == line
            and self.tab_state.get("options") == options
            and len(options) > 1
        ):
            idx = (self.tab_state["idx"] + 1) % len(options)
        else:
            idx = 0

        chosen = options[idx]
        new_line = line[:begidx] + chosen
        self._set_current_input(new_line)
        self.tab_state = {"line": new_line, "options": options, "idx": idx}
        self._update_suggestion_label()
        return "break"

    def _identity_text(self):
        return f"GPT: {self.backend_meta.get('gpt_model', 'unknown')} | User: {self.backend_meta.get('username', 'unknown')}"

    def _refresh_identity_label(self):
        if hasattr(self, "identity_label"):
            self.identity_label.configure(text=self._identity_text())

    def _decode_jwt_payload(self, token: str):
        parts = token.split(".")
        if len(parts) < 2:
            return {}
        payload = parts[1]
        padding = "=" * (-len(payload) % 4)
        try:
            decoded = base64.urlsafe_b64decode(payload + padding).decode("utf-8")
            obj = json.loads(decoded)
            return obj if isinstance(obj, dict) else {}
        except Exception:
            return {}

    def _detect_backend_metadata(self):
        meta = {"gpt_model": "unknown", "username": "unknown"}
        codex_cfg = Path.home() / ".codex" / "config.toml"
        codex_auth = Path.home() / ".codex" / "auth.json"

        try:
            if codex_cfg.exists():
                for ln in codex_cfg.read_text(encoding="utf-8").splitlines():
                    s = ln.strip()
                    if s.startswith("model"):
                        _, rhs = s.split("=", 1)
                        model_name = rhs.strip().strip('"').strip("'")
                        if model_name:
                            meta["gpt_model"] = model_name
                        break
        except Exception:
            pass

        try:
            if codex_auth.exists():
                auth = json.loads(codex_auth.read_text(encoding="utf-8"))
                token = ((auth.get("tokens") or {}).get("id_token") or "").strip()
                claims = self._decode_jwt_payload(token) if token else {}
                email = (claims.get("email") or "").strip()
                if email:
                    if email.startswith("email4") and "@" in email:
                        email = email[len("email4"):]
                    meta["username"] = email
        except Exception:
            pass

        return meta

    def _on_model_change(self):
        selected = cli.normalize_model(self.model_var.get())
        self.model_var.set(selected)
        self.cfg["default_model"] = selected
        cli.save_config(self.cfg)
        self.backend_meta = self._detect_backend_metadata()
        self._refresh_identity_label()
        self._append_status("model", selected)

    def _capture_screen(self):
        tmp = tempfile.NamedTemporaryFile(prefix="aishell-screen-", suffix=".png", delete=False)
        out = tmp.name
        tmp.close()
        platform = os.uname().sysname
        candidates = []
        if platform == "Darwin":
            candidates.append(["screencapture", "-x", out])
        else:
            # Common Linux screenshot commands (Ubuntu and Wayland/X11 variants).
            candidates.extend([
                ["gnome-screenshot", "-f", out],
                ["grim", out],
                ["scrot", out],
                ["maim", out],
                ["import", "-window", "root", out],
            ])

        p = None
        used = None
        for cmd in candidates:
            if shutil.which(cmd[0]) is None:
                continue
            p = subprocess.run(cmd, text=True, capture_output=True)
            used = cmd[0]
            if p.returncode == 0 and os.path.exists(out):
                break
        if p is None or p.returncode != 0 or not os.path.exists(out):
            err = ""
            if p is not None:
                err = (p.stderr or p.stdout or "").strip()
            if not err:
                if platform == "Darwin":
                    err = "Screen capture failed (check macOS Screen Recording permission)."
                else:
                    err = (
                        "Screen capture failed. Install one of: gnome-screenshot, grim, "
                        "scrot, maim, or imagemagick (import)."
                    )
            try:
                if os.path.exists(out):
                    os.remove(out)
            except OSError:
                pass
            return None, err
        if used is not None:
            self._append_status("screen-tool", used)
        return out, None

    def _capture_now(self):
        path, err = self._capture_screen()
        if err:
            self._append_error(err)
        else:
            self._append_status("screen", "captured and discarded")
            try:
                os.remove(path)
            except OSError:
                pass

    def _clear_chat(self):
        self.history = []
        self.output.delete("1.0", END)
        self._append_status("cleared", "chat context")
        self._print_prompt()
        self._update_suggestion_label()

    def _save_transcript(self):
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        out = cli.STATE_DIR / f"transcript.{ts}.{self.model_var.get()}.txt"
        with out.open("w", encoding="utf-8") as f:
            for role, text in self.history:
                f.write(f"{role.upper()}: {text}\n\n")
        self._append_status("saved", str(out))

    def _choose_root(self):
        selected = filedialog.askdirectory(initialdir=self.tree_root, title="Choose file tree root")
        if not selected:
            return
        self.tree_root = selected
        self._load_tree_root(self.tree_root)

    def _load_tree_root(self, root_path: str):
        self.tree.delete(*self.tree.get_children(""))
        root_id = self.tree.insert("", END, text=root_path, values=[root_path], open=True)
        self._insert_children(root_id, root_path)

    def _insert_children(self, parent_id: str, path: str):
        try:
            items = sorted(os.listdir(path), key=lambda name: (not os.path.isdir(os.path.join(path, name)), name.lower()))
        except OSError:
            return

        for name in items:
            full = os.path.join(path, name)
            label = f"{name}/" if os.path.isdir(full) else name
            node = self.tree.insert(parent_id, END, text=label, values=[full])
            if os.path.isdir(full):
                self.tree.insert(node, END, text="...")

    def _on_tree_open(self, _event):
        node = self.tree.focus()
        vals = self.tree.item(node, "values")
        if not vals:
            return
        full = vals[0]
        children = self.tree.get_children(node)
        if len(children) == 1 and self.tree.item(children[0], "text") == "...":
            self.tree.delete(children[0])
            self._insert_children(node, full)

    def _on_tree_double_click(self, _event):
        node = self.tree.focus()
        vals = self.tree.item(node, "values")
        if not vals:
            return
        full = vals[0]
        if os.path.isdir(full):
            self.cwd = full
            self.cwd_label.configure(text=f"CWD: {self.cwd}")
            self._append_status("cwd", self.cwd)
        else:
            self._append_status("file", full)

    def _process_line(self, line: str):
        started = time.perf_counter()
        model = cli.normalize_model(self.model_var.get())
        screen_image = None

        def ui(fn, *args):
            self.root.after(0, fn, *args)

        if line.startswith("/"):
            ui(self._handle_slash, line)
            elapsed = time.perf_counter() - started
            ui(self._append_status, "response-time", f"{elapsed:.3f}s")
            ui(self._set_locked, False)
            ui(self._print_prompt)
            return

        if line.startswith("!"):
            out = self._run_shell_capture(line[1:].lstrip())
            ui(self._append, out, "assistant")
            elapsed = time.perf_counter() - started
            ui(self._append_status, "response-time", f"{elapsed:.3f}s")
            ui(self._set_locked, False)
            ui(self._print_prompt)
            return

        if self._is_shell_command(line):
            partial = cli.shell_partial_suggestions(line)
            if partial:
                ui(self._append_status, "suggest", ", ".join(partial))
            else:
                out = self._run_shell_capture(line)
                ui(self._append, out, "assistant")
            elapsed = time.perf_counter() - started
            ui(self._append_status, "response-time", f"{elapsed:.3f}s")
            ui(self._set_locked, False)
            ui(self._print_prompt)
            return

        if self.screen_ctx_var.get():
            if model == "gpt":
                screen_image, err = self._capture_screen()
                if err:
                    ui(self._append_error, err)
                else:
                    ui(self._append_status, "screen", f"attached: {screen_image}")
            else:
                ui(self._append_status, "screen", "screen context currently supported for GPT only")

        explanation_first = cli.should_prefer_explanation(line)
        prompt = self._build_prompt(line, smart_mode=(self.smart_var.get() and not explanation_first))
        raw = self._call_model(model, prompt, image_path=screen_image)
        if screen_image and os.path.exists(screen_image):
            try:
                os.remove(screen_image)
            except OSError:
                pass
        suggested_cmd, cleaned = cli.extract_cmd(raw)

        if cleaned:
            ui(self._append, cleaned, "assistant")
            if model == "gpt":
                ui(
                    self._append_status,
                    "backend",
                    f"model={self.backend_meta.get('gpt_model', 'unknown')} user={self.backend_meta.get('username', 'unknown')}",
                )

        self.history.append(("user", line))
        self.history.append(("assistant", raw))

        if suggested_cmd and not explanation_first:
            if "\n" in suggested_cmd or "\r" in suggested_cmd:
                ui(self._append_error, "Suggested command contained newlines. Not running.")
            elif self.dryrun_var.get():
                ui(self._append_status, "dryrun", f"Command suggested (not executed): {suggested_cmd}")
            elif model == "claude" or self.autorun_cmd_var.get():
                ui(self._append_cmd, suggested_cmd)
                out = self._run_shell_capture(suggested_cmd)
                ui(self._append, out, "assistant")
            else:
                should_run = self._ask_yes_no(f"Run suggested command?\n\n{suggested_cmd}")
                if should_run:
                    ui(self._append_cmd, suggested_cmd)
                    out = self._run_shell_capture(suggested_cmd)
                    ui(self._append, out, "assistant")
                else:
                    ui(self._append_status, "ok", "Not executed")

        elapsed = time.perf_counter() - started
        ui(self._append_status, "response-time", f"{elapsed:.3f}s")
        ui(self._set_locked, False)
        ui(self._print_prompt)

    def _ask_yes_no(self, message: str):
        result = {"value": False}
        event = threading.Event()

        def ask():
            result["value"] = messagebox.askyesno("AI Shell", message)
            event.set()

        self.root.after(0, ask)
        event.wait()
        return result["value"]

    def _handle_slash(self, line: str):
        parts = line.split()
        cmd = parts[0]

        if cmd == "/help":
            self._append(
                "Commands: /help, /model <name>, /clear, /history, /save, /smart on|off, /dryrun on|off, /exit",
                "status",
            )
            return

        if cmd == "/model":
            if len(parts) != 2:
                self._append_error(f"Usage: /model {'|'.join(self.enabled_models)}")
                return
            req = cli.normalize_model(parts[1])
            if req not in self.enabled_models:
                self._append_error(f"Model '{req}' is not enabled")
                return
            self.model_var.set(req)
            self._on_model_change()
            return

        if cmd == "/clear":
            self._clear_chat()
            return

        if cmd == "/history":
            ctx = self.history[-2 * cli.MAX_TURNS:]
            if not ctx:
                self._append_status("history", "empty")
            else:
                for role, text in ctx:
                    t2 = text[:400] + ("..." if len(text) > 400 else "")
                    self._append(f"{role}: {t2}", "status")
            return

        if cmd == "/save":
            self._save_transcript()
            return

        if cmd == "/smart":
            if len(parts) != 2 or parts[1] not in ("on", "off"):
                self._append_error("Usage: /smart [on|off]")
            else:
                self.smart_var.set(parts[1] == "on")
                self._append_status("smart", parts[1])
            return

        if cmd == "/dryrun":
            if len(parts) != 2 or parts[1] not in ("on", "off"):
                self._append_error("Usage: /dryrun [on|off]")
            else:
                self.dryrun_var.set(parts[1] == "on")
                self._append_status("dryrun", parts[1])
            return

        if cmd == "/setup":
            self._append_status("setup", "Use terminal 'aishell' for interactive setup wizard.")
            return

        if cmd == "/exit":
            self.root.destroy()
            return

        self._append_error("Unknown command")

    def _build_prompt(self, user_text: str, smart_mode: bool):
        ctx = self.history[-2 * cli.MAX_TURNS:]

        system = [
            "You are an assistant inside a terminal for a PhD workflow.",
            "Be concise and practical.",
            "If you propose terminal commands, prefer safe, non-destructive commands.",
            "Never include secrets or tokens.",
        ]

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
            ]

        lines = []
        lines.extend(system)
        lines.append("")
        for r, t in ctx:
            lines.append(("User: " if r == "user" else "Assistant: ") + t)
        lines.append(f"User: {user_text}")
        lines.append("Assistant:")
        return "\n".join(lines)

    def _call_model(self, model: str, prompt: str, image_path: Optional[str] = None):
        cmd = cli.MODEL_CMD.get(model)
        if not cmd:
            return f"[error] unknown model: {model}"
        try:
            if model == "gpt":
                gpt_cmd = cmd + ["--cd", self.cwd]
                if image_path:
                    gpt_cmd += ["-i", image_path]
                p = subprocess.run(
                    gpt_cmd + ["-"],
                    input=prompt,
                    text=True,
                    capture_output=True,
                    timeout=45,
                )
            else:
                p = subprocess.run(cmd + [prompt], cwd=self.cwd, text=True, capture_output=True)
        except FileNotFoundError:
            return f"[error] '{cmd[0]}' not found on PATH."
        except subprocess.TimeoutExpired:
            return "[error] gpt command timed out after 45 seconds."

        out = (p.stdout or "").strip()
        err = (p.stderr or "").strip()
        if not out and err:
            out = err
        return out if out else "[no output returned]"

    def _is_shell_command(self, line: str):
        if line.startswith("!"):
            return True

        if cli.looks_like_natural_language(line):
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
        return first in cli.SHELL_BUILTINS or shutil.which(first) is not None

    def _run_shell_capture(self, line: str):
        line = line.strip()
        if not line:
            return ""

        if line == "cd" or line.startswith("cd "):
            parts = shlex.split(line)
            target = os.path.expanduser("~") if len(parts) == 1 else os.path.expanduser(parts[1])
            try:
                os.chdir(target)
                self.cwd = os.getcwd()
                self.root.after(0, lambda: self.cwd_label.configure(text=f"CWD: {self.cwd}"))
                return self.cwd
            except FileNotFoundError:
                return f"cd: no such file or directory: {target}"
            except NotADirectoryError:
                return f"cd: not a directory: {target}"

        if line == "pwd":
            return self.cwd

        try:
            p = subprocess.run(["bash", "-lc", line], cwd=self.cwd, text=True, capture_output=True)
        except Exception as exc:
            return f"[shell error] {exc}"

        out = (p.stdout or "").rstrip()
        err = (p.stderr or "").rstrip()
        parts = []
        if out:
            parts.append(out)
        if err:
            parts.append(err)
        parts.append(f"[exit {p.returncode}]")
        return "\n".join(parts)


def main():
    root = Tk()
    # Use Aqua-native theme where available (macOS).
    style = ttk.Style(root)
    try:
        if os.uname().sysname == "Darwin":
            style.theme_use("aqua")
    except Exception:
        pass
    AIShellGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
