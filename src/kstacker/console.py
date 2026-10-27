"""Compact, colour-aware reporting for terminals and batch-job logs.

The goal is to make long scientific runs easy to follow without turning the
console into a wall of text. Interactive terminals receive restrained colours;
redirected logs remain plain and readable.
"""
from __future__ import annotations

import os
import sys
from tqdm import tqdm as _tqdm

COLORS = {
    "cyan": "1;36",
    "blue": "1;34",
    "green": "1;32",
    "yellow": "1;33",
    "magenta": "1;35",
    "red": "1;31",
    "muted": "2",
    "white": "1;37",
}

TONE_ICON = {
    "cyan": "•",
    "blue": "→",
    "green": "✓",
    "yellow": "!",
    "magenta": "◆",
    "red": "✗",
    "muted": "·",
    "white": "•",
}


def _terminal_colors_enabled() -> bool:
    """Respect NO_COLOR, TTY detection and the KSTACKER_COLOR override."""
    choice = str(os.environ.get("KSTACKER_COLOR", "auto")).strip().lower()
    if choice in {"never", "no", "0", "false"}:
        return False
    if choice in {"always", "yes", "1", "true"}:
        return True
    return "NO_COLOR" not in os.environ and bool(sys.stdout.isatty())


def _color(value: str, code: str) -> str:
    """Apply ANSI styling only when supported by the current output stream."""
    return f"\033[{code}m{value}\033[0m" if _terminal_colors_enabled() else value


def _print_section(title: str, width: int = 78, *, tone: str = "cyan") -> None:
    """Write a compact section header for a new stage of the workflow."""
    code = COLORS.get(tone, COLORS["cyan"])
    print("\n" + _color("┏" + "━" * (width - 2) + "┓", code), flush=True)
    print(_color("┃ " + title.ljust(width - 4) + " ┃", code), flush=True)
    print(_color("┗" + "━" * (width - 2) + "┛", code), flush=True)


def _progress_mode() -> str:
    value = os.environ.get("KSTACKER_PROGRESS", "auto").lower()
    if value not in {"auto", "terminal", "log", "none"}:
        raise ValueError("KSTACKER_PROGRESS must be auto, terminal, log or none")
    return ("terminal" if sys.stderr.isatty() else "log") if value == "auto" else value


class ProgressReporter:
    """Emit periodic full-line updates for long computations in redirected logs."""

    def __init__(self, label: str, total: int, *, enabled: bool = True) -> None:
        self.label, self.total, self.enabled = label, max(1, int(total)), enabled
        self.last_percent = -1

    def update(self, done: int, *, detail: str = "") -> None:
        """Print at most one line every five percentage points."""
        if not self.enabled:
            return
        percent = min(100, 100 * int(done) // self.total)
        if percent >= self.last_percent + 5 or int(done) >= self.total:
            self.last_percent = percent
            extra = f" | {detail}" if detail else ""
            print(f"[{self.label}] {percent:3d}% ({done:,}/{self.total:,}){extra}", flush=True)


class _LogProgress:
    """Iterable with a tqdm-compatible postfix for line-oriented job logs."""

    def __init__(self, iterable, *, desc="Progress", total=None, disable=False, **kwargs):
        self.iterable = iterable
        self.label = desc
        self.total = total
        self.reporter = ProgressReporter(desc, total or 1, enabled=not disable and total is not None)
        self.postfix = ""

    def __iter__(self):
        for i, value in enumerate(self.iterable, start=1):
            yield value
            self.reporter.update(i, detail=self.postfix)

    def set_postfix(self, ordered_dict=None, *, refresh=False, **kwargs):
        values = dict(ordered_dict or {}, **kwargs)
        self.postfix = ", ".join(f"{k}={v}" for k, v in values.items())

    def close(self):
        pass


def console_progress(iterable=None, **kwargs):
    """Display tqdm on terminals and compact progress lines in batch jobs."""
    mode = _progress_mode()
    if mode == "none":
        kwargs["disable"] = True
    if mode in {"log", "none"}:
        return _LogProgress(iterable, **kwargs)
    return _tqdm(iterable, **kwargs)


def interactive_progress(enabled: bool) -> bool:
    """Indicate whether sampler-native tqdm control sequences are appropriate."""
    return bool(enabled) and _progress_mode() == "terminal"


def console_status(label: str, message: str, *, tone: str = "cyan") -> None:
    """Print one concise status line with a visual tone and icon."""
    code = COLORS.get(tone, COLORS["cyan"])
    icon = TONE_ICON.get(tone, TONE_ICON["cyan"])
    badge = _color(f"[{label}]", code)
    prefix = _color(icon, code)
    print(f"{prefix} {badge} {message}", flush=True)


def console_success(message: str, *, label: str = "done") -> None:
    """Print a short success message in green."""
    console_status(label, message, tone="green")


def console_warning(message: str, *, label: str = "warning") -> None:
    """Print a short warning message in yellow."""
    console_status(label, message, tone="yellow")


def console_error(message: str, *, label: str = "error") -> None:
    """Print a short error message in red."""
    console_status(label, message, tone="red")
