"""Per-run logger.

Every invocation of FolderAngel (GUI launch *and* each Organize run, plus
each CLI ``--cli`` call) gets a fresh timestamped log file under
``~/.folderangel/logs/``.  We attach a ``logging.FileHandler`` to the root
logger so every module's existing ``log.warning(...)`` / ``log.info(...)``
call lands there, plus we install ``sys.excepthook`` so unhandled
exceptions are captured with a full stack trace.

The module also exposes :func:`current_log_path` for the UI to surface a
"로그 파일 열기" button.
"""
from __future__ import annotations

import datetime as _dt
import logging
import os
import re
import sys
import threading
import traceback
from pathlib import Path
from typing import Optional

from .config import default_paths

_lock = threading.Lock()
_active_handler: Optional[logging.Handler] = None
_active_path: Optional[Path] = None
_install_count = 0


# Patterns we MUST never write to disk.  Hits anywhere in a log record
# (message, args, exception text) are replaced before reaching the file.
_SECRET_PATTERNS: tuple[tuple[re.Pattern, str], ...] = (
    # Google AI Studio API keys: "AIza" + 35 chars
    (re.compile(r"AIza[0-9A-Za-z_\-]{35}"), "[REDACTED_KEY]"),
    # OpenAI / generic "sk-…" tokens (covers sk-, sk-proj-, sk-ant-, …)
    (re.compile(r"\bsk-[A-Za-z0-9_\-]{16,}"), "[REDACTED_KEY]"),
    # Bearer tokens in Authorization headers
    (
        re.compile(r"(?i)Authorization:\s*Bearer\s+[A-Za-z0-9_\-.=]+"),
        "Authorization: Bearer [REDACTED]",
    ),
    # ?key=… / &key=… query strings (Google APIs put the key here)
    (re.compile(r"(?i)([?&]key=)[A-Za-z0-9_\-]+"), r"\1[REDACTED]"),
    # Generic api_key=… / api-key=… / X-Api-Key: …
    (
        re.compile(r"(?i)(api[_\-]?key\s*[:=]\s*['\"]?)[A-Za-z0-9_\-]{12,}"),
        r"\1[REDACTED]",
    ),
    # Long hex tokens (matches the local llama-server style)
    (re.compile(r"\b[0-9a-f]{48,}\b"), "[REDACTED_KEY]"),
)


def _redact(text: str) -> str:
    if not text or not isinstance(text, str):
        return text
    out = text
    for pat, replacement in _SECRET_PATTERNS:
        out = pat.sub(replacement, out)
    return out


class _RedactingFormatter(logging.Formatter):
    """Strips API keys / bearer tokens from every log record before
    they land in the file.  Defence-in-depth against accidental leaks
    even when a caller logs an Authorization header verbatim.
    """

    def format(self, record: logging.LogRecord) -> str:
        return _redact(super().format(record))


def _format_handler() -> logging.Formatter:
    return _RedactingFormatter(
        fmt="%(asctime)s.%(msecs)03d %(levelname)-7s %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _silence_chatty_third_parties() -> None:
    """Demote loggers that tend to dump URLs (containing ``?key=``) and
    full request bodies at DEBUG level to WARNING.  Defence-in-depth on
    top of the formatter-level redaction.
    """
    for name in ("urllib3", "urllib3.connectionpool", "requests", "PIL"):
        logging.getLogger(name).setLevel(logging.WARNING)


def start_session(tag: str = "session") -> Path:
    """Open a fresh log file for this run and return its path.

    Calling ``start_session`` again rotates to a new file.  Idempotent under
    threads — only one handler is ever attached.
    """
    global _active_handler, _active_path, _install_count
    with _lock:
        paths = default_paths()
        paths.ensure()
        stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = paths.logs_dir / f"{tag}_{stamp}.log"

        # Remove the previously installed handler so we don't double-write
        # to a stale file from an earlier run.
        root = logging.getLogger()
        if _active_handler is not None:
            try:
                root.removeHandler(_active_handler)
                _active_handler.close()
            except Exception:
                pass

        handler = logging.FileHandler(log_file, encoding="utf-8")
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(_format_handler())
        # Make sure the root logger lets DEBUG through to our file even if
        # the existing console handler is at INFO/WARNING.
        if root.level > logging.DEBUG:
            root.setLevel(logging.DEBUG)
        root.addHandler(handler)

        # Also turn on full tracebacks for unhandled exceptions.
        if _install_count == 0:
            previous = sys.excepthook

            def _hook(exc_type, exc, tb):
                logging.getLogger("folderangel.crash").error(
                    "Unhandled exception:\n%s",
                    "".join(traceback.format_exception(exc_type, exc, tb)),
                )
                previous(exc_type, exc, tb)

            sys.excepthook = _hook
            _install_count += 1

        _silence_chatty_third_parties()
        _active_handler = handler
        _active_path = log_file
        logging.getLogger("folderangel.runlog").info(
            "log session started: %s (pid=%d, python=%s)",
            log_file, os.getpid(), sys.version.split()[0],
        )
        return log_file


def current_log_path() -> Optional[Path]:
    return _active_path


def log_exception(label: str, exc: BaseException) -> None:
    """Convenience helper used by callers that want to capture handled
    exceptions with a full stack trace into the per-run log file.
    """
    logging.getLogger("folderangel.runlog").error(
        "%s: %s\n%s",
        label,
        exc,
        "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
    )
