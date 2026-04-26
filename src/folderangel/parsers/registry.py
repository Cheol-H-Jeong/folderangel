"""Extension → parser dispatcher. Never raises; returns '' on failure."""
from __future__ import annotations

import logging
import signal
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Callable

log = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS: set[str] = {
    ".pdf",
    ".docx",
    ".pptx",
    ".ppsx",
    ".xlsx",
    ".doc",
    ".ppt",
    ".pps",
    ".xls",
    ".hwp",
    ".hwpx",
    ".odt",
    ".rtf",
    ".txt",
    ".md",
    ".markdown",
    ".csv",
    ".tsv",
    ".log",
    ".html",
    ".htm",
    ".json",
    ".jsonl",
    ".xml",
    ".yaml",
    ".yml",
    ".ini",
    ".cfg",
    ".toml",
}


@contextmanager
def _time_limit(seconds: float):
    """Cross-platform-ish soft timeout for parsers.

    Uses SIGALRM on POSIX main thread; otherwise falls back to a watchdog
    thread that sets a flag (parsers poll via callbacks that accept a budget).
    For the simple short files we target, this is adequate.
    """
    # SIGALRM is only available on POSIX main thread; we skip on Windows or non-main threads.
    import sys

    use_alarm = (
        hasattr(signal, "SIGALRM")
        and not sys.platform.startswith("win")
        and threading.current_thread() is threading.main_thread()
    )

    if not use_alarm:
        yield
        return

    def _raise(signum, frame):
        raise TimeoutError("parser timeout")

    old = signal.signal(signal.SIGALRM, _raise)
    signal.setitimer(signal.ITIMER_REAL, max(0.05, float(seconds)))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


def _safe(parser: Callable[[Path, int], str], path: Path, max_chars: int, timeout: float) -> str:
    try:
        with _time_limit(timeout):
            return (parser(path, max_chars) or "").strip()[:max_chars]
    except TimeoutError:
        log.warning("parser timeout: %s", path)
    except Exception as exc:  # pragma: no cover — parser-specific
        log.warning("parser failed for %s: %s", path, exc)
    return ""


def extract_excerpt(path: Path, max_chars: int = 1800, timeout: float = 5.0) -> str:
    """Return up to ``max_chars`` characters of plain text from the document.

    Returns '' if the file is unsupported, unreadable, or parsing fails.
    """
    from . import pdf as pdf_parser
    from . import office, text as text_parser, hwp as hwp_parser

    path = Path(path)
    ext = path.suffix.lower()
    if ext == ".pdf":
        return _safe(pdf_parser.parse, path, max_chars, timeout)
    if ext == ".docx":
        return _safe(office.parse_docx, path, max_chars, timeout)
    if ext in {".pptx", ".ppsx"}:
        # .ppsx is an autoplay variant of .pptx — same XML container.
        return _safe(office.parse_pptx, path, max_chars, timeout)
    if ext == ".xlsx":
        return _safe(office.parse_xlsx, path, max_chars, timeout)
    if ext == ".odt":
        return _safe(office.parse_odt, path, max_chars, timeout)
    if ext in {".doc", ".ppt", ".pps", ".xls"}:
        # Legacy binary Office formats — best-effort text scrape via
        # OLE compound storage; better than nothing for indexing.
        return _safe(office.parse_legacy_office, path, max_chars, timeout)
    if ext == ".hwpx":
        return _safe(hwp_parser.parse_hwpx, path, max_chars, timeout)
    if ext == ".hwp":
        return _safe(hwp_parser.parse_hwp, path, max_chars, timeout)
    if ext == ".rtf":
        return _safe(text_parser.parse_rtf, path, max_chars, timeout)
    if ext in {".txt", ".md", ".markdown", ".csv", ".tsv", ".log",
               ".json", ".jsonl", ".xml", ".yaml", ".yml",
               ".ini", ".cfg", ".toml"}:
        return _safe(text_parser.parse_plain, path, max_chars, timeout)
    if ext in {".html", ".htm"}:
        return _safe(text_parser.parse_html, path, max_chars, timeout)
    return ""
