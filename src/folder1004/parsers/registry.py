"""Extension → parser dispatcher. Never raises; returns '' on failure."""
from __future__ import annotations

import concurrent.futures as _futures
import logging
from pathlib import Path
from typing import Callable

log = logging.getLogger(__name__)

# Single shared worker pool so we don't spawn one thread per file just
# to enforce a timeout.  Cross-platform: works identically on Linux,
# macOS, and Windows (no SIGALRM dependency).
_PARSE_POOL = _futures.ThreadPoolExecutor(
    max_workers=2, thread_name_prefix="folder1004-parse"
)

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
    # Archives — listed for member-name extraction (no decompression).
    ".zip",
    ".jar",
    ".war",
    ".tar",
    ".tgz",
    ".tbz",
    ".txz",
}


def _safe(parser: Callable[[Path, int], str], path: Path, max_chars: int, timeout: float) -> str:
    """Run *parser* with a hard wall-clock timeout, returning ``""`` on
    failure.  Cross-platform: uses a thread-pool ``Future`` so it works
    on Linux, macOS, and Windows alike, including from non-main threads.

    The parser thread may keep running after the timeout (Python has no
    safe way to kill it), but the caller is unblocked immediately.
    """
    try:
        future = _PARSE_POOL.submit(parser, path, max_chars)
        try:
            text = future.result(timeout=max(0.1, float(timeout)))
        except _futures.TimeoutError:
            future.cancel()
            log.warning("parser timeout (%.1fs): %s", timeout, path)
            return ""
    except Exception as exc:  # pragma: no cover — parser-specific
        log.warning("parser failed for %s: %s", path, exc)
        return ""
    return (text or "").strip()[:max_chars]


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
    # Archive containers: list member names so the classifier can use
    # them as a synthetic "body".  See :mod:`folder1004.parsers.archive`.
    from . import archive as archive_parser
    if archive_parser.is_archive(path):
        return _safe(archive_parser.parse, path, max_chars, timeout)
    return ""
