"""Filesystem scanner — yields files the user asked to organize."""
from __future__ import annotations

import fnmatch
import logging
import os
from pathlib import Path
from typing import Iterable

log = logging.getLogger(__name__)


class ScanTooLargeError(RuntimeError):
    def __init__(self, count: int, limit: int):
        super().__init__(f"scanned {count} files (limit {limit})")
        self.count = count
        self.limit = limit


def _matches_any(name: str, patterns: Iterable[str]) -> bool:
    return any(fnmatch.fnmatch(name, p) for p in patterns)


def scan(
    root: Path,
    recursive: bool = False,
    ignore_patterns: Iterable[str] = (),
    max_files: int = 5000,
) -> list[Path]:
    """Collect files under *root* respecting the flags.

    Raises ScanTooLargeError if ``max_files`` exceeded. Never follows symlinks.
    """
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)

    ignore = list(ignore_patterns)
    results: list[Path] = []

    def _walk(current: Path):
        try:
            with os.scandir(current) as it:
                entries = list(it)
        except PermissionError as exc:
            log.warning("skip (permission): %s (%s)", current, exc)
            return
        for entry in entries:
            name = entry.name
            if _matches_any(name, ignore):
                continue
            if entry.is_symlink():
                continue
            if entry.is_dir(follow_symlinks=False):
                if recursive:
                    _walk(Path(entry.path))
                continue
            if entry.is_file(follow_symlinks=False):
                results.append(Path(entry.path))
                if len(results) > max_files:
                    raise ScanTooLargeError(len(results), max_files)

    _walk(root)
    return results
