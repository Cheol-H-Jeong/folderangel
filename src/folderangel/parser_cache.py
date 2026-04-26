"""Persistent cache of parser excerpts.

Re-running ``folderangel`` on the same corpus re-parses every
document — that's the dominant cost on huge folders.  We keep a
small SQLite store keyed by ``(absolute_path, mtime, size)`` so
unchanged files skip parsing entirely.

The cache is best-effort: if anything goes wrong we fall back to a
cold parse and overwrite the (presumably stale / corrupt) row.
"""
from __future__ import annotations

import logging
import sqlite3
import threading
from pathlib import Path
from typing import Callable, Optional

log = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS parser_cache (
    path     TEXT PRIMARY KEY,
    mtime    REAL NOT NULL,
    size     INTEGER NOT NULL,
    excerpt  TEXT NOT NULL,
    updated  TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
"""


class ParserCache:
    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.conn = sqlite3.connect(
            self.db_path, check_same_thread=False, timeout=10.0
        )
        self.conn.row_factory = sqlite3.Row
        try:
            self.conn.execute("PRAGMA journal_mode=WAL")
            self.conn.execute("PRAGMA synchronous=NORMAL")
        except sqlite3.OperationalError:
            pass
        self.conn.executescript(_SCHEMA)
        self.conn.commit()

    def get_or_parse(
        self,
        path: Path,
        mtime: float,
        size: int,
        cold_parse: Callable[[], str],
    ) -> str:
        """Return the cached excerpt for *path* if its identity matches,
        otherwise call *cold_parse()*, store, and return its result.
        """
        key = str(path)
        with self._lock:
            row = self.conn.execute(
                "SELECT mtime, size, excerpt FROM parser_cache WHERE path = ?",
                (key,),
            ).fetchone()
        if row and abs(row["mtime"] - mtime) < 1e-3 and row["size"] == size:
            return row["excerpt"]
        # Miss → parse.
        try:
            excerpt = cold_parse() or ""
        except Exception as exc:  # pragma: no cover — parser-specific
            log.warning("cold parse failed for %s: %s", path, exc)
            excerpt = ""
        with self._lock:
            self.conn.execute(
                "INSERT INTO parser_cache(path, mtime, size, excerpt) "
                "VALUES (?, ?, ?, ?) "
                "ON CONFLICT(path) DO UPDATE SET "
                "  mtime = excluded.mtime, size = excluded.size, "
                "  excerpt = excluded.excerpt, updated = CURRENT_TIMESTAMP",
                (key, mtime, size, excerpt),
            )
            self.conn.commit()
        return excerpt

    def evict_missing(self, root: Optional[Path] = None, batch: int = 500) -> int:
        """Drop rows whose path no longer exists on disk.

        ``root`` narrows the sweep to a subtree; without it everything
        is considered.  Returns the number of evicted rows.
        """
        with self._lock:
            rows = self.conn.execute("SELECT path FROM parser_cache").fetchall()
        gone: list[str] = []
        for r in rows:
            p = Path(r["path"])
            if root is not None and not str(p).startswith(str(root)):
                continue
            if not p.exists():
                gone.append(r["path"])
        if not gone:
            return 0
        with self._lock:
            for i in range(0, len(gone), batch):
                chunk = gone[i : i + batch]
                qmarks = ",".join("?" * len(chunk))
                self.conn.execute(
                    f"DELETE FROM parser_cache WHERE path IN ({qmarks})", chunk
                )
            self.conn.commit()
        return len(gone)

    def close(self) -> None:
        try:
            self.conn.close()
        except Exception:
            pass
