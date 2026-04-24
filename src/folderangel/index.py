"""SQLite index + FTS5 search + simple rollback."""
from __future__ import annotations

import json
import logging
import shutil
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .models import OperationResult

log = logging.getLogger(__name__)


@dataclass
class SearchHit:
    file_id: int
    op_id: int
    original_path: str
    new_path: str
    category: str
    reason: str
    created_at: str


@dataclass
class OperationInfo:
    op_id: int
    target_root: str
    started_at: str
    finished_at: str
    dry_run: bool
    moved_count: int


@dataclass
class RollbackResult:
    restored: int
    failed: list[str]


_SCHEMA = """
CREATE TABLE IF NOT EXISTS operations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    target_root TEXT NOT NULL,
    started_at TEXT NOT NULL,
    finished_at TEXT NOT NULL,
    dry_run INTEGER NOT NULL DEFAULT 0,
    stats_json TEXT
);

CREATE TABLE IF NOT EXISTS files (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    op_id INTEGER NOT NULL REFERENCES operations(id) ON DELETE CASCADE,
    original_path TEXT NOT NULL,
    new_path TEXT NOT NULL,
    filename TEXT NOT NULL,
    folder TEXT NOT NULL,
    category TEXT NOT NULL,
    reason TEXT,
    score REAL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS shortcuts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    op_id INTEGER NOT NULL REFERENCES operations(id) ON DELETE CASCADE,
    file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    shortcut_path TEXT NOT NULL
);

CREATE VIRTUAL TABLE IF NOT EXISTS files_fts USING fts5(
    filename,
    folder,
    category,
    reason,
    original_path,
    content='files',
    content_rowid='id'
);

CREATE TRIGGER IF NOT EXISTS files_ai AFTER INSERT ON files BEGIN
    INSERT INTO files_fts(rowid, filename, folder, category, reason, original_path)
    VALUES (new.id, new.filename, new.folder, new.category, coalesce(new.reason, ''), new.original_path);
END;

CREATE TRIGGER IF NOT EXISTS files_ad AFTER DELETE ON files BEGIN
    INSERT INTO files_fts(files_fts, rowid, filename, folder, category, reason, original_path)
    VALUES ('delete', old.id, old.filename, old.folder, old.category, coalesce(old.reason, ''), old.original_path);
END;
"""


class IndexDB:
    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(_SCHEMA)
        self.conn.commit()

    # ------------------------------------------------------------------
    def record_operation(self, op: OperationResult) -> int:
        stats = {
            "total_scanned": op.total_scanned,
            "total_moved": op.total_moved,
            "total_skipped": op.total_skipped,
            "total_shortcuts": op.total_shortcuts,
            "categories": [c.__dict__ for c in op.categories],
        }
        cur = self.conn.cursor()
        cur.execute(
            "INSERT INTO operations(target_root, started_at, finished_at, dry_run, stats_json) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                str(op.target_root),
                op.started_at.isoformat(timespec="seconds"),
                op.finished_at.isoformat(timespec="seconds"),
                1 if op.dry_run else 0,
                json.dumps(stats, ensure_ascii=False),
            ),
        )
        op_id = cur.lastrowid
        now = datetime.now().astimezone().isoformat(timespec="seconds")
        for mf in op.moved:
            new_path = Path(mf.new_path)
            cur.execute(
                "INSERT INTO files(op_id, original_path, new_path, filename, folder, category, reason, score, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    op_id,
                    str(mf.original_path),
                    str(new_path),
                    new_path.name,
                    str(new_path.parent),
                    mf.category_id,
                    mf.reason,
                    mf.score,
                    now,
                ),
            )
            file_id = cur.lastrowid
            for sp in mf.shortcuts:
                cur.execute(
                    "INSERT INTO shortcuts(op_id, file_id, shortcut_path) VALUES (?, ?, ?)",
                    (op_id, file_id, str(sp)),
                )
        self.conn.commit()
        op.operation_id = op_id
        return op_id

    # ------------------------------------------------------------------
    def search(self, query: str, limit: int = 50) -> list[SearchHit]:
        q = query.strip()
        if not q:
            return []
        try:
            rows = self.conn.execute(
                """
                SELECT f.id, f.op_id, f.original_path, f.new_path, f.category, f.reason, f.created_at
                FROM files_fts
                JOIN files f ON f.id = files_fts.rowid
                WHERE files_fts MATCH ?
                ORDER BY f.id DESC
                LIMIT ?
                """,
                (_prepare_fts_query(q), limit),
            ).fetchall()
        except sqlite3.OperationalError as exc:
            log.debug("fts failed (%s); falling back to LIKE", exc)
            rows = self.conn.execute(
                """
                SELECT id, op_id, original_path, new_path, category, reason, created_at
                FROM files
                WHERE original_path LIKE ? OR new_path LIKE ? OR category LIKE ? OR reason LIKE ? OR filename LIKE ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (f"%{q}%", f"%{q}%", f"%{q}%", f"%{q}%", f"%{q}%", limit),
            ).fetchall()
        return [
            SearchHit(
                file_id=r["id"],
                op_id=r["op_id"],
                original_path=r["original_path"],
                new_path=r["new_path"],
                category=r["category"],
                reason=r["reason"] or "",
                created_at=r["created_at"],
            )
            for r in rows
        ]

    def list_operations(self, limit: int = 50) -> list[OperationInfo]:
        rows = self.conn.execute(
            """
            SELECT o.id, o.target_root, o.started_at, o.finished_at, o.dry_run,
                   (SELECT COUNT(*) FROM files WHERE op_id = o.id) AS n
            FROM operations o
            ORDER BY o.id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [
            OperationInfo(
                op_id=r["id"],
                target_root=r["target_root"],
                started_at=r["started_at"],
                finished_at=r["finished_at"],
                dry_run=bool(r["dry_run"]),
                moved_count=r["n"],
            )
            for r in rows
        ]

    def rollback(self, op_id: int) -> RollbackResult:
        rows = self.conn.execute(
            "SELECT id, original_path, new_path FROM files WHERE op_id = ?",
            (op_id,),
        ).fetchall()
        shortcuts = self.conn.execute(
            "SELECT shortcut_path FROM shortcuts WHERE op_id = ?", (op_id,)
        ).fetchall()

        restored = 0
        failed: list[str] = []
        # remove shortcuts first
        for s in shortcuts:
            sp = Path(s["shortcut_path"])
            try:
                if sp.exists() or sp.is_symlink():
                    sp.unlink()
            except Exception as exc:
                failed.append(f"{sp}: {exc}")

        touched_folders: set[Path] = set()
        for r in rows:
            orig = Path(r["original_path"])
            new = Path(r["new_path"])
            try:
                orig.parent.mkdir(parents=True, exist_ok=True)
                if new.exists():
                    shutil.move(str(new), str(orig))
                    restored += 1
                    touched_folders.add(new.parent)
                else:
                    failed.append(f"{new}: missing")
            except Exception as exc:
                failed.append(f"{new}: {exc}")

        # Best-effort cleanup: remove any category folders that are now empty.
        for folder in sorted(touched_folders, key=lambda p: len(p.parts), reverse=True):
            try:
                if folder.is_dir() and not any(folder.iterdir()):
                    folder.rmdir()
            except Exception as exc:
                log.debug("rmdir skipped %s: %s", folder, exc)

        self.conn.execute("DELETE FROM operations WHERE id = ?", (op_id,))
        self.conn.commit()
        return RollbackResult(restored=restored, failed=failed)

    def close(self) -> None:
        try:
            self.conn.close()
        except Exception:
            pass


def _prepare_fts_query(q: str) -> str:
    """Escape user input for FTS5 MATCH."""
    cleaned = "".join(ch if ch.isalnum() or ch in "-_가-힣 " else " " for ch in q)
    tokens = [t for t in cleaned.split() if t]
    if not tokens:
        # fall back to quoted phrase
        return '"' + q.replace('"', "") + '"'
    return " OR ".join(f'"{t}"*' for t in tokens)
