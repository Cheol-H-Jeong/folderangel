"""SQLite index + FTS5 search + simple rollback."""
from __future__ import annotations

import json
import logging
import shutil
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

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
    snippet: str = ""        # FTS5 ``snippet()`` of where the term hit
    matched_in: str = ""     # which field carried the strongest hit


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
    content_excerpt TEXT DEFAULT '',
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
    content_excerpt,
    content='files',
    content_rowid='id',
    tokenize='unicode61 remove_diacritics 2'
);

CREATE TRIGGER IF NOT EXISTS files_ai AFTER INSERT ON files BEGIN
    INSERT INTO files_fts(rowid, filename, folder, category, reason, original_path, content_excerpt)
    VALUES (new.id, new.filename, new.folder, new.category, coalesce(new.reason, ''), new.original_path, coalesce(new.content_excerpt, ''));
END;

CREATE TRIGGER IF NOT EXISTS files_ad AFTER DELETE ON files BEGIN
    INSERT INTO files_fts(files_fts, rowid, filename, folder, category, reason, original_path, content_excerpt)
    VALUES ('delete', old.id, old.filename, old.folder, old.category, coalesce(old.reason, ''), old.original_path, coalesce(old.content_excerpt, ''));
END;
"""


class IndexDB:
    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(_SCHEMA)
        self._migrate()
        self.conn.commit()

    def _migrate(self) -> None:
        """Bring an older DB up to the current schema, including a
        rebuild of the FTS5 virtual table when its column set drifts.
        """
        cur = self.conn.execute("PRAGMA table_info(files)")
        cols = {row["name"] for row in cur.fetchall()}
        if "content_excerpt" not in cols:
            self.conn.execute(
                "ALTER TABLE files ADD COLUMN content_excerpt TEXT DEFAULT ''"
            )

        # If the FTS5 column list is older (no content_excerpt), drop and
        # rebuild it from the files table — populating from the DB we
        # already have so old rows become searchable too.
        cur = self.conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='files_fts'"
        )
        row = cur.fetchone()
        sql = (row["sql"] or "").lower() if row else ""
        if "content_excerpt" not in sql:
            self.conn.executescript(
                """
                DROP TRIGGER IF EXISTS files_ai;
                DROP TRIGGER IF EXISTS files_ad;
                DROP TABLE IF EXISTS files_fts;
                CREATE VIRTUAL TABLE files_fts USING fts5(
                    filename, folder, category, reason, original_path,
                    content_excerpt,
                    content='files', content_rowid='id',
                    tokenize='unicode61 remove_diacritics 2'
                );
                CREATE TRIGGER files_ai AFTER INSERT ON files BEGIN
                  INSERT INTO files_fts(rowid, filename, folder, category, reason, original_path, content_excerpt)
                  VALUES (new.id, new.filename, new.folder, new.category, coalesce(new.reason, ''), new.original_path, coalesce(new.content_excerpt, ''));
                END;
                CREATE TRIGGER files_ad AFTER DELETE ON files BEGIN
                  INSERT INTO files_fts(files_fts, rowid, filename, folder, category, reason, original_path, content_excerpt)
                  VALUES ('delete', old.id, old.filename, old.folder, old.category, coalesce(old.reason, ''), old.original_path, coalesce(old.content_excerpt, ''));
                END;
                """
            )
            # Backfill FTS from existing rows
            self.conn.execute(
                """
                INSERT INTO files_fts(rowid, filename, folder, category, reason, original_path, content_excerpt)
                SELECT id, filename, folder, category, coalesce(reason,''), original_path, coalesce(content_excerpt,'')
                FROM files
                """
            )

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
                "INSERT INTO files(op_id, original_path, new_path, filename, folder, category, reason, score, content_excerpt, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    op_id,
                    str(mf.original_path),
                    str(new_path),
                    new_path.name,
                    str(new_path.parent),
                    mf.category_id,
                    mf.reason,
                    mf.score,
                    (mf.content_excerpt or "")[:1800],
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
    def search(self, query: str, limit: int = 100) -> list[SearchHit]:
        q = query.strip()
        if not q:
            return []
        # FTS5 path with snippet of the strongest hit; rank by bm25.
        try:
            prep = _prepare_fts_query(q)
            rows = self.conn.execute(
                """
                SELECT
                    f.id, f.op_id, f.original_path, f.new_path,
                    f.category, f.reason, f.created_at, f.content_excerpt,
                    snippet(files_fts, -1, '«', '»', ' … ', 16) AS snip
                FROM files_fts
                JOIN files f ON f.id = files_fts.rowid
                WHERE files_fts MATCH ?
                ORDER BY bm25(files_fts) ASC
                LIMIT ?
                """,
                (prep, limit),
            ).fetchall()
            hits = [
                SearchHit(
                    file_id=r["id"],
                    op_id=r["op_id"],
                    original_path=r["original_path"],
                    new_path=r["new_path"],
                    category=r["category"],
                    reason=r["reason"] or "",
                    created_at=r["created_at"],
                    snippet=(r["snip"] or "").strip(),
                    matched_in=_field_of_match(r["snip"] or "", r),
                )
                for r in rows
            ]
        except sqlite3.OperationalError as exc:
            log.debug("fts failed (%s); falling back to LIKE", exc)
            hits = []

        # Always supplement with a LIKE pass on filename / folder /
        # original_path / content_excerpt — Korean tokenisation in FTS5 is
        # imperfect (no morphology), so a pure FTS pass occasionally
        # misses substrings the user obviously expects to find.  Dedup
        # by file_id, FTS hits stay first.
        seen = {h.file_id for h in hits}
        if len(hits) < limit:
            like = f"%{q}%"
            rest = self.conn.execute(
                """
                SELECT id, op_id, original_path, new_path, category, reason, created_at,
                       content_excerpt
                FROM files
                WHERE id NOT IN (%s) AND (
                      filename LIKE ?
                   OR folder LIKE ?
                   OR original_path LIKE ?
                   OR new_path LIKE ?
                   OR category LIKE ?
                   OR reason LIKE ?
                   OR content_excerpt LIKE ?
                )
                ORDER BY id DESC
                LIMIT ?
                """ % (",".join(str(i) for i in seen) or "0",),
                (like, like, like, like, like, like, like, limit - len(hits)),
            ).fetchall()
            for r in rest:
                excerpt = r["content_excerpt"] or ""
                snip = _excerpt_around(excerpt, q) if q.lower() in excerpt.lower() else ""
                hits.append(
                    SearchHit(
                        file_id=r["id"],
                        op_id=r["op_id"],
                        original_path=r["original_path"],
                        new_path=r["new_path"],
                        category=r["category"],
                        reason=r["reason"] or "",
                        created_at=r["created_at"],
                        snippet=snip,
                        matched_in="content" if snip else "name",
                    )
                )
        return hits

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

    def latest_operation_id(self) -> Optional[int]:
        row = self.conn.execute(
            "SELECT id FROM operations ORDER BY id DESC LIMIT 1"
        ).fetchone()
        return row["id"] if row else None

    def rollback(
        self,
        op_id: int,
        *,
        force: bool = False,
    ) -> RollbackResult:
        """Restore the files moved during operation ``op_id``.

        Safety policy:
          * Rolling back the **most recent** operation is always
            allowed.  The on-disk state is the closest to what we
            recorded, so the move-back is well-defined.
          * Rolling back **any older** operation is dangerous: the
            user (or a later FolderAngel op) may have re-organised the
            same files in between, so naively moving them back can
            overwrite newer work.  We refuse unless ``force=True`` is
            passed.
          * Even with ``force=True`` we still skip individual files
            whose recorded ``new_path`` no longer exists at that exact
            location (someone moved them elsewhere) or whose recorded
            ``original_path`` is now occupied by a different file —
            those collisions are reported in ``failed`` and not
            overwritten.
        """
        latest = self.latest_operation_id()
        is_latest = (latest is not None and op_id == latest)
        if not is_latest and not force:
            return RollbackResult(
                restored=0,
                failed=[
                    f"op_id {op_id} is not the most recent operation "
                    f"(latest is {latest}); pass force=True to attempt "
                    "a guarded rollback of an older operation",
                ],
            )

        rows = self.conn.execute(
            "SELECT id, original_path, new_path FROM files WHERE op_id = ?",
            (op_id,),
        ).fetchall()
        shortcuts = self.conn.execute(
            "SELECT shortcut_path FROM shortcuts WHERE op_id = ?", (op_id,)
        ).fetchall()

        restored = 0
        failed: list[str] = []
        # remove shortcuts first — these are derived state, safe either way.
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
                if not new.exists():
                    failed.append(f"{new}: missing (file moved or deleted since op)")
                    continue
                if orig.exists() and orig.resolve() != new.resolve():
                    # Collision at the original location — refuse to
                    # overwrite a newer file, even with force.
                    failed.append(
                        f"{orig}: target already occupied by a different file; "
                        "refusing to overwrite"
                    )
                    continue
                orig.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(new), str(orig))
                restored += 1
                touched_folders.add(new.parent)
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


def _excerpt_around(text: str, needle: str, radius: int = 60) -> str:
    """Return ``…before«needle»after…`` around the first match."""
    if not text or not needle:
        return ""
    idx = text.lower().find(needle.lower())
    if idx < 0:
        return ""
    start = max(0, idx - radius)
    end = min(len(text), idx + len(needle) + radius)
    prefix = "…" if start > 0 else ""
    suffix = "…" if end < len(text) else ""
    chunk = text[start:end].replace("\n", " ")
    chunk = (
        chunk[: idx - start]
        + "«" + chunk[idx - start : idx - start + len(needle)] + "»"
        + chunk[idx - start + len(needle):]
    )
    return f"{prefix}{chunk}{suffix}"


def _field_of_match(snip: str, row) -> str:
    """Best-effort: which column carried the snippet."""
    if not snip:
        return ""
    s = snip.replace("«", "").replace("»", "").lower()
    for label, value in (
        ("name",     (row["new_path"] or "").rsplit("/", 1)[-1].lower()),
        ("category", (row["category"] or "").lower()),
        ("reason",   (row["reason"] or "").lower()),
        ("content",  (row["content_excerpt"] or "").lower()[:200]),
        ("path",     (row["original_path"] or "").lower()),
    ):
        if value and value[:30] and value[:30] in s:
            return label
    return "match"


def _prepare_fts_query(q: str) -> str:
    """Escape user input for FTS5 MATCH."""
    cleaned = "".join(ch if ch.isalnum() or ch in "-_가-힣 " else " " for ch in q)
    tokens = [t for t in cleaned.split() if t]
    if not tokens:
        # fall back to quoted phrase
        return '"' + q.replace('"', "") + '"'
    return " OR ".join(f'"{t}"*' for t in tokens)
