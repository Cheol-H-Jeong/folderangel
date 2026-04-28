"""SQLite index + FTS5 search."""
from __future__ import annotations

import json
import logging

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
    report_path: str = ""


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
    category_name TEXT DEFAULT '',
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
    category_name,
    reason,
    original_path,
    content_excerpt,
    content='files',
    content_rowid='id',
    tokenize='unicode61 remove_diacritics 2'
);

CREATE TRIGGER IF NOT EXISTS files_ai AFTER INSERT ON files BEGIN
    INSERT INTO files_fts(rowid, filename, folder, category, category_name, reason, original_path, content_excerpt)
    VALUES (new.id, new.filename, new.folder, new.category, coalesce(new.category_name, ''), coalesce(new.reason, ''), new.original_path, coalesce(new.content_excerpt, ''));
END;

CREATE TRIGGER IF NOT EXISTS files_ad AFTER DELETE ON files BEGIN
    INSERT INTO files_fts(files_fts, rowid, filename, folder, category, category_name, reason, original_path, content_excerpt)
    VALUES ('delete', old.id, old.filename, old.folder, old.category, coalesce(old.category_name, ''), coalesce(old.reason, ''), old.original_path, coalesce(old.content_excerpt, ''));
END;
"""


class IndexDB:
    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        # ``check_same_thread=False`` lets the worker thread share the
        # connection.  ``timeout=10`` prevents Windows file-lock errors
        # when another short-lived connection (e.g. the search view)
        # touches the DB simultaneously.  WAL mode plays nicer with
        # NTFS / APFS than the default rollback journal.
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
        self._migrate()
        self.conn.commit()

    def _migrate(self) -> None:
        """Bring an older DB up to the current schema, including a
        rebuild of the FTS5 virtual table when its column set drifts
        and a back-fill of newly-added columns from existing data
        sources (e.g. operation stats_json carries the human-readable
        category names that older runs never wrote into ``files``).
        """
        cur = self.conn.execute("PRAGMA table_info(files)")
        cols = {row["name"] for row in cur.fetchall()}
        if "content_excerpt" not in cols:
            self.conn.execute(
                "ALTER TABLE files ADD COLUMN content_excerpt TEXT DEFAULT ''"
            )
        if "category_name" not in cols:
            self.conn.execute(
                "ALTER TABLE files ADD COLUMN category_name TEXT DEFAULT ''"
            )

        # Back-fill category_name from each operation's stats_json.  The
        # human-readable category title (e.g. "범정부 초거대 AI 공통기반
        # BPR_ISP") was previously buried in stats; now we lift it onto
        # every file row so it joins the search index properly.
        for op in self.conn.execute(
            "SELECT id, stats_json FROM operations"
        ).fetchall():
            try:
                stats = json.loads(op["stats_json"] or "{}")
            except json.JSONDecodeError:
                continue
            id_to_name = {
                c["id"]: c.get("name", "")
                for c in (stats.get("categories") or [])
                if c.get("id")
            }
            if not id_to_name:
                continue
            for cid, cname in id_to_name.items():
                self.conn.execute(
                    "UPDATE files SET category_name = ? "
                    "WHERE op_id = ? AND category = ? AND coalesce(category_name,'') = ''",
                    (cname, op["id"], cid),
                )

        # If the FTS5 column list is older (missing content_excerpt or
        # category_name), drop and rebuild it from the files table —
        # populating from the DB we already have so old rows become
        # searchable too.
        cur = self.conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='files_fts'"
        )
        row = cur.fetchone()
        sql = (row["sql"] or "").lower() if row else ""
        needs_rebuild = ("content_excerpt" not in sql) or ("category_name" not in sql)
        if needs_rebuild:
            self.conn.executescript(
                """
                DROP TRIGGER IF EXISTS files_ai;
                DROP TRIGGER IF EXISTS files_ad;
                DROP TABLE IF EXISTS files_fts;
                CREATE VIRTUAL TABLE files_fts USING fts5(
                    filename, folder, category, category_name, reason,
                    original_path, content_excerpt,
                    content='files', content_rowid='id',
                    tokenize='unicode61 remove_diacritics 2'
                );
                CREATE TRIGGER files_ai AFTER INSERT ON files BEGIN
                  INSERT INTO files_fts(rowid, filename, folder, category, category_name, reason, original_path, content_excerpt)
                  VALUES (new.id, new.filename, new.folder, new.category, coalesce(new.category_name, ''), coalesce(new.reason, ''), new.original_path, coalesce(new.content_excerpt, ''));
                END;
                CREATE TRIGGER files_ad AFTER DELETE ON files BEGIN
                  INSERT INTO files_fts(files_fts, rowid, filename, folder, category, category_name, reason, original_path, content_excerpt)
                  VALUES ('delete', old.id, old.filename, old.folder, old.category, coalesce(old.category_name, ''), coalesce(old.reason, ''), old.original_path, coalesce(old.content_excerpt, ''));
                END;
                """
            )
            self.conn.execute(
                """
                INSERT INTO files_fts(rowid, filename, folder, category, category_name, reason, original_path, content_excerpt)
                SELECT id, filename, folder, category, coalesce(category_name,''),
                       coalesce(reason,''), original_path, coalesce(content_excerpt,'')
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
        # Persist the report path (set by the pipeline right before this
        # call) so history-tab double-click can open it without globbing.
        if getattr(op, "report_path", None):
            stats["report_path"] = str(op.report_path)
        # id → human-readable category name, used to keep ``files.category_name``
        # in sync with what's shown to the user.
        cat_name_by_id = {c.id: c.name for c in op.categories}
        cur = self.conn.cursor()
        # ----- de-dup the search index --------------------------------
        # Every file in this operation has been moved, so any prior row
        # whose ``new_path`` matches either the OLD or NEW location of
        # one of these files is now obsolete: same file, indexed under a
        # path that either no longer holds it (old location) or is about
        # to be re-inserted (new location).  Without this purge,
        # search results listed the same file twice and double-clicking
        # the stale row opened a broken link.
        if op.moved:
            stale_paths = {str(mf.original_path) for mf in op.moved} | {
                str(mf.new_path) for mf in op.moved
            }
            placeholders = ",".join("?" for _ in stale_paths)
            cur.execute(
                f"DELETE FROM files WHERE new_path IN ({placeholders})",
                tuple(stale_paths),
            )
        # Also sweep any orphaned rows whose recorded ``new_path`` no
        # longer exists on disk — these are leftovers from earlier runs
        # whose files have been deleted, renamed, or moved by the user.
        # We cap the scan at 5000 rows per operation to keep this cheap;
        # full housekeeping can be done via reindex.
        try:
            for r in cur.execute(
                "SELECT id, new_path FROM files ORDER BY id DESC LIMIT 5000"
            ).fetchall():
                p = r["new_path"]
                if p and not Path(p).exists():
                    cur.execute("DELETE FROM files WHERE id = ?", (r["id"],))
        except Exception as exc:
            log.debug("orphan sweep skipped: %s", exc)

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
                "INSERT INTO files(op_id, original_path, new_path, filename, folder, category, category_name, reason, score, content_excerpt, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    op_id,
                    str(mf.original_path),
                    str(new_path),
                    new_path.name,
                    str(new_path.parent),
                    mf.category_id,
                    cat_name_by_id.get(mf.category_id, ""),
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
    # ------------------------------------------------------------------
    def reindex_folder(self, root: Path, recursive: bool = True) -> int:
        """Walk *root* on disk and refresh the index for that subtree.

        Use case: the user reorganised manually, or runs the search
        before kicking off a new organize pass, or simply wants
        everything under a folder findable without re-running the LLM.
        We synthesise a special "scan" operation that records every
        existing file with its current ``new_path`` (= where it lives
        right now), the parent folder name as ``category_name`` (so
        searching for the folder name finds the children), and an
        empty content excerpt — which the next real organize pass will
        upgrade.  Returns the number of files indexed.
        """
        root = Path(root).resolve()
        if not root.is_dir():
            return 0
        # Drop any prior "scan" op for the same root so the index doesn't
        # accumulate stale duplicates as the user reindexes repeatedly.
        for old in self.conn.execute(
            "SELECT id FROM operations WHERE target_root = ? AND "
            "json_extract(coalesce(stats_json,'{}'), '$.kind') = 'reindex'",
            (str(root),),
        ).fetchall():
            self.conn.execute("DELETE FROM operations WHERE id = ?", (old["id"],))

        now = datetime.now().astimezone().isoformat(timespec="seconds")
        cur = self.conn.cursor()
        cur.execute(
            "INSERT INTO operations(target_root, started_at, finished_at, dry_run, stats_json) "
            "VALUES (?, ?, ?, 0, ?)",
            (str(root), now, now, json.dumps({"kind": "reindex"})),
        )
        op_id = cur.lastrowid

        count = 0
        iterator = root.rglob("*") if recursive else root.iterdir()
        for p in iterator:
            try:
                if not p.is_file():
                    continue
            except OSError:
                continue
            parent = p.parent
            cur.execute(
                "INSERT INTO files(op_id, original_path, new_path, filename, folder, "
                "category, category_name, reason, score, content_excerpt, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, '', 1.0, '', ?)",
                (
                    op_id,
                    str(p),                # no prior original path; reuse current
                    str(p),
                    p.name,
                    str(parent),
                    parent.name,           # category id ≈ parent folder name
                    parent.name,           # human-readable category = same
                    now,
                ),
            )
            count += 1
        self.conn.commit()
        return count

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
                   o.stats_json,
                   (SELECT COUNT(*) FROM files WHERE op_id = o.id) AS n
            FROM operations o
            ORDER BY o.id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        out: list[OperationInfo] = []
        for r in rows:
            report_path = ""
            try:
                stats = json.loads(r["stats_json"] or "{}")
                report_path = str(stats.get("report_path") or "")
            except Exception:
                report_path = ""
            out.append(OperationInfo(
                op_id=r["id"],
                target_root=r["target_root"],
                started_at=r["started_at"],
                finished_at=r["finished_at"],
                dry_run=bool(r["dry_run"]),
                moved_count=r["n"],
                report_path=report_path,
            ))
        return out

    def latest_operation_id(self) -> Optional[int]:
        row = self.conn.execute(
            "SELECT id FROM operations ORDER BY id DESC LIMIT 1"
        ).fetchone()
        return row["id"] if row else None

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
