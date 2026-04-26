"""Cross-platform smoke tests.

Run on every OS in CI; verify the platform-sensitive bits behave the
same way on Linux, macOS, and Windows wherever possible.
"""
import os
import sys
from pathlib import Path

import pytest

from folderangel.config import default_paths


def test_default_paths_uses_platform_dir(tmp_path, monkeypatch):
    """Each OS should pick its conventional data dir."""
    monkeypatch.delenv("FOLDERANGEL_HOME", raising=False)
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "AppData" / "Local"))
    monkeypatch.setenv("APPDATA", str(tmp_path / "AppData" / "Roaming"))
    p = default_paths()
    if sys.platform.startswith("win"):
        assert "FolderAngel" in str(p.root)
    elif sys.platform == "darwin":
        # On macOS we expect Library/Application Support/FolderAngel
        # *or* the legacy ~/.folderangel if it pre-exists.
        assert ("Application Support" in str(p.root)) or str(p.root).endswith(".folderangel")
    else:
        assert ("share/folderangel" in str(p.root)) or str(p.root).endswith(".folderangel")


def test_default_paths_honours_override(tmp_path, monkeypatch):
    monkeypatch.setenv("FOLDERANGEL_HOME", str(tmp_path / "custom"))
    p = default_paths()
    assert p.root == tmp_path / "custom"


def test_parser_timeout_works_off_main_thread(tmp_path):
    """The cross-platform parser timeout used to depend on SIGALRM, which
    only works on the POSIX main thread.  Verify it now fires correctly
    from a worker thread, on every OS."""
    import threading
    from folderangel.parsers.registry import _safe

    def slow(_p, _n):
        import time
        time.sleep(2.0)
        return "should-not-arrive"

    out = []
    def runner():
        out.append(_safe(slow, tmp_path / "x.txt", 100, timeout=0.2))
    t = threading.Thread(target=runner)
    t.start()
    t.join(timeout=5.0)
    assert out == [""]  # timed out → empty


def test_index_open_with_long_path(tmp_path):
    """SQLite + WAL must work in a deeply-nested path (Windows MAX_PATH)."""
    from folderangel.index import IndexDB
    deep = tmp_path
    for i in range(10):
        deep = deep / f"세부폴더_{i:02d}"
    deep.mkdir(parents=True)
    db = IndexDB(deep / "index.db")
    db.close()
    assert (deep / "index.db").exists()


def test_korean_filename_round_trips_through_index(tmp_path):
    """Korean characters in filenames must survive insert + retrieve on
    every filesystem.  Catches HFS+ NFD vs APFS NFC differences and
    NTFS UTF-16 quirks."""
    from datetime import datetime
    from folderangel.index import IndexDB
    from folderangel.models import Category, MovedFile, OperationResult

    folder = tmp_path / "프로젝트"
    folder.mkdir()
    f = folder / "한국지역정보개발원_제안서.pdf"
    f.write_text("x")
    op = OperationResult(
        target_root=tmp_path, started_at=datetime.now().astimezone(),
        finished_at=datetime.now().astimezone(), dry_run=False,
        categories=[Category(id="c", name="프로젝트")],
        moved=[MovedFile(original_path=tmp_path / "한국지역정보개발원_제안서.pdf",
                         new_path=f, category_id="c", reason="x", score=1.0)],
        skipped=[], total_scanned=1,
    )
    db = IndexDB(tmp_path / "idx.db")
    db.record_operation(op)
    hits = db.search("한국지역")
    assert hits and "한국지역" in hits[0].new_path
    db.close()
