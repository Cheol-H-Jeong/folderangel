"""End-to-end smoke test with the mock planner."""
from pathlib import Path

from folderangel.config import Config
from folderangel.index import IndexDB
from folderangel.pipeline import run


def test_full_pipeline_mock(tmp_path, monkeypatch):
    # Avoid touching user's real keyring/config
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    (tmp_path / "meeting-notes.md").write_text("# 회의 메모\n오늘 회의록.")
    (tmp_path / "budget.xlsx").write_bytes(b"")
    (tmp_path / "invoice_2025.pdf").write_bytes(b"%PDF-1.4")
    (tmp_path / "photo.jpg").write_bytes(b"\xff\xd8\xff\xe0")
    (tmp_path / "photo2.jpg").write_bytes(b"\xff\xd8\xff\xe0")
    (tmp_path / "readme.txt").write_text("Hello world")

    db = IndexDB(tmp_path / "_idx.db")
    op = run(
        target_root=tmp_path,
        config=Config(),
        recursive=False,
        dry_run=False,
        index_db=db,
        force_mock=True,
    )
    assert op.total_moved >= 5
    assert len(op.categories) >= 2
    for mf in op.moved:
        assert mf.new_path.exists()
    db.close()
