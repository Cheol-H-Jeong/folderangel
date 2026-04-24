from datetime import datetime
from pathlib import Path

from folderangel.index import IndexDB
from folderangel.models import Category, MovedFile, OperationResult


def _sample_op(tmp_path: Path) -> OperationResult:
    cats = [Category(id="notes", name="메모")]
    moved = [
        MovedFile(
            original_path=tmp_path / "a.txt",
            new_path=tmp_path / "메모" / "a.txt",
            category_id="notes",
            reason="plain text",
            score=0.9,
        )
    ]
    now = datetime.now().astimezone()
    return OperationResult(
        target_root=tmp_path,
        started_at=now,
        finished_at=now,
        dry_run=False,
        categories=cats,
        moved=moved,
        skipped=[],
        total_scanned=1,
    )


def test_record_and_search(tmp_path):
    db = IndexDB(tmp_path / "idx.db")
    op_id = db.record_operation(_sample_op(tmp_path))
    assert op_id

    hits = db.search("a")
    assert any("a.txt" in h.new_path for h in hits)

    ops = db.list_operations()
    assert len(ops) == 1
    assert ops[0].moved_count == 1
    db.close()


def test_rollback_restores_files(tmp_path):
    # Create the actual files so rollback can move them back.
    notes = tmp_path / "메모"
    notes.mkdir()
    f = notes / "a.txt"
    f.write_text("hello")
    op = _sample_op(tmp_path)
    op.moved[0].original_path = tmp_path / "a.txt"
    op.moved[0].new_path = f

    db = IndexDB(tmp_path / "idx.db")
    op_id = db.record_operation(op)
    res = db.rollback(op_id)
    assert res.restored == 1
    assert (tmp_path / "a.txt").exists()
    assert not f.exists()
    db.close()
