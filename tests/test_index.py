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


def _make_recorded_op(tmp_path: Path, src_name: str, dst_dir: str):
    from folderangel.models import Category, MovedFile, OperationResult
    folder = tmp_path / dst_dir
    folder.mkdir(exist_ok=True)
    f = folder / src_name
    f.write_text("x")
    cats = [Category(id="c", name=dst_dir)]
    moved = [
        MovedFile(
            original_path=tmp_path / src_name,
            new_path=f,
            category_id="c",
            reason="x",
            score=1.0,
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


def test_rollback_refuses_older_op_without_force(tmp_path):
    db = IndexDB(tmp_path / "idx.db")
    op_id_1 = db.record_operation(_make_recorded_op(tmp_path, "a.txt", "f1"))
    db.record_operation(_make_recorded_op(tmp_path, "b.txt", "f2"))  # newer
    res = db.rollback(op_id_1)  # not the latest — must refuse
    assert res.restored == 0
    assert res.failed and "not the most recent" in res.failed[0]
    db.close()


def test_rollback_force_skips_collisions(tmp_path):
    db = IndexDB(tmp_path / "idx.db")
    op_id_1 = db.record_operation(_make_recorded_op(tmp_path, "a.txt", "f1"))
    db.record_operation(_make_recorded_op(tmp_path, "b.txt", "f2"))
    # Simulate the user manually creating a file at the original location
    # in the meantime — rollback must NOT overwrite it even with force.
    blocker = tmp_path / "a.txt"
    blocker.write_text("user-edit")
    res = db.rollback(op_id_1, force=True)
    assert res.restored == 0
    assert any("already occupied" in msg for msg in res.failed)
    # And the user's file is intact.
    assert blocker.read_text() == "user-edit"
    db.close()


def test_latest_operation_id(tmp_path):
    db = IndexDB(tmp_path / "idx.db")
    assert db.latest_operation_id() is None
    a = db.record_operation(_make_recorded_op(tmp_path, "a.txt", "f1"))
    b = db.record_operation(_make_recorded_op(tmp_path, "b.txt", "f2"))
    assert db.latest_operation_id() == b
    assert b > a
    db.close()
