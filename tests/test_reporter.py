from datetime import datetime
from pathlib import Path

from folderangel.models import Category, MovedFile, OperationResult
from folderangel.reporter import emit_markdown


def test_emit_markdown(tmp_path):
    now = datetime.now().astimezone()
    op = OperationResult(
        target_root=tmp_path,
        started_at=now,
        finished_at=now,
        dry_run=False,
        categories=[Category(id="notes", name="메모")],
        moved=[
            MovedFile(
                original_path=tmp_path / "a.txt",
                new_path=tmp_path / "메모" / "a.txt",
                category_id="notes",
                reason="txt",
                score=0.9,
            )
        ],
        skipped=[],
        total_scanned=1,
    )
    path = emit_markdown(op, tmp_path)
    assert path.exists()
    text = path.read_text(encoding="utf-8")
    assert "FolderAngel Report" in text
    assert "notes" in text
