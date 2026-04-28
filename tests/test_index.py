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


def test_search_finds_by_filename_substring(tmp_path):
    db = IndexDB(tmp_path / "idx.db")
    op_id = db.record_operation(_make_recorded_op(tmp_path, "한국지역_제안서_v1.pdf", "f1"))
    hits = db.search("제안서")
    assert hits, "expected to find by Korean substring of filename"
    assert any("제안서" in h.new_path for h in hits)
    db.close()


def test_search_finds_by_content_excerpt(tmp_path):
    """A file whose name says nothing but whose content_excerpt contains
    the term should still be findable.
    """
    from folderangel.models import Category, MovedFile, OperationResult
    folder = tmp_path / "f1"
    folder.mkdir()
    f = folder / "anonymous_doc.pdf"
    f.write_text("x")
    op = OperationResult(
        target_root=tmp_path,
        started_at=datetime.now().astimezone(),
        finished_at=datetime.now().astimezone(),
        dry_run=False,
        categories=[Category(id="c", name="f1")],
        moved=[
            MovedFile(
                original_path=tmp_path / "anonymous_doc.pdf",
                new_path=f,
                category_id="c",
                reason="x",
                score=1.0,
                content_excerpt="이 문서는 한국지역정보개발원의 초거대 AI 공통기반 사업 보고서입니다.",
            )
        ],
        skipped=[],
        total_scanned=1,
    )
    db = IndexDB(tmp_path / "idx.db")
    db.record_operation(op)
    hits = db.search("초거대")
    assert hits and "anonymous_doc.pdf" in hits[0].new_path
    assert "초거대" in hits[0].snippet or "초거대" in hits[0].matched_in
    db.close()


def test_latest_operation_id(tmp_path):
    db = IndexDB(tmp_path / "idx.db")
    assert db.latest_operation_id() is None
    a = db.record_operation(_make_recorded_op(tmp_path, "a.txt", "f1"))
    b = db.record_operation(_make_recorded_op(tmp_path, "b.txt", "f2"))
    assert db.latest_operation_id() == b
    assert b > a
    db.close()


def test_record_operation_dedups_by_path(tmp_path):
    """A file that has been organised before — at any path — must not
    appear twice in the index after it is re-classified.  The user's
    pain: search returned the same file under both old and new paths,
    and the old row's link was broken."""
    db = IndexDB(tmp_path / "idx.db")
    # Run 1: organise a.txt into f1/
    op1 = _make_recorded_op(tmp_path, "a.txt", "f1")
    db.record_operation(op1)
    # Run 2: a.txt's NEW location is now f1/a.txt → it gets moved into f2/
    from folderangel.models import Category, MovedFile, OperationResult
    folder2 = tmp_path / "f2"
    folder2.mkdir()
    new_path = folder2 / "a.txt"
    # Simulate the disk move: file lives at new_path now.
    (tmp_path / "f1" / "a.txt").rename(new_path)
    op2 = OperationResult(
        target_root=tmp_path,
        started_at=datetime.now().astimezone(),
        finished_at=datetime.now().astimezone(),
        dry_run=False,
        categories=[Category(id="c", name="f2")],
        moved=[MovedFile(
            original_path=tmp_path / "f1" / "a.txt",
            new_path=new_path,
            category_id="c",
            reason="re-org",
            score=1.0,
        )],
        skipped=[],
        total_scanned=1,
    )
    db.record_operation(op2)
    hits = db.search("a.txt")
    # Exactly ONE hit — the latest location.  No stale duplicates.
    matching = [h for h in hits if h.new_path.endswith("a.txt")]
    assert len(matching) == 1, (
        f"expected single de-duped hit, got {len(matching)}: "
        f"{[h.new_path for h in matching]}"
    )
    assert matching[0].new_path == str(new_path)
    db.close()


def test_record_operation_persists_report_path(tmp_path):
    """Pipeline writes the report first then records the operation; the
    report path must round-trip through stats_json so HistoryView can
    open it on double-click."""
    db = IndexDB(tmp_path / "idx.db")
    op = _make_recorded_op(tmp_path, "a.txt", "f1")
    rp = tmp_path / "FolderAngel_Report_20260101_120000.md"
    rp.write_text("# Report")
    op.report_path = rp
    db.record_operation(op)
    ops = db.list_operations()
    assert ops and ops[0].report_path == str(rp)
    db.close()


def test_config_preset_round_trips_through_save_load(tmp_path, monkeypatch):
    """Adding a preset, saving config, reloading it must keep the
    preset list intact — covers the JSON serialisation path."""
    from folderangel.config import Config, load_config, save_config, AppPaths
    paths = AppPaths(
        root=tmp_path,
        config=tmp_path / "config.json",
        index_db=tmp_path / "idx.db",
        logs_dir=tmp_path / "logs",
    )
    paths.ensure()
    cfg = Config()
    cfg.llm_presets = [
        {"name": "회사 Gemini", "llm_provider": "gemini",
         "base_url": "", "model": "gemini-2.5-flash", "reasoning_mode": "off"},
        {"name": "로컬 Ollama", "llm_provider": "openai_compat",
         "base_url": "http://localhost:11434/v1", "model": "qwen2.5",
         "reasoning_mode": "off"},
    ]
    cfg.active_preset = "로컬 Ollama"
    save_config(cfg, paths)
    loaded = load_config(paths)
    assert loaded.active_preset == "로컬 Ollama"
    names = {p["name"] for p in loaded.llm_presets}
    assert names == {"회사 Gemini", "로컬 Ollama"}
    by_name = {p["name"]: p for p in loaded.llm_presets}
    assert by_name["로컬 Ollama"]["base_url"] == "http://localhost:11434/v1"
