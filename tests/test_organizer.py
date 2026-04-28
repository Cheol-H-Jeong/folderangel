from pathlib import Path

from folderangel.config import Config
from folderangel.models import Assignment, Category, Plan, SecondaryAssignment
from folderangel.organizer import Organizer, sanitize_folder_name


def test_sanitize_invalid_chars():
    # Trailing underscore (from a stripped invalid char) is also cleaned up.
    assert sanitize_folder_name("foo/bar?") == "foo_bar"
    assert sanitize_folder_name("") == "folder"
    # "CON" is too short to survive the visible-chars heuristic; the key
    # is that bare "CON" never reaches disk as a real folder name.  Anything
    # that *does* survive must not be a Windows reserved name.
    s = sanitize_folder_name("CON 보고서")
    assert s.upper() != "CON"
    assert sanitize_folder_name(".") == "folder"  # trailing dot stripped → empty


def _find_dir(root, core: str):
    for child in root.iterdir():
        if child.is_dir() and core in child.name:
            return child
    return None


def test_execute_moves_files(tmp_path):
    a = tmp_path / "a.txt"
    a.write_text("hello")
    b = tmp_path / "b.txt"
    b.write_text("world")

    cats = [
        Category(id="notes", name="메모", group=1),
        Category(id="other", name="기타파일", group=2),
    ]
    assignments = [
        Assignment(file_path=a, primary_category_id="notes", primary_score=0.9),
        Assignment(file_path=b, primary_category_id="other", primary_score=0.8),
    ]
    cfg = Config()
    op = Organizer(cfg).execute(tmp_path, Plan(categories=cats, assignments=assignments))
    assert op.total_moved == 2
    notes_dir = _find_dir(tmp_path, "메모")
    other_dir = _find_dir(tmp_path, "기타파일")
    assert notes_dir is not None and (notes_dir / "a.txt").exists()
    assert other_dir is not None and (other_dir / "b.txt").exists()


def test_dry_run_does_not_move(tmp_path):
    a = tmp_path / "a.txt"
    a.write_text("hello")
    cats = [Category(id="notes", name="메모", group=1)]
    assignments = [Assignment(file_path=a, primary_category_id="notes", primary_score=0.9)]
    op = Organizer(Config()).execute(tmp_path, Plan(cats, assignments), dry_run=True)
    assert op.dry_run
    assert a.exists()
    assert _find_dir(tmp_path, "메모") is None


def test_shortcut_created_for_secondary(tmp_path):
    a = tmp_path / "resume.pdf"
    a.write_text("dummy")
    cats = [
        Category(id="resumes", name="이력서", group=1),
        Category(id="contracts", name="계약서", group=1),
    ]
    assignments = [
        Assignment(
            file_path=a,
            primary_category_id="resumes",
            primary_score=0.7,
            secondary=[SecondaryAssignment(category_id="contracts", score=0.65)],
        )
    ]
    op = Organizer(Config()).execute(tmp_path, Plan(cats, assignments))
    resumes_dir = _find_dir(tmp_path, "이력서")
    assert resumes_dir is not None and (resumes_dir / "resume.pdf").exists()
    # at least one shortcut path recorded
    assert op.total_shortcuts >= 1


def test_humanise_skip_reason_filenotfound_uses_korean():
    from folderangel.organizer import _humanise_skip_reason
    p = Path("/tmp/missing/file.pdf")
    out = _humanise_skip_reason(FileNotFoundError(p), p)
    assert "사라짐" in out and "/tmp/missing" not in out  # not raw path


def test_organizer_recovers_moved_source_by_basename(tmp_path):
    """If the recorded source is gone but the same file exists
    elsewhere under target_root, organizer should find and move it
    instead of skipping."""
    from folderangel.config import Config
    from folderangel.models import Assignment, Category, Plan
    from folderangel.organizer import Organizer

    real = tmp_path / "actual" / "report.pdf"
    real.parent.mkdir()
    real.write_text("doc")
    stale = tmp_path / "stale" / "report.pdf"  # never existed

    cats = [Category(id="c", name="Reports", group=1)]
    op = Organizer(Config()).execute(
        tmp_path,
        Plan(cats, [Assignment(file_path=stale, primary_category_id="c", primary_score=0.9)]),
    )
    assert op.total_moved == 1
    assert op.total_skipped == 0
    moved = op.moved[0]
    assert moved.new_path.name == "report.pdf"
    assert moved.new_path.exists()


def test_organizer_dedups_duplicate_assignments(tmp_path):
    from folderangel.config import Config
    from folderangel.models import Assignment, Category, Plan
    from folderangel.organizer import Organizer

    f = tmp_path / "doc.txt"
    f.write_text("x")
    cats = [Category(id="c", name="Docs", group=1)]
    a1 = Assignment(file_path=f, primary_category_id="c", primary_score=0.9)
    a2 = Assignment(file_path=f, primary_category_id="c", primary_score=0.9)
    op = Organizer(Config()).execute(tmp_path, Plan(cats, [a1, a2]))
    assert op.total_moved == 1
    assert op.total_skipped == 0


def test_compose_folder_name_per_duration():
    from folderangel.organizer import compose_folder_name

    burst = Category(id="a", name="제안발표", group=2,
                     time_label="2024-03", duration="burst")
    short = Category(id="b", name="AVOCA", group=1,
                     time_label="2024-Q3", duration="short")
    annual = Category(id="c", name="연간 보고", group=3,
                      time_label="2024", duration="annual")
    multi = Category(id="d", name="범정부 초거대 AI 공통기반", group=1,
                     time_label="2023–2025", duration="multi-year")
    mixed = Category(id="e", name="기타", group=9,
                     time_label="", duration="mixed")

    # Every name now carries the FA signature suffix so we can
    # detect "this is a folderangel folder" later.  The signature is
    # 6 hex chars derived from the category id.
    import re
    fa = re.compile(r"\s\[FA·[a-f0-9]{4,12}\]$")
    assert fa.search(compose_folder_name(burst))
    assert compose_folder_name(burst).startswith("2. 제안발표 (2024-03)")
    assert compose_folder_name(short).startswith("1. AVOCA (2024-Q3)")
    assert compose_folder_name(annual).startswith("3. 연간 보고 (2024)")
    name_multi = compose_folder_name(multi)
    assert name_multi.startswith("1. 범정부 초거대 AI 공통기반")
    assert "〈2023–2025〉" in name_multi
    assert fa.search(name_multi)
    assert compose_folder_name(mixed).startswith("9. 기타")
    assert fa.search(compose_folder_name(mixed))


def test_existing_folder_with_same_core_name_is_reused(tmp_path):
    pre_existing = tmp_path / "AVOCA 시스템"
    pre_existing.mkdir()
    leftover = pre_existing / "old.pptx"
    leftover.write_text("legacy")

    new_file = tmp_path / "new.pptx"
    new_file.write_text("fresh")

    cats = [Category(id="avoca", name="AVOCA 시스템", group=1, time_label="2024-Q3")]
    assignments = [Assignment(file_path=new_file, primary_category_id="avoca", primary_score=0.9)]
    op = Organizer(Config()).execute(tmp_path, Plan(cats, assignments))

    # Folder name now carries the FA signature suffix.  Find it by
    # the prefix and assert the legacy + new files both ended up
    # inside.
    matches = [d for d in tmp_path.iterdir() if d.is_dir()
               and d.name.startswith("1. AVOCA 시스템 (2024-Q3)")]
    assert matches, f"no folder matching prefix found in {list(tmp_path.iterdir())}"
    canonical = matches[0]
    assert "[FA·" in canonical.name
    assert (canonical / "old.pptx").exists()
    assert (canonical / "new.pptx").exists()
    assert op.total_moved >= 2
