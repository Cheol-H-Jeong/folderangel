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

    canonical = tmp_path / "1. AVOCA 시스템 (2024-Q3)"
    assert canonical.is_dir()
    assert (canonical / "old.pptx").exists()
    assert (canonical / "new.pptx").exists()
    # Stats should reflect the absorbed leftover too.
    assert op.total_moved >= 2
