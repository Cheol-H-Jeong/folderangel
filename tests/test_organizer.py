from pathlib import Path

from folderangel.config import Config
from folderangel.models import Assignment, Category, Plan, SecondaryAssignment
from folderangel.organizer import Organizer, sanitize_folder_name


def test_sanitize_invalid_chars():
    assert sanitize_folder_name("foo/bar?") == "foo_bar_"
    assert sanitize_folder_name("") == "folder"
    assert sanitize_folder_name("CON") == "_CON"
    assert sanitize_folder_name(".") == "folder"  # trailing dot stripped → empty


def test_execute_moves_files(tmp_path):
    a = tmp_path / "a.txt"
    a.write_text("hello")
    b = tmp_path / "b.txt"
    b.write_text("world")

    cats = [
        Category(id="notes", name="메모"),
        Category(id="other", name="기타파일"),
    ]
    assignments = [
        Assignment(file_path=a, primary_category_id="notes", primary_score=0.9),
        Assignment(file_path=b, primary_category_id="other", primary_score=0.8),
    ]
    cfg = Config()
    op = Organizer(cfg).execute(tmp_path, Plan(categories=cats, assignments=assignments))
    assert op.total_moved == 2
    assert (tmp_path / "메모" / "a.txt").exists()
    assert (tmp_path / "기타파일" / "b.txt").exists()


def test_dry_run_does_not_move(tmp_path):
    a = tmp_path / "a.txt"
    a.write_text("hello")
    cats = [Category(id="notes", name="메모")]
    assignments = [Assignment(file_path=a, primary_category_id="notes", primary_score=0.9)]
    op = Organizer(Config()).execute(tmp_path, Plan(cats, assignments), dry_run=True)
    assert op.dry_run
    assert a.exists()
    assert not (tmp_path / "메모" / "a.txt").exists()


def test_shortcut_created_for_secondary(tmp_path):
    a = tmp_path / "resume.pdf"
    a.write_text("dummy")
    cats = [
        Category(id="resumes", name="이력서"),
        Category(id="contracts", name="계약서"),
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
    assert (tmp_path / "이력서" / "resume.pdf").exists()
    # at least one shortcut path recorded
    assert op.total_shortcuts >= 1
