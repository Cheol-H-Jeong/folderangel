"""Sanitiser must reject corrupt LLM output; recovery must cope with
truncated JSON that ran out of ``max_tokens`` mid-stream.
"""
from folder1004.organizer import sanitize_folder_name
from folder1004.llm.client import _recover_truncated_json


def test_replacement_char_is_stripped():
    s = sanitize_folder_name("한국지�역정보�개발원")
    assert "�" not in s
    assert "한국지" in s and "역정보" in s


def test_bom_is_stripped():
    s = sanitize_folder_name("﻿회사 자료")
    assert "﻿" not in s
    assert "회사" in s


def test_control_chars_replaced():
    s = sanitize_folder_name("foo\x00bar\x1fbaz")
    assert "\x00" not in s and "\x1f" not in s
    assert "foo" in s and "baz" in s


def test_json_fragment_leakage_cleaned():
    s = sanitize_folder_name('"name":"AVOCA 시스템"')
    # The leading "name":"  and trailing " all stripped; AVOCA preserved
    assert s.strip().startswith("AVOCA")
    assert "name" not in s.lower()


def test_garbage_only_falls_back():
    s = sanitize_folder_name('���')
    assert s == "folder"


def test_recover_simple_truncation():
    text = '{"categories":[{"id":"a","name":"한'
    fixed = _recover_truncated_json(text)
    import json

    obj = json.loads(fixed)
    assert isinstance(obj, dict)
    assert obj["categories"][0]["id"] == "a"


def test_recover_preserves_existing_valid_json():
    text = '{"a": 1, "b": [1,2,3]}'
    assert _recover_truncated_json(text) == text


def test_user_reported_latin1_mojibake_folder_name_is_rejected():
    """Exact string the user saw on disk: should never reach the
    filesystem.  Per-field strict mojibake check must catch it."""
    bad = "íì ë¶ ì´ê±°ë AI ê³µíµê¸°ë° BPR_ISP"
    out = sanitize_folder_name(bad)
    assert out == "folder", f"mojibake leaked through sanitiser: {out!r}"


def test_organizer_quarantines_preexisting_mojibake_folder(tmp_path):
    """User-reported case: pre-existing mojibake-named folders must be
    cleaned up — empty ones deleted, non-empty ones merged into the
    single canonical "9. 기타" bucket (no "기타 (2)" siblings)."""
    from folder1004.config import Config
    from folder1004.models import Plan
    from folder1004.organizer import Organizer

    bad_empty = tmp_path / "6. ì ì¡° AI ì¤ì¦ ì§ì (2024)"
    bad_empty.mkdir()
    bad_kept_a = tmp_path / "7. ëª¨ì§ ë°ì½ë A"
    bad_kept_a.mkdir()
    (bad_kept_a / "leftover_a.txt").write_text("x")
    bad_kept_b = tmp_path / "8. ëª¨ì§ ë°ì½ë B"
    bad_kept_b.mkdir()
    (bad_kept_b / "leftover_b.txt").write_text("y")

    Organizer(Config()).execute(tmp_path, Plan(categories=[], assignments=[]))

    assert not bad_empty.exists()
    assert not bad_kept_a.exists()
    assert not bad_kept_b.exists()
    # Both mojibake folders' contents merged into ONE 기타 — never
    # produce "기타 (2)".
    misc = tmp_path / "9. 기타"
    assert misc.is_dir()
    assert (misc / "leftover_a.txt").exists()
    assert (misc / "leftover_b.txt").exists()
    assert not (tmp_path / "9. 기타 (2)").exists()


def test_organizer_uses_median_mtime_of_files(tmp_path):
    """Folder mtime must equal the median modified-time of the moved
    files inside it, not the LLM's time_label heuristic."""
    import os
    from datetime import datetime, timezone
    from folder1004.config import Config
    from folder1004.models import Assignment, Category, Plan
    from folder1004.organizer import Organizer

    # Three source files with known mtimes spread across years.
    targets = []
    epochs = [
        datetime(2023, 1, 15, tzinfo=timezone.utc).timestamp(),
        datetime(2024, 6, 15, tzinfo=timezone.utc).timestamp(),  # median
        datetime(2025, 9, 15, tzinfo=timezone.utc).timestamp(),
    ]
    for i, ts in enumerate(epochs):
        f = tmp_path / f"f{i}.md"
        f.write_text("x")
        os.utime(f, (ts, ts))
        targets.append(f)

    cats = [Category(id="proj", name="Alpha", group=1, time_label="2099")]
    assigns = [
        Assignment(file_path=t, primary_category_id="proj", primary_score=0.9)
        for t in targets
    ]
    Organizer(Config()).execute(tmp_path, Plan(cats, assigns))

    folder = next(p for p in tmp_path.iterdir() if p.is_dir() and "Alpha" in p.name)
    folder_mtime = folder.stat().st_mtime
    median_ts = epochs[1]
    # Allow a 2-day slop because filesystems may round mtimes.
    assert abs(folder_mtime - median_ts) < 2 * 86400, (
        f"folder mtime {folder_mtime} not near median {median_ts}"
    )


def test_safe_path_repr_redacts_mojibake_parents():
    from folder1004.llm.client import _looks_like_mojibake
    from folder1004.planner import _safe_path_repr

    bad = "/work/6. ì ì¡° AI ì¤ì¦ ì§ì (2024)/제안서_v1.pdf"
    out = _safe_path_repr(bad, _looks_like_mojibake)
    assert "ì¡°" not in out
    assert "[unknown-folder]" in out
    assert "제안서_v1.pdf" in out


def test_planner_drops_mojibake_category_in_otherwise_clean_response():
    """The leak path: only one of several category names is mojibake."""
    from pathlib import Path
    from folder1004.models import FileEntry
    from folder1004.planner import _plan_from_dict
    from datetime import datetime

    now = datetime.now().astimezone()
    entries = [
        FileEntry(
            path=Path("/tmp/a.md"), name="a.md", ext=".md",
            size=1, created=now, modified=now, accessed=now,
        )
    ]
    plan_dict = {
        "categories": [
            {"id": "good", "name": "한국지역정보개발원 제안사업", "group": 1},
            {"id": "bad", "name": "íì ë¶ ì´ê±°ë AI ê³µíµê¸°ë° BPR_ISP", "group": 1},
        ],
        "assignments": [
            {"path": "/tmp/a.md", "primary": "good", "primary_score": 0.9}
        ],
    }
    plan = _plan_from_dict(plan_dict, entries)
    names = [c.name for c in plan.categories]
    assert any("한국지역정보개발원" in n for n in names)
    assert not any("íì" in n for n in names), (
        f"mojibake category survived: {names!r}"
    )
