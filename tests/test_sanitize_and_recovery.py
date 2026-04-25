"""Sanitiser must reject corrupt LLM output; recovery must cope with
truncated JSON that ran out of ``max_tokens`` mid-stream.
"""
from folderangel.organizer import sanitize_folder_name
from folderangel.llm.client import _recover_truncated_json


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


def test_planner_drops_mojibake_category_in_otherwise_clean_response():
    """The leak path: only one of several category names is mojibake."""
    from pathlib import Path
    from folderangel.models import FileEntry
    from folderangel.planner import _plan_from_dict
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
