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
