from pathlib import Path

import pytest

from folder1004.scanner import ScanTooLargeError, scan


def _touch(p: Path, data: bytes = b"x"):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)


def test_non_recursive_collects_only_top_level(tmp_path):
    _touch(tmp_path / "a.txt")
    _touch(tmp_path / "sub" / "b.txt")
    files = scan(tmp_path, recursive=False)
    names = sorted(p.name for p in files)
    assert names == ["a.txt"]


def test_recursive_collects_nested(tmp_path):
    _touch(tmp_path / "a.txt")
    _touch(tmp_path / "sub" / "b.txt")
    files = scan(tmp_path, recursive=True)
    names = sorted(p.name for p in files)
    assert names == ["a.txt", "b.txt"]


def test_ignore_patterns_skip_hidden(tmp_path):
    _touch(tmp_path / ".hidden")
    _touch(tmp_path / "a.txt")
    files = scan(tmp_path, recursive=False, ignore_patterns=[".*"])
    assert [p.name for p in files] == ["a.txt"]


def test_symlink_not_followed(tmp_path):
    sub = tmp_path / "real"
    sub.mkdir()
    _touch(sub / "t.txt")
    link = tmp_path / "link"
    try:
        link.symlink_to(sub, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlink not supported")
    files = scan(tmp_path, recursive=True)
    assert all("link" not in p.parts for p in files)


def test_max_files_exceeded(tmp_path):
    for i in range(5):
        _touch(tmp_path / f"f{i}.txt")
    with pytest.raises(ScanTooLargeError):
        scan(tmp_path, recursive=False, max_files=3)
