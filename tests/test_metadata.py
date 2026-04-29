from pathlib import Path

from folder1004.metadata import collect


def test_basic_metadata(tmp_path):
    p = tmp_path / "example.pdf"
    p.write_bytes(b"%PDF-1.4 not really a pdf")
    entry = collect(p)
    assert entry.ext == ".pdf"
    assert entry.name == "example.pdf"
    assert entry.size > 0
    # created/modified should be populated as naive-or-aware datetimes
    assert entry.modified.year >= 2000
