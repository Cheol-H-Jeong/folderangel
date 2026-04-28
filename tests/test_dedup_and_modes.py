"""Duplicate detection + new/incremental mode."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from folderangel.dedup import find_duplicate_groups, remove_duplicate_files
from folderangel.models import FileEntry


def _entry(path: Path, size: int) -> FileEntry:
    dt = datetime.now(tz=timezone.utc)
    return FileEntry(
        path=path, name=path.name, ext=path.suffix.lower(),
        size=size, mime="", created=dt, modified=dt, accessed=dt,
        content_excerpt="",
    )


def test_dedup_finds_byte_identical_files(tmp_path):
    body = b"X" * (2 * 1024 * 1024)
    a = tmp_path / "doc.pdf"
    b = tmp_path / "subdir" / "doc-copy.pdf"
    b.parent.mkdir()
    a.write_bytes(body)
    b.write_bytes(body)
    groups = find_duplicate_groups(
        [_entry(a, len(body)), _entry(b, len(body))],
        min_bytes=1_048_576,
    )
    assert len(groups) == 1
    g = groups[0]
    assert g.canonical.path == a
    assert [d.path for d in g.duplicates] == [b]
    assert g.total_bytes_freed == len(body)


def test_dedup_skips_below_threshold(tmp_path):
    a = tmp_path / "small_a.txt"
    b = tmp_path / "small_b.txt"
    a.write_bytes(b"x" * 1024)
    b.write_bytes(b"x" * 1024)
    groups = find_duplicate_groups(
        [_entry(a, 1024), _entry(b, 1024)], min_bytes=1_048_576,
    )
    assert groups == []


def test_dedup_skips_different_content(tmp_path):
    a = tmp_path / "a.bin"
    b = tmp_path / "b.bin"
    a.write_bytes(b"A" * (2 << 20))
    b.write_bytes(b"B" * (2 << 20))
    groups = find_duplicate_groups(
        [_entry(a, a.stat().st_size), _entry(b, b.stat().st_size)],
        min_bytes=1_048_576,
    )
    assert groups == []


def test_remove_duplicate_files_deletes_dupes(tmp_path):
    body = b"Z" * (2 << 20)
    canon = tmp_path / "keep.dat"
    dup1 = tmp_path / "copy1.dat"
    dup2 = tmp_path / "copy2.dat"
    for p in (canon, dup1, dup2):
        p.write_bytes(body)
    groups = find_duplicate_groups(
        [_entry(canon, len(body)), _entry(dup1, len(body)), _entry(dup2, len(body))],
        min_bytes=1_048_576,
    )
    actions = remove_duplicate_files(groups, dry_run=False)
    assert len(actions) == 2
    assert canon.exists()
    assert not dup1.exists()
    assert not dup2.exists()


def test_seed_categories_from_disk(tmp_path):
    from folderangel.pipeline import _seed_categories_from_disk
    (tmp_path / "1. 의약품 AI 심사 〈2025-2026〉").mkdir()
    (tmp_path / "2. 행안부 범정부 AI (2024)").mkdir()
    (tmp_path / "9. 기타").mkdir()
    (tmp_path / ".hidden").mkdir()
    seeds = _seed_categories_from_disk(tmp_path)
    names = {s["name"] for s in seeds}
    assert names == {"의약품 AI 심사", "행안부 범정부 AI", "기타"}
    by_name = {s["name"]: s for s in seeds}
    assert by_name["의약품 AI 심사"]["time_label"] == "2025-2026"
    assert by_name["행안부 범정부 AI"]["time_label"] == "2024"


def test_seed_categories_skips_non_dir(tmp_path):
    from folderangel.pipeline import _seed_categories_from_disk
    f = tmp_path / "not-a-dir.txt"
    f.write_text("x")
    assert _seed_categories_from_disk(f) == []


def test_planner_passes_seeds_to_rolling(tmp_path):
    """Planner.__init__ accepts seed_categories and the rolling
    planner uses them as the initial cum_cats — no LLM call needed
    for that to be testable, just inspect the attribute."""
    from folderangel.config import Config
    from folderangel.planner import Planner
    seeds = [
        {"id": "drug-ai", "name": "의약품 AI 심사",
         "description": "", "duration": "annual",
         "time_label": "2025", "group": 1},
    ]
    p = Planner(Config(), seed_categories=seeds)
    assert p.seed_categories == seeds
