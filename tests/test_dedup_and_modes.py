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


def test_report_includes_dedup_ledger(tmp_path):
    """The markdown report must list every duplicate that was deleted,
    along with its canonical and the bytes recovered — so the user can
    audit the dedup pass instead of trusting a single summary line."""
    from folderangel.models import (
        Category, MovedFile, OperationResult, LLMUsage,
    )
    from folderangel.reporter import emit_markdown
    op = OperationResult(
        target_root=tmp_path,
        started_at=datetime.now(tz=timezone.utc),
        finished_at=datetime.now(tz=timezone.utc),
        dry_run=False,
        categories=[Category(id="c", name="문서")],
        moved=[],
        skipped=[],
        total_scanned=3,
    )
    op.dupes_removed = [
        ("/dl/old/movie.mp4", "/dl/movie.mp4", 50 * 1024 * 1024),
        ("/dl/old/big.zip", "/dl/big.zip", 25 * 1024 * 1024),
    ]
    op.bytes_freed = 75 * 1024 * 1024
    op.llm_usage = LLMUsage(model="mock")
    out_path = emit_markdown(op, out_dir=tmp_path)
    text = out_path.read_text(encoding="utf-8")
    assert "## 중복 삭제 내역" in text
    assert "/dl/old/movie.mp4" in text
    assert "/dl/movie.mp4" in text
    assert "50.0 MB" in text or "50 MB" in text
    assert "75.0 MB" in text or "75 MB" in text


def test_get_api_key_does_not_leak_gemini_to_openai_compat(tmp_path, monkeypatch):
    """Switching from Gemini → openai_compat must NOT return the
    legacy gemini_api_key slot's value, which is what caused 401s
    when users picked their local Qwen preset.
    """
    from folderangel.config import (
        Config, AppPaths, get_api_key, save_config,
    )

    # Force config-level fallback path (no real keyring).  We exercise
    # the slot logic by patching the keyring lookup to mimic 'only
    # gemini slot is populated'.
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("FOLDERANGEL_OPENAI_API_KEY", raising=False)

    class FakeKeyring:
        def __init__(self):
            self.store = {("folderangel", "gemini_api_key"): "GEMINI-SECRET"}

        def get_password(self, service, slot):
            return self.store.get((service, slot))

    fake = FakeKeyring()
    monkeypatch.setattr("folderangel.config._try_keyring", lambda: fake)

    cfg = Config()
    cfg.llm_provider = "gemini"
    cfg.api_key_fallback = ""
    assert get_api_key(cfg) == "GEMINI-SECRET"

    cfg.llm_provider = "openai_compat"
    leaked = get_api_key(cfg)
    assert leaked is None, (
        f"openai_compat lookup leaked the gemini key: {leaked!r}"
    )


def test_make_llm_client_allows_local_without_key():
    """Local Ollama / vLLM endpoints don't need an API key — the
    client builder must synthesise a placeholder so the user doesn't
    have to register a fake one."""
    from folderangel.config import Config
    from folderangel.llm.client import make_llm_client, OpenAICompatClient

    cfg = Config()
    cfg.llm_provider = "openai_compat"
    cfg.llm_base_url = "http://localhost:11434/v1"
    cfg.model = "qwen2.5"
    client = make_llm_client(cfg, api_key=None)
    assert isinstance(client, OpenAICompatClient)
    # Cloud URL with no key still falls to mock.
    cfg.llm_base_url = "https://api.openai.com/v1"
    assert make_llm_client(cfg, api_key=None) is None
