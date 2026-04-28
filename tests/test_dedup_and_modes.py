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


def test_folder_signature_round_trip():
    """compose_folder_name → parse_fa_folder_name should recover the
    clean name + period, and is_folderangel_folder_name must say yes."""
    from folderangel.models import Category
    from folderangel.organizer import (
        compose_folder_name, is_folderangel_folder_name,
        parse_fa_folder_name, folder_signature,
    )
    cat = Category(
        id="drug-ai", name="의약품 AI 심사",
        time_label="2025-2026", duration="multi-year", group=2,
    )
    name = compose_folder_name(cat)
    assert is_folderangel_folder_name(name)
    parsed = parse_fa_folder_name(name)
    assert parsed is not None
    assert parsed["clean_name"] == "의약품 AI 심사"
    assert parsed["period"] == "2025-2026"
    # Signature is deterministic from the category id.
    assert parsed["signature"] in folder_signature("drug-ai")


def test_folder_signature_reject_non_fa():
    from folderangel.organizer import is_folderangel_folder_name, parse_fa_folder_name
    assert not is_folderangel_folder_name("1. 일반 폴더")
    assert not is_folderangel_folder_name("내가 손으로 만든 폴더")
    assert parse_fa_folder_name("foo (2024)") is None


def test_additive_mode_seeds_only_fa_folders(tmp_path):
    """Additive mode's seed list must come from FA folders only —
    user-created plain folders are NOT used as categories (their
    contents will be reclassified as loose files)."""
    from folderangel.pipeline import _seed_categories_from_disk
    # FA folder
    (tmp_path / "1. 의약품 AI 심사 〈2025-2026〉 [FA·a3b9c1]").mkdir()
    # User-created plain folder
    (tmp_path / "임시 작업 폴더").mkdir()
    (tmp_path / "2. 손으로 만든 (2025)").mkdir()

    fa_only = _seed_categories_from_disk(tmp_path, fa_only=True)
    all_seeds = _seed_categories_from_disk(tmp_path, fa_only=False)
    fa_names = {s["name"] for s in fa_only}
    all_names = {s["name"] for s in all_seeds}

    assert fa_names == {"의약품 AI 심사"}
    assert all_names == {"의약품 AI 심사", "임시 작업 폴더", "손으로 만든"}
    # FA seed retains its signature in the slug for stable reuse on
    # next compose_folder_name call.
    assert any(s["id"].endswith("-a3b9c1") for s in fa_only)


def test_compatibility_pulls_same_pattern_files_together():
    """Two files with the same prefix pattern but different
    date/version suffixes must score *higher* against a category
    seeded with their siblings than against an unrelated category.
    The user's pain: 강의평가_*.pdf siblings landed in different
    singleton folders because the old weights ignored title pattern.
    """
    from datetime import datetime, timezone
    from pathlib import Path
    from folderangel.models import Category, FileEntry
    from folderangel import similarity as sim

    def E(name, ext=".pdf"):
        dt = datetime.now(tz=timezone.utc)
        return FileEntry(
            path=Path(f"/x/{name}"), name=name, ext=ext, size=1, mime="",
            created=dt, modified=dt, accessed=dt, content_excerpt="",
        )

    eval_a = sim.signals_for_entry(E("강의평가_2025-01-08.pdf"))
    eval_b = sim.signals_for_entry(E("강의평가_2025-01-15.pdf"))
    eval_c = sim.signals_for_entry(E("강의평가_사회과목.pdf"))
    drug_a = sim.signals_for_entry(E("의약품 AI 심사 제안서.pdf"))
    drug_b = sim.signals_for_entry(E("의약품 AI 심사 보고서.pdf"))

    cat_eval = Category(id="lecture-eval", name="강의 평가",
                        description="", time_label="", duration="mixed", group=1)
    cat_drug = Category(id="drug-ai", name="의약품 AI 심사",
                        description="", time_label="", duration="mixed", group=2)

    eval_sig = sim.category_signals(cat_eval, members=[eval_a, eval_b])
    drug_sig = sim.category_signals(cat_drug, members=[drug_a, drug_b])

    # New 강의평가 file should score way higher against the eval cat
    # than against the drug cat, even though both have ".pdf".
    s_match = sim.compatibility(eval_c, eval_sig)
    s_other = sim.compatibility(eval_c, drug_sig)
    assert s_match > s_other + 0.10, (
        f"pattern signal too weak: match={s_match:.3f} vs other={s_other:.3f}"
    )


def test_extension_boost_clusters_media_batch():
    """A run of .mp4 files with shared name prefix should cluster on
    extension + pattern even when proper-noun overlap is sparse."""
    from datetime import datetime, timezone
    from pathlib import Path
    from folderangel.models import Category, FileEntry
    from folderangel import similarity as sim

    def E(name, ext):
        dt = datetime.now(tz=timezone.utc)
        return FileEntry(
            path=Path(f"/x/{name}"), name=name, ext=ext, size=1, mime="",
            created=dt, modified=dt, accessed=dt, content_excerpt="",
        )

    # Distinctive .mp4 batch — same prefix pattern.
    a = sim.signals_for_entry(E("회의녹화_2025-04-01.mp4", ".mp4"))
    b = sim.signals_for_entry(E("회의녹화_2025-04-08.mp4", ".mp4"))
    c = sim.signals_for_entry(E("회의녹화_2025-04-15.mp4", ".mp4"))
    cat = Category(id="meeting-rec", name="회의 녹화",
                   description="", time_label="", duration="mixed", group=1)
    cat_sig = sim.category_signals(cat, members=[a, b])
    score = sim.compatibility(c, cat_sig)
    assert score >= 0.5, f"same-prefix .mp4 batch score too low: {score:.3f}"


def test_generic_pdf_alone_does_not_force_match():
    """Two files that share *only* ``.pdf`` (no pattern, no nouns)
    must NOT score above the singleton-absorption threshold (0.20)
    in reclassify mode — which is where the real planner runs."""
    from datetime import datetime, timezone
    from pathlib import Path
    from folderangel.models import Category, FileEntry
    from folderangel import similarity as sim

    def E(name, parent, ext=".pdf"):
        dt = datetime.now(tz=timezone.utc)
        return FileEntry(
            path=Path(f"/{parent}/{name}"), name=name, ext=ext, size=1, mime="",
            created=dt, modified=dt, accessed=dt, content_excerpt="",
        )

    student = sim.signals_for_entry(E("김민지kimminji_과제11.pdf", "lecture"))
    drug_a = sim.signals_for_entry(E("의약품 AI 심사 제안서.pdf", "drug"))
    drug_b = sim.signals_for_entry(E("의약품 AI 심사 보고서.pdf", "drug"))
    cat = Category(id="drug-ai", name="의약품 AI 심사",
                   description="", time_label="", duration="mixed", group=1)
    cat_sig = sim.category_signals(cat, members=[drug_a, drug_b])
    score = sim.compatibility(student, cat_sig, reclassify_mode=True)
    assert score < 0.20, (
        f"generic .pdf overlap leaked student → drug-ai: {score:.3f}"
    )
