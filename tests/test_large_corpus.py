"""Stress / efficiency tests for the large-corpus path."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from folder1004.rolling import signature
from folder1004.config import Config
from folder1004.models import FileEntry
from folder1004.planner import Planner


def _entry(name: str, ts: float = 1700000000.0, content: str = "") -> FileEntry:
    p = Path(f"/work/{name}")
    return FileEntry(
        path=p,
        name=name,
        ext=p.suffix.lower(),
        size=len(content) or 1024,
        created=datetime.fromtimestamp(ts, tz=timezone.utc).astimezone(),
        modified=datetime.fromtimestamp(ts, tz=timezone.utc).astimezone(),
        accessed=datetime.fromtimestamp(ts, tz=timezone.utc).astimezone(),
        mime="application/x-test",
        content_excerpt=content,
    )


# ----- signature -----------------------------------------------------------

def test_signature_collapses_versioned_korean_filenames():
    names = [
        "한국지역정보개발원_제안발표_240301_v0.5_투이컨설팅.pptx",
        "한국지역정보개발원_제안발표_240304_R1.pptx",
        "★한국지역정보개발원_제안발표_240302_v0.5_작성요청 (1).pptx",
        "한국지역정보개발원_제안발표_240308_최종_4.pptx",
        "한국지역정보개발원 제안발표 2024-03-12 최종본.pptx",
    ]
    sigs = {signature(n) for n in names}
    # All five must collapse to a single signature.
    assert len(sigs) == 1, f"expected single sig, got {sigs}"


def test_signature_keeps_distinct_projects_distinct():
    a = signature("AVOCA_특허임시명세서_240820.pptx")
    b = signature("한국지역정보개발원_정성제안서_v1.0.pptx")
    c = signature("초거대AI_공통기반_목표모델정의서_HF_1028.pptx")
    assert a != b != c != a


# ----- planner: rolling-window propagation ---------------------------------

class _FakeClient:
    """Tiny stub matching the surface ``Planner._llm_call`` needs.

    Returns a fixed plan that classifies every representative under a
    ``proj-N`` category derived from its filename prefix.  The point of
    the test isn't LLM accuracy — it's that the hierarchical path
    propagates the rep's assignment to every cluster member.
    """

    def __init__(self):
        self.calls = 0

    def generate_json(self, prompt, **_kw):
        self.calls += 1
        import re
        # Detect rolling-window prompt format vs legacy format.
        is_rolling = '"i":' in prompt and '"n":' in prompt
        if is_rolling:
            # Rolling: respond with new_categories + fid-keyed assignments.
            seen_cats: dict[str, str] = {}
            cats: list[dict] = []
            assigns: list[dict] = []
            for m in re.finditer(r'\{"i":(\d+),"n":"([^"]+)"', prompt):
                fid, fname = int(m.group(1)), m.group(2)
                pm = re.match(r"([A-Za-z]+)_제안서_v\d+\.pdf", fname)
                cid = f"proj-{pm.group(1)}" if pm else "misc"
                if cid not in seen_cats and cid != "misc":
                    seen_cats[cid] = cid
                    cats.append({
                        "id": cid,
                        "name": cid.replace("-", " ").title(),
                        "group": 1, "time_label": "", "duration": "mixed",
                    })
                assigns.append({"i": fid, "c": cid, "p": 0.9, "r": "fake"})
            return {"new_categories": cats, "assignments": assigns}

        # Legacy format: parse rep filenames from /work/ paths.
        names = re.findall(r"/work/([^\"\\\s]+)", prompt)
        seen_cats: dict[str, str] = {}
        cats: list[dict] = []
        assigns: list[dict] = []
        for name in names:
            m = re.match(r"([A-Za-z]+)_제안서_v\d+\.pdf", name)
            if m:
                cid = f"proj-{m.group(1)}"
            else:
                cid = "misc"
            if cid not in seen_cats:
                seen_cats[cid] = cid
                cats.append({
                    "id": cid,
                    "name": cid.replace("-", " ").title(),
                    "group": 1,
                    "time_label": "",
                    "duration": "mixed",
                })
            assigns.append({
                "path": f"/work/{name}",
                "primary": cid,
                "primary_score": 0.9,
                "secondary": [],
                "reason": "테스트 가짜 분류",
            })
        return {"categories": cats, "assignments": assigns}


def test_hierarchical_propagates_rep_assignment_to_all_members():
    """5 projects × 20 versions = 100 files.  Hierarchical plan must:
        * make a single (or very few) LLM calls,
        * yield exactly 100 file assignments (no member dropped),
        * place every member of cluster N into category proj-N.
    """
    cfg = Config()
    cfg.small_corpus_files = 10      # so 100 files isn't "small"
    cfg.hierarchical_min_files = 50  # force hierarchical at this size
    cfg.cluster_min_size = 3
    cfg.economy_max_files = 200      # let single-call attempt fit

    entries: list[FileEntry] = []
    base_ts = 1700000000.0
    names = ["alpha", "bravo", "charlie", "delta", "echo"]
    for proj in range(5):
        for ver in range(20):
            entries.append(_entry(
                f"{names[proj]}_제안서_v{ver}.pdf",
                ts=base_ts + proj * 10000 + ver * 60,
            ))

    fake = _FakeClient()
    p = Planner(cfg, gemini=fake)
    plan = p.plan(entries)

    # Hierarchical path used → very few LLM calls (1, possibly 2 if
    # the long-tail singleton path also fired).
    assert fake.calls <= 2, f"expected ≤2 LLM calls, got {fake.calls}"

    # Every file got an assignment.
    assert len(plan.assignments) == 100

    # Members inherited their cluster's category.
    expected = {f"proj-{n}": 0 for n in names}
    for a in plan.assignments:
        if a.primary_category_id in expected:
            expected[a.primary_category_id] += 1
    assert all(v == 20 for v in expected.values()), expected




def test_time_guess_requires_token_overlap_with_category(tmp_path):
    """The "시기로 추정" rescue (``_guess_by_time``) must NOT pull a
    file into a project category just because the file's mtime fell
    inside that category's time window.  Without an additional
    token-overlap check the planner kept dumping unrelated files
    (1152.PDF, IMG_8933.jpeg, NTS_eTaxInvoice.html, RTX PRO 6000 GPU
    구매 계약.pdf) into whichever project was currently active —
    e.g. "행안부_범정부AI공통기반_문서인식 〈2025-2026〉".

    This test reproduces that exact failure: feed a planner result
    that leaves the unrelated files unassigned, and assert that they
    end up in misc/기타 rather than getting "시기로 추정" snapped to
    the project category whose 2025-2026 window happens to contain
    their mtime.
    """
    from datetime import datetime, timezone
    from folder1004.models import Category
    from folder1004.planner import _guess_by_time

    project_cats = [
        Category(
            id="haengan-doc-recog",
            name="행안부 범정부AI공통기반 문서인식",
            description="범정부 AI 공통기반 문서인식 사업",
            time_label="2025–2026",
            duration="multi-year",
            group=2,
        ),
        Category(id="misc", name="기타", description="", group=9),
    ]

    # All three files have mtime *inside* the project window (2025–2026)
    # but no token overlap with the category — must NOT be matched.
    in_window_ts = datetime(2025, 6, 1, tzinfo=timezone.utc).timestamp()
    for fname in [
        "1152.PDF",
        "IMG_8933.jpeg",
        "NTS_eTaxInvoice.html",
        "RTX PRO 6000 GPU 3대 구매 계약.pdf",
        "4aL6Fv3rfd1N2lTHeCvhROHBhuY.mp4",
    ]:
        e = _entry(fname, ts=in_window_ts, content="")
        guess = _guess_by_time(e, project_cats)
        assert guess is None, (
            f"_guess_by_time pulled '{fname}' into '{guess}' purely on "
            f"timestamp — should have required token overlap"
        )

    # Counter-example — a file that DOES share tokens with the category
    # SHOULD be matched even if its mtime is inside the same window.
    e = _entry(
        "행안부_문서인식_사업계획서.pdf",
        ts=in_window_ts,
        content="문서인식 사업계획서 본문",
    )
    guess = _guess_by_time(e, project_cats)
    assert guess == "haengan-doc-recog", (
        f"file with clear keyword overlap got '{guess}' — expected match"
    )


def test_time_guess_rejects_generic_two_letter_ascii_overlap(tmp_path):
    """A 2-char ASCII abbreviation like "AI" / "ML" / "VR" appears in
    almost every filename AND every category name in an AI-leaning
    corpus, so it must not by itself satisfy the token-overlap check.
    Past leak: every student presentation got snapped to a drug-AI
    project just because both contained "AI".
    """
    from datetime import datetime, timezone
    from folder1004.models import Category
    from folder1004.planner import _guess_by_time, _tokens_overlap

    drug_ai = [
        Category(
            id="drug-ai-project",
            name="의약품 AI 심사 및 산업지원 체계 구축",
            description="의약품 AI 심사·산업지원 사업",
            time_label="2025-2026",
            duration="multi-year",
            group=1,
        ),
        Category(id="misc", name="기타", description="", group=9),
    ]
    in_window_ts = datetime(2025, 11, 1, tzinfo=timezone.utc).timestamp()

    # Real student-presentation filenames the user reported.  Each has
    # "AI" in its name and falls inside the drug-AI project's window —
    # but shares no SUBSTANTIVE token with the project's name.
    student_files = [
        "김민지kimminji2031_197578_10350428_11주차_ AI 치매예방돌봄 로봇 다솜.pptx",
        "조수빈chosubin2095_223631_10348706_로봇 공학_조수빈 2023038095.pptx",
        "김이은kimieun2021_222707_10357446_로봇 공학_ 로봇의 설계, 제작 및 운영을 지원하는 AI 기술.pptx",
        "장윤성jangyoonsung2009_225560_10350506_로봇의 두뇌를 학습시키는 가상 세계, NVIDIA Isaac.pptx",
        "박규리parkgyuri2016_224614_10409047_소프트뱅크-softvoice.pptx",
        "이서진leeseojin2066_224805_10418060_12 KT의 믿음_이서진.pptx",
    ]
    for fname in student_files:
        e = _entry(fname, ts=in_window_ts, content="")
        guess = _guess_by_time(e, drug_ai)
        assert guess is None, (
            f"student presentation '{fname}' was snapped to '{guess}' on "
            f"the strength of a bare 'AI' overlap — _is_substantive_token "
            f"should reject 2-char ASCII abbreviations"
        )

    # Direct unit assertions on _tokens_overlap so a future edit to
    # _guess_by_time can't silently bypass this rule.
    assert _tokens_overlap({"ai"}, {"ai", "의약품"}) is False
    assert _tokens_overlap({"ai", "ml"}, {"ai", "ml", "의약품"}) is False
    assert _tokens_overlap({"의약품", "ai"}, {"ai", "의약품"}) is True
    # Korean 2-char tokens (감정/분석/로봇) are still substantive.
    assert _tokens_overlap({"감정", "분석"}, {"감정", "분석", "ai"}) is True
    assert _tokens_overlap({"로봇"}, {"로봇", "공학"}) is True


def test_proper_noun_extraction_restricts_to_named_entities():
    """The ``_guess_by_time`` rescue path must extract NNP + SL≥3 + SH +
    NNG≥3 only — generic NNG (지원/운영/체계) and 2-char ASCII (AI/ML)
    must NOT survive into the proper-noun set, because they're the
    tokens that caused the original "시기로 추정" leak.
    """
    from folder1004.morph import extract_proper_nouns, is_available

    if not is_available():
        import pytest
        pytest.skip("kiwipiepy not installed in this environment")

    # The leak signal: "AI 지원 운영 체계 구축" is pure generic
    # vocabulary — must yield NO proper nouns.
    pn = set(extract_proper_nouns("AI 지원 운영 체계 구축 심사 산업"))
    assert pn == set(), (
        f"generic NNG/SL2 tokens leaked into proper-noun set: {pn}"
    )

    # Real project markers must come through.
    pn = set(extract_proper_nouns("한양대 인간-인공지능 협업"))
    assert "한양대" in pn, (
        f"한양대 (NNP institution) missing from proper nouns: {pn}"
    )

    pn = set(extract_proper_nouns("의약품 AI 심사 및 산업지원"))
    assert "의약품" in pn, (
        f"의약품 (3-char NNG domain word) missing: {pn}"
    )
    assert "ai" not in pn, "2-char SL must be excluded"
    assert "지원" not in pn, "2-char NNG must be excluded"

    pn = set(extract_proper_nouns("행안부 범정부 AI 공통기반"))
    assert "행안부" in pn, (
        f"행안부 (3-char institution acronym) missing: {pn}"
    )

    # Person names must NOT leak as project markers.
    pn = set(extract_proper_nouns("김민지 kimminji 박규리 parkgyuri"))
    assert "김민지" not in pn, f"김민지 leaked as proper noun: {pn}"
    assert "박규리" not in pn, f"박규리 leaked as proper noun: {pn}"
    # Latin variants (≥3 chars) DO survive — they're useful as
    # filename-fingerprint tokens for student-project matching.
    assert "kimminji" in pn

    # Brand acronyms ≥ 3 chars: kept.
    pn = set(extract_proper_nouns("RTX PRO 6000 GPU 3대 구매"))
    assert "rtx" in pn and "gpu" in pn


def test_similarity_module_signal_axes():
    """Each of the 5 axes (S1 file core PN / S2 schema / S3 time /
    S4 path / S5 body PN) must score independently — verify with
    contrived inputs that the signal really fires."""
    from datetime import date
    from pathlib import Path
    from folder1004 import similarity as sim

    # Two files with identical schemas (학생 발표자료 batch shape).
    a = sim.signals_for_entry(_entry(
        "김민지kimminji2031_197578_10350428_11주차_ AI 치매예방돌봄 로봇 다솜.pptx"
    ))
    b = sim.signals_for_entry(_entry(
        "박규리parkgyuri2016_224614_10409047_소프트뱅크-softvoice.pptx"
    ))
    # Same student-batch schema → S2 should be high
    assert sim._normalised_lev(a.schema, b.schema) > 0.6, (a.schema, b.schema)

    # Filename-core noun stripping must remove the student-id block.
    assert "kimminji" not in a.core_stem, a.core_stem
    assert "11주차" in a.core_stem  # ← content survives

    # Two files with same parent dir → S4 = 1.0
    p_drug = Path("/x/의약품/file.pdf")
    p_drug2 = Path("/x/의약품/another.pdf")

    class _E:
        def __init__(self, path):
            self.path = path
            self.name = path.name
            self.content_excerpt = ""
            from datetime import datetime as dt
            self.modified = dt(2025, 11, 1)

    sa = sim.signals_for_entry(_E(p_drug))
    sb = sim.signals_for_entry(_E(p_drug2))
    cat_sig = sim.category_signals(
        {"name": "의약품 AI 심사", "description": "", "duration": "annual"},
        members=[sa],
    )
    score = sim.s4_path(sb, cat_sig.parent_paths)
    assert score == 1.0


def test_filename_core_strips_student_id_block():
    """The student-id-shaped prefix (`한글이름englishname2031_NNNNNN_NNNNNNNN_`)
    must be stripped *before* proper-noun extraction, so the file's
    *content* tokens (11주차, 로봇, 다솜) drive matching — not the
    Latin transliteration of the student's name (kimminji), which
    would otherwise tie the file to a hash bucket of arbitrary
    other students' Latin tokens.
    """
    from folder1004.similarity import signals_for_entry, _strip_filename_for_core

    sig = signals_for_entry(_entry(
        "김민지kimminji2031_197578_10350428_11주차_ AI 치매예방돌봄 로봇 다솜.pptx"
    ))
    assert "kimminji" not in sig.core_stem
    assert "197578" not in sig.core_stem
    # Numeric prefix patterns: "1. 의약품…" must drop "1." too
    assert _strip_filename_for_core("1. 의약품 AI 심사 제안서") == "의약품 AI 심사 제안서"
    # Version / date / draft postfix
    assert _strip_filename_for_core("범초공_제안서_v1.2_20251110_초안") in (
        "범초공_제안서",
        "범초공 제안서",
        "범초공_제안서_20251110",  # accept either order of stripping
    ) or _strip_filename_for_core("범초공_제안서_v1.2_20251110_초안").startswith("범초공_제안서")


def test_compatibility_rejects_student_to_drug_ai():
    """End-to-end: feed the user-reported student presentations and
    the drug-AI category to ``similarity.compatibility`` and confirm
    the score falls below ``THRESHOLD_GUESS_BY_TIME`` so the rescue
    refuses the snap.  The 의약품 file goes the other way."""
    from datetime import date
    from folder1004.models import Category
    from folder1004 import similarity as sim

    drug_cat = Category(
        id="drug-ai",
        name="의약품 AI 심사 및 산업지원 체계 구축",
        description="의약품 AI 심사·산업지원 사업",
        time_label="2025-2026",
        duration="multi-year",
        group=1,
    )
    cat_sig = sim.category_signals(drug_cat, time_range=(date(2025, 1, 1), date(2026, 12, 31)))

    student_files = [
        "김민지kimminji2031_197578_10350428_11주차_ AI 치매예방돌봄 로봇 다솜.pptx",
        "조수빈chosubin2095_223631_10348706_로봇 공학_조수빈 2023038095.pptx",
        "박규리parkgyuri2016_224614_10409047_소프트뱅크-softvoice.pptx",
        "이서진leeseojin2066_224805_10418060_12 KT의 믿음_이서진.pptx",
        "장윤성jangyoonsung2009_225560_10350506_로봇의 두뇌를 학습시키는 가상 세계, NVIDIA Isaac.pptx",
    ]
    for fname in student_files:
        f_sig = sim.signals_for_entry(_entry(fname))
        score = sim.compatibility(f_sig, cat_sig, reclassify_mode=True)
        assert score < sim.THRESHOLD_GUESS_BY_TIME, (
            f"student '{fname}' scored {score:.3f} ≥ rescue threshold "
            f"{sim.THRESHOLD_GUESS_BY_TIME} — leak"
        )

    # Counter-example: real 의약품 file MUST clear the threshold.
    f_sig = sim.signals_for_entry(_entry("의약품 AI 심사 제안서.pdf"))
    score = sim.compatibility(f_sig, cat_sig, reclassify_mode=True)
    assert score >= sim.THRESHOLD_GUESS_BY_TIME, (
        f"의약품 file scored only {score:.3f}, expected ≥ rescue threshold"
    )


def test_rolling_capacity_and_chunk_sizing():
    """The rolling planner's per-call capacity must be derived from
    the *effective* context window, not the advertised one."""
    from folder1004 import rolling
    from folder1004.config import Config

    cfg = Config()
    cfg.model = "gemini-2.5-flash"
    cap = rolling.estimate_files_capacity(cfg)
    assert 300 <= cap <= 700, f"flash effective cap looks wrong: {cap}"

    cfg.model = "gemini-2.5-pro"
    cap = rolling.estimate_files_capacity(cfg)
    assert 600 <= cap <= 1500, f"pro effective cap looks wrong: {cap}"

    cfg.model = "claude-sonnet-4"
    cap = rolling.estimate_files_capacity(cfg)
    assert 500 <= cap <= 1100, f"sonnet effective cap looks wrong: {cap}"

    # Unknown model + tiny assumed ctx → effective ≈ 4K, all eaten by
    # overhead → cap == 0 → ``should_use_rolling`` returns False and
    # the corpus goes through the legacy micro-batch path.
    cfg.model = "totally-unknown-model"
    cfg.assumed_ctx_tokens = 8192
    cap = rolling.estimate_files_capacity(cfg)
    assert cap < rolling.MIN_CHUNK_FILES, (
        f"unknown small-ctx model expected to fall through, cap={cap}"
    )


def test_rolling_skips_for_tiny_corpus_or_starved_ctx():
    """should_use_rolling must NOT fire for sub-MIN_CHUNK corpora
    (too small) or for models that can't take MIN_CHUNK_FILES per
    call (better routed to micro-batch)."""
    from folder1004 import rolling
    from folder1004.config import Config

    cfg = Config()
    cfg.model = "gemini-2.5-flash"
    assert rolling.should_use_rolling(cfg, n_files=10) is False
    assert rolling.should_use_rolling(cfg, n_files=200) is True

    cfg.model = "tiny-local"
    cfg.assumed_ctx_tokens = 4096   # too small for the rolling path
    assert rolling.should_use_rolling(cfg, n_files=200) is False


def test_rolling_collapses_duplicate_signatures():
    """Files with identical signatures (e.g. 100 weekly invoices that
    differ only by date) must collapse into a single FileRow — sending
    100 nearly-identical rows wastes the prompt budget."""
    from folder1004 import rolling

    entries = (
        [_entry(f"강의평가_{i:03}.pdf", content="강의 평가 의견 본문") for i in range(50)]
        + [_entry("의약품 AI 심사 제안서.pdf", content="의약품 AI 심사 제안서 본문")]
    )
    rows = rolling.build_rows(entries)
    # All 50 lecture-eval files share a sig + body slice → collapse to 1
    eval_rows = [r for r in rows if r.name.startswith("강의평가_")]
    drug_rows = [r for r in rows if r.name.startswith("의약품")]
    assert len(eval_rows) == 1, (
        f"expected duplicate collapse, got {len(eval_rows)} rows"
    )
    assert eval_rows[0].members and len(eval_rows[0].members) == 50
    assert len(drug_rows) == 1


def test_rolling_end_to_end_with_fake_llm(tmp_path):
    """Walk the full rolling-plan path with a fake LLM that returns a
    minimal response — verify fid expansion replicates the assignment
    to every collapsed sibling."""
    from folder1004.config import Config
    from folder1004.planner import Planner

    cfg = Config()
    cfg.model = "gemini-2.5-flash"
    cfg.reclassify_mode = True
    cfg.small_corpus_files = 1   # force out of "small" tier
    cfg.min_category_size = 1    # disable singleton absorption — this
                                 # test checks raw rolling behaviour, the
                                 # singleton absorption pass has its own
                                 # dedicated test below.

    seen_prompts: list[str] = []

    class _Fake:
        def generate_json(self, prompt, **_kw):
            seen_prompts.append(prompt)
            # Look at the file rows in the prompt and assign each fid.
            # Trust that the prompt JSON has "files":[{"i":N,...}].
            import re as _re
            ids = [int(m) for m in _re.findall(r'"i"\s*:\s*(\d+)', prompt)]
            # Distinct: fids appear in files block + possibly in
            # categories — but we're early so categories are empty.
            new_cats = [
                {"id": "drug-ai", "name": "의약품 AI 심사",
                 "description": "", "duration": "annual",
                 "time_label": "2025", "group": 1},
                {"id": "lecture-eval", "name": "강의 평가",
                 "description": "", "duration": "mixed",
                 "time_label": "", "group": 2},
            ]
            assigns = []
            # Decide based on filename in the prompt.  Crude.
            for m in _re.finditer(
                r'\{"i":(\d+),"n":"([^"]+)"', prompt
            ):
                fid, fname = int(m.group(1)), m.group(2)
                if "강의평가" in fname:
                    assigns.append({"i": fid, "c": "lecture-eval",
                                    "p": 0.9, "r": "강의평가 자료"})
                elif "의약품" in fname:
                    assigns.append({"i": fid, "c": "drug-ai",
                                    "p": 0.95, "r": "의약품 AI 심사"})
                else:
                    assigns.append({"i": fid, "c": "misc",
                                    "p": 0.3, "r": "단서 부족"})
            return {"new_categories": new_cats, "assignments": assigns}

    entries = (
        [_entry(f"강의평가_{i:03}.pdf", content="강의 평가") for i in range(50)]
        + [_entry("의약품 AI 심사 제안서.pdf", content="의약품 AI 심사 제안서 본문")]
        + [_entry("정체불명_파일_x9z.bin", content="")]
    )

    p = Planner(cfg, gemini=_Fake())
    plan = p.plan(entries)

    assert len(plan.assignments) == len(entries), (
        f"every entry must get assigned, got {len(plan.assignments)}/{len(entries)}"
    )
    eval_assigns = [a for a in plan.assignments
                    if a.file_path.name.startswith("강의평가_")]
    assert len(eval_assigns) == 50
    # All 50 should have inherited the same category.
    assert {a.primary_category_id for a in eval_assigns} == {"lecture-eval"}
    drug = [a for a in plan.assignments
            if a.file_path.name.startswith("의약품")]
    assert drug[0].primary_category_id == "drug-ai"


def test_singleton_absorption_collapses_tiny_categories():
    """A category with only 1–2 members must be absorbed into the
    closest larger category (proper-noun overlap) or into misc, so
    the user does not see a forest of 1-file folders.
    """
    from folder1004.planner import Planner

    cfg = Config()
    cfg.model = "gemini-2.5-flash"
    cfg.reclassify_mode = True
    cfg.small_corpus_files = 1
    cfg.min_category_size = 3   # default — but state it for the test record

    class _Fake:
        def generate_json(self, prompt, **_kw):
            import re
            ids = [
                int(m.group(1)) for m in re.finditer(
                    r'\{"i":(\d+),"n":"([^"]+)"', prompt
                )
            ]
            files = list(re.finditer(r'\{"i":(\d+),"n":"([^"]+)"', prompt))
            new_cats = [
                {"id": "lecture-ai", "name": "AI 강의자료",
                 "description": "AI 수업 발표·과제",
                 "duration": "annual", "time_label": "2025", "group": 1},
                {"id": "drug-ai", "name": "의약품 AI 심사",
                 "description": "의약품 AI 심사 사업",
                 "duration": "annual", "time_label": "2025", "group": 2},
                {"id": "rtx-buy", "name": "RTX GPU 구매계약",
                 "description": "RTX 구매 1건",
                 "duration": "burst", "time_label": "2025", "group": 3},
            ]
            assigns = []
            for m in files:
                fid, fname = int(m.group(1)), m.group(2)
                if "강의" in fname or "AI 발표" in fname:
                    cid = "lecture-ai"
                elif "의약품" in fname:
                    cid = "drug-ai"
                elif "RTX" in fname:
                    cid = "rtx-buy"
                else:
                    cid = "misc"
                assigns.append({"i": fid, "c": cid, "p": 0.9, "r": "fake"})
            return {"new_categories": new_cats, "assignments": assigns}

    # 50 lecture files (large), 2 drug-ai (small), 1 rtx (singleton),
    # plus 1 odd file → ≥ MIN_CHUNK_FILES so rolling fires.
    entries = (
        [_entry(f"AI 발표_{i:03}.pptx", content="AI 강의 발표") for i in range(50)]
        + [_entry("의약품 AI 제안서.pdf", content="의약품 AI 제안서"),
           _entry("의약품 AI 보고서.pdf", content="의약품 AI 보고서")]
        + [_entry("RTX 구매계약.pdf", content="RTX 구매 계약")]
    )

    # Bypass should_use_rolling tier checks by calling _rolling_plan directly,
    # then feeding through _plan_from_dict to apply singleton absorption.
    p = Planner(cfg, gemini=_Fake())
    payloads = [e.to_summary_dict() for e in entries]
    raw = p._rolling_plan(entries, payloads, progress=None)
    from folder1004.planner import _plan_from_dict
    plan = _plan_from_dict(raw, entries, reclassify_mode=True)
    cat_ids = {c.id for c in plan.categories}
    assert "lecture-ai" in cat_ids
    # drug-ai (2 files < 3) and rtx-buy (1 file < 3) must be absorbed.
    assert "drug-ai" not in cat_ids, f"singleton drug-ai survived: {cat_ids}"
    assert "rtx-buy" not in cat_ids, f"singleton rtx-buy survived: {cat_ids}"
    # Their assignments are redirected — never left dangling.
    assert all(a.primary_category_id in cat_ids | {"misc"}
               for a in plan.assignments)


def test_tier_picker_picks_correct_tier():
    cfg = Config()
    # Use the production defaults so this test catches regressions to
    # the user-test thresholds (small ≤ 60, large ≥ 100).
    p = Planner(cfg, gemini=_FakeClient())

    small = [_entry(f"alpha_보고서_v{i}.pdf") for i in range(20)]
    assert p._pick_tier(small) == "small"

    # 80 files: above small (60) but below large (100) → medium
    medium = [_entry(f"alpha_보고서_v{i}.pdf") for i in range(80)]
    assert p._pick_tier(medium) == "medium"


def test_tier_picker_uses_count_only_not_collapse():
    """User-test policy: file count alone decides the tier.  Even a
    100+ corpus where every filename is unique (no signature collapse)
    must still be classified ``large`` so the hierarchical path runs."""
    cfg = Config()
    p = Planner(cfg, gemini=_FakeClient())
    entries = [_entry(f"distinct_{i}_보고서_v0.pdf") for i in range(119)]
    assert p._pick_tier(entries) == "large"

    # 50 distinct projects × 100 versions = 5,000 → large
    proj_names = [
        "프로젝트", "사업", "과제", "정책", "연구", "교육", "마케팅",
        "재무", "영업", "기술", "운영", "전략", "기획", "지원", "관리",
        "개발", "분석", "평가", "검토", "도입", "확산", "보안", "구축",
        "조사", "조달", "수행", "컨설팅", "협력", "교류", "혁신",
        "표준", "인증", "예산", "감사", "내부", "외부", "정보",
        "데이터", "통신", "네트워크", "플랫폼", "서비스", "콘텐츠",
        "프로그램", "이벤트", "실증", "검증", "행정", "투자", "리서치",
    ]
    large = [_entry(f"{proj_names[i]}_보고서_v{j}.pdf")
             for i in range(50) for j in range(100)]
    assert p._pick_tier(large) == "large"


def test_mock_mode_still_announces_tier():
    """Even without an LLM client we should tell the user which tier
    *would* have been picked, so the file-count → strategy mapping is
    visible regardless of API key configuration."""
    cfg = Config()
    p = Planner(cfg, gemini=None)  # no client → mock path
    entries = [_entry(f"alpha_보고서_v{i}.pdf") for i in range(120)]
    seen: list[str] = []
    p.plan(entries, progress=lambda msg, pct: seen.append(msg))
    tier_msgs = [m for m in seen if "모드" in m]
    assert any("대규모" in m for m in tier_msgs), tier_msgs


def test_tier_announcement_describes_chosen_mode():
    cfg = Config()
    p = Planner(cfg, gemini=_FakeClient())
    msg = p._tier_announcement("small", 50)
    assert "소규모" in msg and "50" in msg and "한 번" in msg
    msg = p._tier_announcement("medium", 350)
    assert "중간" in msg and "micro-batch" in msg
    msg = p._tier_announcement("large", 5000)
    assert "대규모" in msg and "5000" in msg and "대표" in msg


