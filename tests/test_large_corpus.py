"""Stress / efficiency tests for the large-corpus path."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from folderangel.cluster import (
    Cluster,
    cluster_files,
    collapse_ratio,
    signature,
)
from folderangel.config import Config
from folderangel.models import FileEntry
from folderangel.planner import Planner


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


# ----- clustering ----------------------------------------------------------

def test_clustering_produces_long_tail_for_singletons():
    e1 = _entry("proj_alpha_v1.md")
    e2 = _entry("proj_alpha_v2.md")
    e3 = _entry("proj_alpha_v3.md")
    e4 = _entry("oneoff_misc.txt")
    clusters, long_tail = cluster_files([e1, e2, e3, e4], min_cluster_size=3)
    assert len(clusters) == 1
    assert clusters[0].size == 3
    assert long_tail == [e4]


def test_collapse_ratio_drops_for_large_repetitive_corpus():
    entries: list[FileEntry] = []
    # 50 distinct projects × 100 versions each = 5,000 files, very
    # repetitive on purpose.
    base_ts = 1700000000.0
    # 50 distinct project names whose Korean morpheme-tokenisation
    # yields a different first-noun for each project.  Without that
    # the signatures collapse to one cluster (which would be the
    # *correct* answer for that input but defeats the purpose of the
    # test).
    proj_names = [
        "프로젝트", "사업", "과제", "정책", "연구", "교육", "마케팅",
        "재무", "영업", "기술", "운영", "전략", "기획", "지원", "관리",
        "개발", "분석", "평가", "검토", "도입", "확산", "보안", "구축",
        "조사", "조달", "수행", "컨설팅", "협력", "교류", "혁신",
        "표준", "인증", "예산", "감사", "내부", "외부", "정보",
        "데이터", "통신", "네트워크", "플랫폼", "서비스", "콘텐츠",
        "프로그램", "이벤트", "실증", "검증", "행정", "투자", "리서치",
    ]
    assert len(proj_names) == len(set(proj_names)) >= 50
    for proj in range(50):
        for version in range(100):
            entries.append(_entry(
                f"{proj_names[proj]}_보고서_v{version}.pdf",
                ts=base_ts + proj * 10000 + version * 60,
            ))
    clusters, long_tail = cluster_files(entries, min_cluster_size=3)
    # 50 clusters of 100 members each, no long tail.
    assert len(clusters) == 50
    assert long_tail == []
    ratio = collapse_ratio(len(entries), clusters, len(long_tail))
    # Sending up to 2 reps per cluster ≈ 100/5000 = 2 % of the data.
    assert ratio <= 0.05


# ----- planner: hierarchical decision + propagation ------------------------

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
        # Parse rep filenames out of the prompt — tests synthesise paths
        # like /work/알파_007_v3.pdf, so a substring scan suffices.
        import re
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


def test_embedding_first_cluster_groups_by_body(tmp_path):
    """Two filenames whose tokens differ but whose bodies talk about
    the same project should now end up in one cluster — option F."""
    from folderangel.cluster import cluster_files

    body_alpha = "범정부 초거대 AI 공통기반 사업 BPR/ISP 추진 계획"
    body_alpha2 = "초거대 AI 공통기반 BPR_ISP 작업 보고서"
    entries = [
        _entry("한국지역정보개발원_제안발표_v1.pptx", content=body_alpha),
        _entry("투이컨설팅_제안발표_v2.pptx", content=body_alpha2),
        _entry("공통기반_플랫폼_v1.pdf", content=body_alpha),
    ]
    clusters, long_tail = cluster_files(entries, min_cluster_size=2,
                                        primary_embedding=True)
    # Only one cluster of three semantically-related files (assuming
    # sklearn is available; if not, the test still asserts no crash).
    assert sum(c.size for c in clusters) + len(long_tail) == 3
    # All three should be in *some* cluster of size ≥ 2 if sklearn is on.
    sizes = sorted([c.size for c in clusters], reverse=True)
    assert sizes[:1] == [3] or sizes[:1] == [2]  # 2 acceptable when fallback


def test_outlier_demoted_to_individual_classification(tmp_path):
    """A cluster member whose body looks unrelated to the rep should
    be sent for individual LLM classification, not blindly inherit
    the rep's category."""
    from folderangel.planner import Planner

    cfg = Config()
    cfg.hierarchical_min_files = 5
    cfg.cluster_min_size = 2
    cfg.reps_per_cluster = 2
    cfg.outlier_min_similarity = 0.30

    seen_extra: list[list[dict]] = []

    class _Fake:
        def __init__(self):
            self.calls = 0

        def generate_json(self, prompt, **_kw):
            self.calls += 1
            # First call: hand back the project category.
            if self.calls == 1:
                return {
                    "categories": [{"id": "proj", "name": "ProjectX",
                                    "group": 1, "duration": "annual"}],
                    "assignments": [
                        {"path": "/work/proj_v1.pdf", "primary": "proj",
                         "primary_score": 0.9, "secondary": [],
                         "reason": "rep"},
                        {"path": "/work/proj_v2.pdf", "primary": "proj",
                         "primary_score": 0.9, "secondary": [],
                         "reason": "rep"},
                    ],
                }
            # Subsequent call: outlier individual re-classify.
            data = {"assignments": [
                {"path": "/work/something_else.pdf", "primary": "proj",
                 "primary_score": 0.5, "secondary": [], "reason": "outlier-fallback"},
            ]}
            seen_extra.append(data["assignments"])
            return data

    fake = _Fake()
    p = Planner(cfg, gemini=fake)
    entries = [
        _entry("proj_v1.pdf", ts=1700000000.0, content="ProjectX summary v1"),
        _entry("proj_v2.pdf", ts=1700000050.0, content="ProjectX summary v2"),
        _entry("proj_v3.pdf", ts=1700000100.0, content="ProjectX summary v3"),
        # outlier — same filename pattern, totally different body
        _entry("something_else.pdf", ts=1700000150.0,
               content="totally unrelated topic about cooking recipes ramen"),
    ]
    plan = p.plan(entries)

    # Find the outlier's assignment.  It must NOT carry the
    # "동일 패턴 클러스터 자동 상속 — rep" reason of the proj cluster
    # — that would mean the outlier inherited a category it doesn't
    # actually belong to.  Either it was demoted to individual
    # classification (extra LLM call) or it landed in long-tail
    # singletons.
    outlier_assigns = [a for a in plan.assignments
                       if a.file_path.name == "something_else.pdf"]
    assert outlier_assigns, "outlier missing from plan"
    reason = outlier_assigns[0].reason or ""
    assert "자동 상속" not in reason, (
        f"outlier inherited cluster category: {reason!r}"
    )


def test_longtail_discover_creates_new_category_for_unrelated_file(tmp_path):
    """A file that doesn't fit any cluster (long-tail) must reach the
    LLM via the discover-or-assign call, and a *new* category proposed
    there must be merged into the final plan.  This guards the user
    flow: '관련이 없으면 롱테일로 넘겨서 LLM이 추가 카테고리 생성'."""
    cfg = Config()
    # Force the hierarchical tier on a tiny synthetic corpus so we can
    # exercise the rep-call → long-tail-discover handoff deterministically.
    cfg.small_corpus_files = 3
    cfg.hierarchical_min_files = 4
    cfg.cluster_min_size = 2
    cfg.reps_per_cluster = 2
    cfg.outlier_min_similarity = 0.45

    class _Fake:
        def __init__(self):
            self.calls = 0

        def generate_json(self, prompt, **_kw):
            self.calls += 1
            if self.calls == 1:
                # First call: only the cluster reps reach the LLM
                # (long-tail is held back for the second call).
                return {
                    "categories": [{"id": "proj", "name": "ProjectX",
                                    "group": 1, "duration": "annual"}],
                    "assignments": [
                        {"path": "/work/proj_v1.pdf", "primary": "proj",
                         "primary_score": 0.9, "secondary": [],
                         "reason": "rep"},
                        {"path": "/work/proj_v2.pdf", "primary": "proj",
                         "primary_score": 0.9, "secondary": [],
                         "reason": "rep"},
                    ],
                }
            # Second call: long-tail discover.  LLM proposes a NEW
            # category 'insurance-policy' for the 약관 file.
            return {
                "new_categories": [
                    {"id": "insurance-policy", "name": "원더풀S 통합보험 약관",
                     "group": 2, "duration": "mixed"},
                ],
                "assignments": [
                    {"path": "/work/insurance_terms.pdf",
                     "primary": "insurance-policy",
                     "primary_score": 0.85, "secondary": [],
                     "reason": "약관 전용 폴더 신설"},
                ],
            }

    fake = _Fake()
    p = Planner(cfg, gemini=fake)
    entries = [
        _entry("proj_v1.pdf", ts=1700000000.0, content="ProjectX summary v1"),
        _entry("proj_v2.pdf", ts=1700000050.0, content="ProjectX summary v2"),
        _entry("proj_v3.pdf", ts=1700000100.0, content="ProjectX summary v3"),
        # Long-tail singleton — completely different domain (insurance).
        _entry("insurance_terms.pdf", ts=1700000150.0,
               content="무배당 원더풀S 통합보험 약관 보험금 청구 면책"),
    ]
    plan = p.plan(entries)

    cat_ids = {c.id for c in plan.categories}
    assert "insurance-policy" in cat_ids, (
        f"new category from long-tail discover not merged: {cat_ids}"
    )
    insurance_assigns = [
        a for a in plan.assignments if a.file_path.name == "insurance_terms.pdf"
    ]
    assert insurance_assigns, "insurance file missing from plan"
    assert insurance_assigns[0].primary_category_id == "insurance-policy", (
        "insurance file did not land in the new category"
    )
    # Exactly two LLM calls — reps + long-tail.  No silent fallback.
    assert fake.calls == 2, f"unexpected LLM call count: {fake.calls}"


def test_reclassify_filename_first_pass_then_body_pass(tmp_path):
    """Re-classify mode runs a filename-only pre-pass.  Files the LLM
    can confidently classify by name alone (e.g. those with the project
    keyword in the filename) get assigned right there.  Generic-named
    files (e.g. "약관.pdf") are deferred to the body-aware tier
    pipeline below, which may discover entirely new categories from
    body content."""
    cfg = Config()
    cfg.reclassify_mode = True
    # Force the medium tier so the filename pass triggers but a single
    # body call covers the deferred slice.
    cfg.small_corpus_files = 3
    cfg.hierarchical_min_files = 100
    cfg.cluster_min_size = 2
    cfg.reps_per_cluster = 2

    class _Fake:
        def __init__(self):
            self.calls = 0
            self.prompts: list[str] = []

        def generate_json(self, prompt, **_kw):
            self.calls += 1
            self.prompts.append(prompt)
            if self.calls == 1:
                # Filename-first pass.  Classifies the proj_X files by
                # filename alone, defers the generic 약관/보고서 files.
                return {
                    "categories": [
                        {"id": "proj-x", "name": "ProjectX",
                         "group": 1, "duration": "annual"},
                    ],
                    "assignments": [
                        {"path": "/work/projx_overview.pdf",
                         "primary": "proj-x", "primary_score": 0.92,
                         "secondary": [], "reason": "파일명에 ProjectX"},
                        {"path": "/work/projx_v2.pdf",
                         "primary": "proj-x", "primary_score": 0.92,
                         "secondary": [], "reason": "파일명에 ProjectX"},
                    ],
                    "deferred": [
                        "/work/약관.pdf",
                        "/work/보고서.docx",
                    ],
                }
            # Second call: deferred subset goes through normal pipeline.
            # Mock returns a new category.
            return {
                "categories": [
                    {"id": "insurance-policy",
                     "name": "원더풀S 통합보험 약관",
                     "group": 2, "duration": "mixed"},
                ],
                "assignments": [
                    {"path": "/work/약관.pdf",
                     "primary": "insurance-policy",
                     "primary_score": 0.85,
                     "secondary": [], "reason": "본문이 보험 약관"},
                    {"path": "/work/보고서.docx",
                     "primary": "insurance-policy",
                     "primary_score": 0.6,
                     "secondary": [], "reason": "본문 유사"},
                ],
            }

    fake = _Fake()
    p = Planner(cfg, gemini=fake)
    entries = [
        _entry("projx_overview.pdf", ts=1700000000.0,
               content="ProjectX overview milestone summary"),
        _entry("projx_v2.pdf", ts=1700000050.0,
               content="ProjectX v2 release notes"),
        _entry("약관.pdf", ts=1700000100.0,
               content="무배당 원더풀S 통합보험 약관 보험금 청구 면책"),
        _entry("보고서.docx", ts=1700000150.0,
               content="원더풀S 약관 검토 보고 — 면책 조항"),
    ]
    plan = p.plan(entries)

    cat_ids = {c.id for c in plan.categories}
    assert "proj-x" in cat_ids, f"filename-pass category missing: {cat_ids}"
    assert "insurance-policy" in cat_ids, (
        f"deferred-pass category missing: {cat_ids}"
    )

    # First call must be the filename-only pass — body excerpts must
    # not appear in that prompt.
    pass1 = fake.prompts[0]
    assert "ProjectX overview milestone" not in pass1, (
        "filename-first pass leaked body content"
    )
    assert "원더풀S 통합보험 약관" not in pass1, (
        "filename-first pass leaked body content"
    )
    # But filename does.
    assert "projx_overview.pdf" in pass1
    assert "약관.pdf" in pass1

    # Confidently-named files carry the filename-pass marker.
    px = [a for a in plan.assignments if a.file_path.name == "projx_overview.pdf"]
    assert px and "파일명 패스" in (px[0].reason or "")
    # Deferred files were not assigned in pass 1, they took the body path.
    yk = [a for a in plan.assignments if a.file_path.name == "약관.pdf"]
    assert yk and yk[0].primary_category_id == "insurance-policy"


def test_opaque_filenames_force_deferred_in_filename_pass(tmp_path):
    """The exact files the user reported wrongly landed in
    "한양대 인간-인공지능 협업 제품서비스 설계":
      - 1152.PDF                   pure numeric
      - 1767000341906.pdf          pure numeric (timestamp-like)
      - 4aL6Fv3rfd1N2lTHeCvhROHBhuY.mp4   random hash
      - IMG_8933.jpeg              camera auto-name
    Must never reach the filename LLM call — they carry no project
    identity in their name and the LLM was lumping them into the
    most-active project category."""
    from folderangel.planner import _is_opaque_filename

    assert _is_opaque_filename("1152.PDF", ".pdf")
    assert _is_opaque_filename("1767000341906.pdf", ".pdf")
    assert _is_opaque_filename("4aL6Fv3rfd1N2lTHeCvhROHBhuY.mp4", ".mp4")
    assert _is_opaque_filename("IMG_8933.jpeg", ".jpeg")
    # Counter-examples — these have real project signals.
    assert not _is_opaque_filename(
        "한국지역정보개발원_제안발표.pptx", ".pptx"
    )
    assert not _is_opaque_filename(
        "RTX PRO 6000 GPU 3대 구매 계약.pdf", ".pdf"
    )
    assert not _is_opaque_filename("projx_overview.pdf", ".pdf")


def test_filename_pass_keyword_overlap_veto_demotes_to_deferred(tmp_path):
    """When the LLM confidently assigns a file but the filename has
    zero substantive token overlap with the target category, the
    veto must demote that file to deferred so the body-aware pass
    gets a fair shot at it.

    Reproduces "RTX PRO 6000 GPU 3대 구매 계약.pdf" being shoved into
    한양대 협업 — the GPU contract has zero overlap with the
    한양대/협업/인공지능 keyword set, so it must defer."""
    cfg = Config()
    cfg.reclassify_mode = True
    cfg.small_corpus_files = 3
    cfg.hierarchical_min_files = 100
    cfg.cluster_min_size = 2
    cfg.reps_per_cluster = 2

    class _Fake:
        def __init__(self):
            self.calls = 0

        def generate_json(self, prompt, **_kw):
            self.calls += 1
            if self.calls == 1:
                # LLM tries to confidently shove the GPU file into
                # the 한양대 category.  The veto must catch this.
                return {
                    "categories": [
                        {"id": "hanyang-ai", "name": "한양대 인간-인공지능 협업 제품서비스 설계",
                         "keywords": ["한양대", "협업", "인공지능"],
                         "group": 1, "duration": "annual"},
                    ],
                    "assignments": [
                        {"path": "/work/한양대_협업_발표.pptx",
                         "primary": "hanyang-ai", "primary_score": 0.9,
                         "secondary": [], "reason": "파일명에 한양대"},
                        {"path": "/work/RTX PRO 6000 GPU 3대 구매 계약.pdf",
                         "primary": "hanyang-ai", "primary_score": 0.9,
                         "secondary": [], "reason": "AI 연구용 GPU"},
                    ],
                    "deferred": [],
                }
            # Pass 2 (body-aware) returns a proper procurement category.
            return {
                "categories": [
                    {"id": "gpu-procure", "name": "GPU 구매·계약",
                     "group": 2, "duration": "mixed"},
                ],
                "assignments": [
                    {"path": "/work/RTX PRO 6000 GPU 3대 구매 계약.pdf",
                     "primary": "gpu-procure", "primary_score": 0.9,
                     "secondary": [], "reason": "본문이 구매 계약"},
                ],
            }

    fake = _Fake()
    p = Planner(cfg, gemini=fake)
    entries = [
        _entry("한양대_협업_발표.pptx", ts=1700000000.0,
               content="한양대 인간-AI 협업 제품서비스 설계 발표"),
        _entry("RTX PRO 6000 GPU 3대 구매 계약.pdf", ts=1700000050.0,
               content="발주처 매수인 RTX PRO 6000 단가 납품 계약"),
        _entry("협업_가이드라인.docx", ts=1700000100.0,
               content="협업 절차 가이드"),
        _entry("AI 협업 결과보고.pdf", ts=1700000150.0,
               content="협업 결과 정리"),
    ]
    plan = p.plan(entries)

    # The legitimately-named one must land in hanyang-ai.
    hy = [a for a in plan.assignments if a.file_path.name == "한양대_협업_발표.pptx"]
    assert hy and hy[0].primary_category_id == "hanyang-ai"

    # The GPU contract must NOT have inherited hanyang-ai via Pass 1.
    gpu = [a for a in plan.assignments
           if a.file_path.name == "RTX PRO 6000 GPU 3대 구매 계약.pdf"]
    assert gpu, "GPU file disappeared"
    assert gpu[0].primary_category_id != "hanyang-ai", (
        f"keyword-overlap veto failed — GPU file still in 한양대: "
        f"{gpu[0].primary_category_id}"
    )


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
    from folderangel.models import Category
    from folderangel.planner import _guess_by_time

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
    from folderangel.models import Category
    from folderangel.planner import _guess_by_time, _tokens_overlap

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
    from folderangel.morph import extract_proper_nouns, is_available

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


def test_hierarchical_skipped_for_small_corpora():
    """Below ``hierarchical_min_files`` threshold the planner must NOT
    pick the hierarchical path even if files would cluster.
    """
    cfg = Config()
    cfg.hierarchical_min_files = 500
    cfg.cluster_min_size = 3
    fake = _FakeClient()
    p = Planner(cfg, gemini=fake)
    entries = [_entry(f"알파_001_v{i}.pdf", ts=1700000000.0 + i)
               for i in range(20)]
    # Should NOT pick the hierarchical path; use the existing single-call
    # economy.  We only check the decision function here so we don't
    # accidentally exercise the full planner on a stub.
    assert p._should_use_hierarchical(entries, [e.to_summary_dict() for e in entries]) is False
