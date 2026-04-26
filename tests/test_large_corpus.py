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
