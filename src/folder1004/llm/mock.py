"""Deterministic heuristic planner used when no API key is configured, or when
Gemini calls fail.  Goal: produce *reasonable* categorisation even offline, so
the pipeline is testable and resilient.
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Iterable


# extension → (category_id, human name)
_EXT_GROUPS = {
    # documents
    ".pdf": ("documents", "문서(PDF·리포트)"),
    ".doc": ("documents", "문서(PDF·리포트)"),
    ".docx": ("documents", "문서(PDF·리포트)"),
    ".odt": ("documents", "문서(PDF·리포트)"),
    ".rtf": ("documents", "문서(PDF·리포트)"),
    ".hwp": ("hangul_docs", "한글 문서"),
    ".hwpx": ("hangul_docs", "한글 문서"),
    # presentations
    ".ppt": ("slides", "프레젠테이션"),
    ".pptx": ("slides", "프레젠테이션"),
    ".key": ("slides", "프레젠테이션"),
    # spreadsheets
    ".xls": ("spreadsheets", "스프레드시트"),
    ".xlsx": ("spreadsheets", "스프레드시트"),
    ".csv": ("spreadsheets", "스프레드시트"),
    # notes / text
    ".txt": ("notes", "메모·노트"),
    ".md": ("notes", "메모·노트"),
    ".markdown": ("notes", "메모·노트"),
    # images
    ".png": ("images", "이미지"),
    ".jpg": ("images", "이미지"),
    ".jpeg": ("images", "이미지"),
    ".gif": ("images", "이미지"),
    ".webp": ("images", "이미지"),
    ".heic": ("images", "이미지"),
    ".bmp": ("images", "이미지"),
    ".svg": ("images", "이미지"),
    # video
    ".mp4": ("videos", "영상"),
    ".mov": ("videos", "영상"),
    ".mkv": ("videos", "영상"),
    ".avi": ("videos", "영상"),
    ".webm": ("videos", "영상"),
    # audio
    ".mp3": ("audio", "오디오"),
    ".wav": ("audio", "오디오"),
    ".flac": ("audio", "오디오"),
    ".m4a": ("audio", "오디오"),
    # archives
    ".zip": ("archives", "압축·아카이브"),
    ".tar": ("archives", "압축·아카이브"),
    ".gz": ("archives", "압축·아카이브"),
    ".7z": ("archives", "압축·아카이브"),
    ".rar": ("archives", "압축·아카이브"),
    # code
    ".py": ("code", "코드·개발"),
    ".js": ("code", "코드·개발"),
    ".ts": ("code", "코드·개발"),
    ".json": ("code", "코드·개발"),
    ".yaml": ("code", "코드·개발"),
    ".yml": ("code", "코드·개발"),
    ".html": ("code", "코드·개발"),
    ".css": ("code", "코드·개발"),
    ".sh": ("code", "코드·개발"),
    ".cpp": ("code", "코드·개발"),
    ".c": ("code", "코드·개발"),
    ".go": ("code", "코드·개발"),
    ".rs": ("code", "코드·개발"),
    # installers
    ".exe": ("installers", "설치·실행파일"),
    ".msi": ("installers", "설치·실행파일"),
    ".dmg": ("installers", "설치·실행파일"),
    ".deb": ("installers", "설치·실행파일"),
    ".rpm": ("installers", "설치·실행파일"),
    ".appimage": ("installers", "설치·실행파일"),
}


_KEYWORD_HINTS = [
    # (regex, category_id, name)
    (re.compile(r"(invoice|영수증|청구|견적|receipt)", re.I), "receipts", "영수증·청구서"),
    (re.compile(r"(resume|이력서|self.intro|자기소개서|cover\s*letter)", re.I), "resumes", "이력서·자기소개"),
    (re.compile(r"(contract|계약|nda|합의)", re.I), "contracts", "계약·합의"),
    (re.compile(r"(lecture|강의|syllabus|수업|강좌)", re.I), "lectures", "강의·수업자료"),
    (re.compile(r"(paper|논문|thesis|dissertation)", re.I), "papers", "논문·연구자료"),
    (re.compile(r"(report|리포트|보고서|월간|분기)", re.I), "reports", "보고서"),
    (re.compile(r"(book|ebook|도서|소설)", re.I), "books", "책·도서"),
    (re.compile(r"(photo|screenshot|사진|스크린샷|캡처)", re.I), "screenshots", "스크린샷·사진"),
    (re.compile(r"(meeting|회의|회의록|minutes)", re.I), "meetings", "회의·미팅"),
    (re.compile(r"(travel|여행|ticket|예약|항공)", re.I), "travel", "여행·예약"),
]


def _keyword_category(text: str):
    for rx, cid, name in _KEYWORD_HINTS:
        if rx.search(text):
            return cid, name
    return None


def plan(files: Iterable[dict], ambiguity_threshold: float = 0.15) -> dict:
    """Return a structure compatible with the planner's final shape.

    Output: {"categories": [...], "assignments": [...]}.
    """
    files = list(files)
    assignments: list[dict] = []
    category_buckets: dict[str, dict] = {}

    for f in files:
        name = f.get("name", "")
        excerpt = f.get("excerpt", "") or ""
        ext = f.get("ext", "").lower()

        score = 0.6
        cat = _keyword_category(name + "\n" + excerpt)
        if cat is not None:
            cid, cname = cat
            score = 0.8
            desc = "파일명/본문 키워드로 식별"
        else:
            grp = _EXT_GROUPS.get(ext)
            if grp is not None:
                cid, cname = grp
                desc = "확장자 기반 분류"
            else:
                cid, cname = "misc", "기타"
                desc = "매칭된 카테고리 없음"
                score = 0.4

        category_buckets.setdefault(cid, {"id": cid, "name": cname, "description": desc})
        assignments.append(
            {
                "path": f.get("path"),
                "primary": cid,
                "primary_score": round(score, 3),
                "secondary": [],
                "reason": f"{desc} ({ext or 'noext'})",
            }
        )

    # Collapse categories with < 2 files into "misc" unless only one category exists.
    counts = Counter(a["primary"] for a in assignments)
    keep_ids = {cid for cid, n in counts.items() if n >= 2}
    if not keep_ids or len(keep_ids) < 2:
        keep_ids = set(category_buckets.keys())
    for a in assignments:
        if a["primary"] not in keep_ids:
            a["primary"] = "misc"
            a["reason"] = a["reason"] + " (분류 수가 적어 기타로 병합)"
            category_buckets.setdefault(
                "misc", {"id": "misc", "name": "기타", "description": "수량이 적어 묶은 파일"}
            )

    # Produce ordered category list: most populated first, but cap at 12.
    final_counts = Counter(a["primary"] for a in assignments)
    ordered_ids = [cid for cid, _ in final_counts.most_common()]
    categories = [category_buckets[cid] for cid in ordered_ids if cid in category_buckets][:12]

    # Drop assignments whose primary id did not survive.
    surviving = {c["id"] for c in categories}
    for a in assignments:
        if a["primary"] not in surviving:
            a["primary"] = "misc"
            if not any(c["id"] == "misc" for c in categories):
                categories.append({"id": "misc", "name": "기타", "description": "기타 파일"})
                surviving.add("misc")

    return {"categories": categories, "assignments": assignments}
