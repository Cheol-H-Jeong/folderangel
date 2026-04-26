"""Korean morpheme extraction wrapper.

We want a fast, deterministic way to pull project / agency / system
names out of a filename — and the same nouns out of a document body —
without any LLM call.  ``kiwipiepy`` is the right tool: pre-built
wheels for Linux / macOS / Windows, sub-millisecond per filename,
strong on Korean compound nouns and on Latin-letter brand tokens
(``AVOCA``, ``BPR_ISP``).

If kiwi isn't installed (e.g. an experimental environment, or a
PyInstaller bundle missing the model) we transparently fall back to a
character-class tokeniser so callers don't have to branch.

Public API:
    extract_nouns(text, *, top_k=None) -> list[str]
        Returns project-relevant nouns / proper nouns / Latin tokens
        with people-name / dates / sequence numbers / generic clerical
        nouns filtered out.  Order: order of first appearance in text.

    is_available() -> bool
        True iff kiwi (and its model) is loaded.

The module is import-safe even when kiwi is missing — the import
itself never raises.
"""
from __future__ import annotations

import re
import threading
from typing import Iterable, Optional

# ----- kiwi singleton (lazy) ----------------------------------------------

_kiwi = None
_kiwi_lock = threading.Lock()
_kiwi_unavailable = False


def _get_kiwi():
    global _kiwi, _kiwi_unavailable
    if _kiwi is not None:
        return _kiwi
    if _kiwi_unavailable:
        return None
    with _kiwi_lock:
        if _kiwi is not None:
            return _kiwi
        if _kiwi_unavailable:
            return None
        try:
            from kiwipiepy import Kiwi  # type: ignore

            _kiwi = Kiwi()
        except Exception:
            _kiwi_unavailable = True
            _kiwi = None
        return _kiwi


def is_available() -> bool:
    return _get_kiwi() is not None


# ----- noise filters -------------------------------------------------------

# Korean person-name surname list (the common ones).  Followed by
# given-name letters in the same morpheme is the classic person-name
# shape — we filter those out aggressively because they almost never
# carry project identity.
_PERSON_SURNAMES = {
    "김", "이", "박", "최", "정", "강", "조", "윤", "장", "임", "한",
    "오", "서", "신", "권", "황", "안", "송", "전", "홍", "유", "고",
    "문", "양", "손", "배", "백", "허", "남", "심", "노", "하", "곽",
    "성", "차", "주", "우", "구", "민", "류", "나", "진", "지", "엄",
    "원", "방", "변", "함", "표", "도", "선", "설", "마", "길", "연",
    "위", "표", "명", "기", "반", "라", "왕", "금",
}

# Generic clerical nouns that don't help identify a project.
_NOISE_NOUNS = {
    "복사본", "복사", "사본", "수정본", "변경본", "최종본", "최종판",
    "확정본", "발표용", "작성요청", "임시", "원본", "공유용", "초안",
    "수정", "확정", "최종", "버전", "버젼", "검토", "검토본", "회신",
    "의견", "참고", "공유", "전송", "회의", "회의록",
    # Pure time-of-year words that almost never identify a project on
    # their own.  (When they DO matter — e.g. a "워크샵" series — the
    # second token usually carries the real signal.)
    "연도", "년도", "분기", "상반기", "하반기", "월간", "주간", "오전", "오후",
}

# Tokens that are almost certainly tags / extensions, not nouns.
_EXT_LIKE = re.compile(
    r"^(?:pdf|pptx?|ppsx?|docx?|xlsx?|xls|hwp(?:x)?|txt|md|csv|tsv|"
    r"jpe?g|png|gif|webp|heic|bmp|svg|mp[34]|m4[av]|mov|webm|zip|7z|"
    r"rar|tar|gz|bz2|exe|dmg|iso|deb|rpm|appimage)$",
    re.IGNORECASE,
)

# Pure-numeric / sequence-only tokens.
_DIGITS_ONLY = re.compile(r"^\d+(?:[.,]\d+)?$")

_FALLBACK_TOKEN_RE = re.compile(r"[A-Za-z가-힣][A-Za-z가-힣0-9]+")


# ----- main API ------------------------------------------------------------

def extract_nouns(text: str, *, top_k: Optional[int] = None) -> list[str]:
    """Extract project-identity nouns from *text*, ordered by first
    appearance.

    Filters applied:
      * keep tags NNG / NNP (general / proper noun) and SL / SH
        (Latin / Hanja word — picks up brand tokens like ``AVOCA``);
      * drop pure numeric tokens, file-extension-shaped tokens,
        clerical nouns (``최종``, ``작성요청``, ``회의록``, …), and
        single-character Korean tokens that are most often a person
        surname (``김`` / ``이`` / …).  A surname is only kept if the
        *next* morpheme is also tagged NNG / NNP and isn't a typical
        given-name shape.
      * lowercase Latin tokens for case-insensitive matching.

    On environments without kiwi we still return *something* useful —
    a coarse character-class tokenisation, with the same noise filter.
    """
    if not text:
        return []
    kiwi = _get_kiwi()
    if kiwi is None:
        return _fallback(text, top_k)

    out: list[str] = []
    seen: set[str] = set()
    try:
        tokens = list(kiwi.tokenize(text))
    except Exception:
        return _fallback(text, top_k)

    i = 0
    while i < len(tokens):
        tok = tokens[i]
        form = (tok.form or "").strip()
        tag = tok.tag
        if not form:
            i += 1
            continue
        accept = False
        if tag in ("NNG", "NNP"):
            accept = True
        elif tag in ("SL", "SH"):  # Latin word / Hanja word
            accept = True

        if accept:
            norm = form.casefold() if tag in ("SL",) else form
            # Drop pure numerics, extensions, clerical nouns
            if _DIGITS_ONLY.match(norm) or _EXT_LIKE.match(norm):
                accept = False
            elif norm in _NOISE_NOUNS:
                accept = False
            # Skip 1-char Korean noun that's a known surname AND the
            # next token looks like a Korean given-name fragment.
            elif len(norm) == 1 and norm in _PERSON_SURNAMES:
                nxt = tokens[i + 1] if i + 1 < len(tokens) else None
                if nxt is not None and nxt.tag in ("NNG", "NNP") and 1 <= len(nxt.form) <= 2:
                    # Looks like a person name → skip both tokens.
                    i += 2
                    continue

        if accept and norm not in seen and len(norm) >= 2:
            out.append(norm)
            seen.add(norm)
            if top_k is not None and len(out) >= top_k:
                break
        i += 1
    return out


def _fallback(text: str, top_k: Optional[int]) -> list[str]:
    """Best-effort extractor when kiwi is unavailable."""
    out: list[str] = []
    seen: set[str] = set()
    for m in _FALLBACK_TOKEN_RE.finditer(text):
        tok = m.group(0)
        norm = tok.casefold() if tok[0].isascii() else tok
        if (
            len(norm) < 2
            or _DIGITS_ONLY.match(norm)
            or _EXT_LIKE.match(norm)
            or norm in _NOISE_NOUNS
        ):
            continue
        if norm in seen:
            continue
        out.append(norm)
        seen.add(norm)
        if top_k is not None and len(out) >= top_k:
            break
    return out
