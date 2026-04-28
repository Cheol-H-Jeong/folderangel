"""Multi-axis file/category compatibility score.

Replaces the older bag-of-tokens overlap check used by the rescue
("시기로 추정") path and the filename-first-pass veto.  Five axes, each
deriving a sub-score in [0, 1]:

    S1  filename-core proper-noun Jaccard
        Strip prefix/postfix patterns (numbered prefix "1.", versions
        "_v1", dates "_20251215", "(1)" duplicates, drafts/finals,
        student-id blocks) BEFORE running kiwi.  Then NNP+SL≥3+SH+
        NNG≥3 set Jaccard.

    S2  naming-schema similarity
        Compress the raw stem into a {Korean / ASCII / digit / sep}
        sequence — same shape ⇒ same source/process (lecture batch,
        invoice batch, …).  Normalised Levenshtein.

    S3  modified-time proximity
        Gaussian-style decay: exp(-Δdays / decay) where decay = 90 d
        for short/burst categories, 365 d otherwise.

    S4  scan-time path co-residence
        Same parent directory at scan time → 1.0; same grandparent
        → 0.5; else 0.  *Disabled in re-classify mode* — the user
        is explicitly trying to escape the existing folder layout,
        so re-anchoring on it would re-introduce the bug.

    S5  body-head proper-noun Jaccard
        First 500 chars of parsed body, NNP+SL≥3+NNG≥3 set Jaccard.

Composite::

    s = w1·S1 + w2·S2 + w4·S4 + w3·S3 + w5·S5

with the user-confirmed priority S1 ≈ S2 ≈ S4  ≫  S3 ≈ S5
(weights 0.27 · 3  +  0.095 · 2  =  1.00).  In re-classify mode S4 is
forced to 0 and the remaining weights are renormalised so the score
range stays [0, 1].
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime
from functools import lru_cache
from pathlib import Path
from typing import Optional

from . import morph as _morph


# --- prefix / postfix patterns to strip before noun extraction ------------

# Leading "1.", "2-", "3) ", "(4)" — numeric prefixes the user adds to
# manually order folders and files.
_PREFIX_NUM = re.compile(r"^\s*[\(\[]?\d+[\.\-_)\]\s]+")

# Trailing tokens that carry NO project identity — versions, dates,
# clerical descriptors, duplicate markers, redaction tags.
_POSTFIX_PATTERNS = [
    re.compile(r"_v\d+(?:\.\d+)?\s*$", re.IGNORECASE),
    re.compile(r"_r\d+\s*$", re.IGNORECASE),
    re.compile(r"_\d{6,8}\s*$"),                   # _251124 / _20251215
    re.compile(r"\s*\(\d+\)\s*$"),                 # "(1)" duplicate marker
    re.compile(
        r"_(?:초안|최종|확정|수정본?|확정본|최종본|발표용|"
        r"개인정보삭제|예산삭제|공유용|작성요청|fin|final|draft|copy|복사본?|사본)\s*$",
        re.IGNORECASE,
    ),
]

# Student-presentation-style filename head:
#   "한글이름englishname2031_197578_10350428_"
# i.e. {Korean name}{Latin name}{2-3 digit class section}_{6-digit
# upload id}_{8-digit assignment id}_  — drop the whole prefix block.
_STUDENT_HEAD = re.compile(
    r"^[가-힣]{2,4}[A-Za-z]{3,}\d{2,4}_\d{4,7}_\d{6,9}_"
)


def _strip_filename_for_core(stem: str) -> str:
    """Remove prefix/postfix patterns that do not carry project identity.

    What's left is the *content-bearing core* of the filename — the
    part where named entities and topical nouns live.
    """
    s = stem.strip()
    s = _PREFIX_NUM.sub("", s)
    s = _STUDENT_HEAD.sub("", s)
    # Postfix trim is iterative — multiple suffix patterns can stack
    # ("_초안_v1.2", "_(1)_최종").
    changed = True
    while changed:
        changed = False
        for pat in _POSTFIX_PATTERNS:
            new = pat.sub("", s).rstrip()
            if new != s:
                s = new
                changed = True
    return s


# --- schema sequence -------------------------------------------------------

def _schema_sequence(stem: str) -> str:
    """Compress a filename stem into a class-sequence.

    Each character is mapped to {K Korean, A Latin alpha, N digit,
    S separator, P other punctuation}, then runs are collapsed so
    "한글이름englishname2031" → "KAN".
    """
    classes: list[str] = []
    for ch in stem:
        if "가" <= ch <= "힣":
            classes.append("K")
        elif ch.isascii() and ch.isalpha():
            classes.append("A")
        elif ch.isdigit():
            classes.append("N")
        elif ch in "_-. ()[]{}<>·":
            classes.append("S")
        else:
            classes.append("P")
    out = []
    last = None
    for c in classes:
        if c != last:
            out.append(c)
        last = c
    return "".join(out)


def _normalised_lev(a: str, b: str) -> float:
    """Levenshtein-based similarity in [0, 1]."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    n, m = len(a), len(b)
    # iterative DP, O(min(n,m)) memory
    if n < m:
        a, b = b, a
        n, m = m, n
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            cur[j] = min(cur[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        prev = cur
    dist = prev[m]
    return 1.0 - dist / max(n, m)


# --- per-entry signals ----------------------------------------------------

@dataclass
class Signals:
    """Pre-computed signal payload for a FileEntry — cache once per
    entry so we don't pay kiwi cost in tight loops."""
    path: Optional[Path] = None
    raw_stem: str = ""
    core_stem: str = ""
    schema: str = ""
    ext: str = ""                  # ".pdf" / ".mp4" / "" — driver for S6
    name_pn: frozenset[str] = field(default_factory=frozenset)
    body_pn: frozenset[str] = field(default_factory=frozenset)
    modified: Optional[date] = None


def _strip_ext(name: str) -> str:
    return re.sub(r"\.[A-Za-z0-9]{1,5}$", "", name or "")


@lru_cache(maxsize=8192)
def _proper_nouns(text: str) -> frozenset[str]:
    if not text:
        return frozenset()
    try:
        return frozenset(_morph.extract_proper_nouns(text))
    except Exception:
        return frozenset()


def signals_for_entry(entry) -> Signals:
    name = getattr(entry, "name", "") or ""
    raw_stem = _strip_ext(name)
    core_stem = _strip_filename_for_core(raw_stem)
    # Replace separators with spaces before kiwi — kiwi otherwise
    # treats "_과제11_김민지" as one blob.
    core_text = re.sub(r"[_\-.]+", " ", core_stem)
    name_pn = _proper_nouns(core_text)

    body = (getattr(entry, "content_excerpt", "") or "")[:500]
    body_pn = _proper_nouns(body) if body else frozenset()

    try:
        modified = entry.modified.date()
    except Exception:
        modified = None

    ext = (getattr(entry, "ext", None) or "").lower()
    if not ext:
        m = re.search(r"\.([A-Za-z0-9]{1,6})$", name)
        ext = ("." + m.group(1).lower()) if m else ""

    return Signals(
        path=getattr(entry, "path", None),
        raw_stem=raw_stem,
        core_stem=core_stem,
        schema=_schema_sequence(raw_stem),
        ext=ext,
        name_pn=name_pn,
        body_pn=body_pn,
        modified=modified,
    )


@dataclass
class CategorySignals:
    """Signals derived from a Category and (optionally) its current
    member files."""
    cat_pn: frozenset[str] = field(default_factory=frozenset)
    member_signals: list[Signals] = field(default_factory=list)
    parent_paths: frozenset[str] = field(default_factory=frozenset)
    time_range: Optional[tuple[date, date]] = None
    duration: str = ""


def category_signals(category, members: Optional[list[Signals]] = None,
                     time_range: Optional[tuple[date, date]] = None) -> CategorySignals:
    members = members or []
    if isinstance(category, dict):
        name = category.get("name", "") or ""
        desc = category.get("description", "") or ""
        kws = category.get("keywords") or []
        duration = category.get("duration", "") or ""
    else:
        name = getattr(category, "name", "") or ""
        desc = getattr(category, "description", "") or ""
        kws = getattr(category, "keywords", None) or []
        duration = getattr(category, "duration", "") or ""
    parts = [name, desc]
    if isinstance(kws, list):
        parts.extend(str(k) for k in kws if k)
    cat_pn = _proper_nouns(" ".join(parts))
    parent_paths: set[str] = set()
    for m in members:
        if m.path is not None:
            try:
                parent_paths.add(str(Path(m.path).parent))
            except Exception:
                pass
    return CategorySignals(
        cat_pn=cat_pn,
        member_signals=members,
        parent_paths=frozenset(parent_paths),
        time_range=time_range,
        duration=duration,
    )


# --- per-axis sub-scores ---------------------------------------------------

def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a or not b:
        return 0.0
    inter = a & b
    if not inter:
        return 0.0
    union = a | b
    return len(inter) / len(union)


def s1_filename_core(file: Signals, target_pn: frozenset[str]) -> float:
    return _jaccard(file.name_pn, target_pn)


def s2_schema(file: Signals, target_schemas: list[str]) -> float:
    if not target_schemas:
        return 0.0
    return max(_normalised_lev(file.schema, t) for t in target_schemas)


def s3_time(file: Signals, time_range: Optional[tuple[date, date]],
            duration: str = "") -> float:
    if not file.modified or not time_range:
        return 0.0
    start, end = time_range
    # Inside the window → distance 0.
    if start <= file.modified <= end:
        return 1.0
    delta = (file.modified - end).days if file.modified > end else (start - file.modified).days
    decay_days = 90 if duration in ("burst", "short") else 365
    import math
    return math.exp(-delta / decay_days) if delta > 0 else 1.0


def s4_path(file: Signals, parent_paths: frozenset[str]) -> float:
    if not parent_paths or file.path is None:
        return 0.0
    try:
        my_parent = str(Path(file.path).parent)
        my_grand = str(Path(file.path).parent.parent)
    except Exception:
        return 0.0
    if my_parent in parent_paths:
        return 1.0
    for p in parent_paths:
        try:
            grand = str(Path(p).parent)
        except Exception:
            continue
        if my_grand == grand or my_parent == grand or my_grand == p:
            return 0.5
    return 0.0


def s5_body(file: Signals, target_pn: frozenset[str]) -> float:
    return _jaccard(file.body_pn, target_pn)


# Generic clerical extensions that nearly every corpus has — matching
# on these alone (every PDF together) is the failure mode the system
# prompt warns against ("확장자 기반 분류 절대 금지").  We still let
# extension contribute, but cap the score against these so the LLM is
# *only nudged*, not steered, by ext alone.
_GENERIC_EXTS = {
    ".pdf", ".docx", ".doc", ".txt", ".md", ".html", ".htm",
    ".png", ".jpg", ".jpeg", ".gif",
}


def s6_extension(file: Signals, target_exts: frozenset[str]) -> float:
    """Extension match — strong for distinctive extensions (.mp4 / .zip /
    .hwp / .xlsx batches), weak for "everything is a .pdf" cases.

    Returns 1.0 when ``file.ext`` matches any member's ext AND the ext
    is reasonably specific; 0.5 for matches on generic extensions
    (.pdf / .png / etc — common across the whole corpus); 0 for no
    match.
    """
    if not file.ext or not target_exts:
        return 0.0
    if file.ext not in target_exts:
        return 0.0
    return 0.5 if file.ext in _GENERIC_EXTS else 1.0


def s7_literal_prefix(file: Signals, target_stems: list[str]) -> float:
    """Longest literal common prefix between ``file.raw_stem`` and any
    member stem, normalised by the file's own stem length.  Catches
    same-prefix batches that share NO proper nouns (e.g.
    ``강의평가_2025-01-08`` / ``강의평가_사회과목`` collapse on the
    literal prefix ``강의평가_`` even when kiwi drops every 2-char
    NNG to nothing).
    """
    src = file.raw_stem or ""
    if not src or not target_stems:
        return 0.0
    best = 0
    for t in target_stems:
        if not t:
            continue
        n = 0
        for ch_a, ch_b in zip(src, t):
            if ch_a == ch_b:
                n += 1
            else:
                break
        if n > best:
            best = n
    if best < 2:
        return 0.0
    # Normalise by the SHORTER stem (so "강의평가" ⊂ "강의평가_사회"
    # scores as 1.0, not 0.4 because of the longer side).
    shortest = min(
        len(src),
        min((len(t) for t in target_stems if t), default=len(src)),
    )
    return min(1.0, best / max(1, shortest))


# --- composite ------------------------------------------------------------

@dataclass
class Weights:
    """Multi-axis compatibility weights.

    Tuned for the user-reported failure where files with similar
    title patterns or matching extensions kept getting split into
    sibling singleton folders.  S7 (literal prefix overlap) is the
    main pull for batch-file pattern matching ("강의평가_*" series).
    S6 (extension) and S2 (abstract schema) provide weaker but
    additive signals.  S4 (parent path) drops to a minor role
    because reclassify mode zeroes it anyway.
    """
    s1: float = 0.20       # filename-core proper-noun Jaccard
    s2: float = 0.15       # filename schema/pattern similarity (abstract)
    s3: float = 0.05       # modified-time proximity
    s4: float = 0.10       # scan-time path co-residence
    s5: float = 0.10       # body-head proper-noun Jaccard
    s6: float = 0.15       # extension match
    s7: float = 0.25       # ★ literal name-prefix overlap (new axis)

    def reclassify(self) -> "Weights":
        """S4 is unreliable when the user is escaping the existing
        layout — disable it and renormalise the rest.
        """
        keep = self.s1 + self.s2 + self.s3 + self.s5 + self.s6 + self.s7
        if keep <= 0:
            return Weights(
                s1=0.22, s2=0.17, s3=0.05, s4=0.0,
                s5=0.10, s6=0.18, s7=0.28,
            )
        return Weights(
            s1=self.s1 / keep,
            s2=self.s2 / keep,
            s3=self.s3 / keep,
            s4=0.0,
            s5=self.s5 / keep,
            s6=self.s6 / keep,
            s7=self.s7 / keep,
        )


_DEFAULT_WEIGHTS = Weights()


def compatibility(
    file: Signals,
    cat: CategorySignals,
    *,
    reclassify_mode: bool = False,
    weights: Optional[Weights] = None,
) -> float:
    """Composite score in [0, 1] — see module docstring."""
    w = (weights or _DEFAULT_WEIGHTS)
    if reclassify_mode:
        w = w.reclassify()

    # S1: file core noun Jaccard against category PN ∪ all members'
    # core nouns.  Falling back to category-name-only is fine when
    # the category has no members yet (first pass).
    target_name_pn: frozenset[str] = cat.cat_pn
    if cat.member_signals:
        target_name_pn = cat.cat_pn.union(*(m.name_pn for m in cat.member_signals))
    s1 = s1_filename_core(file, target_name_pn)

    # S2: max schema sim across members + the category-name schema
    # (last is usually weak but keeps the path defined when 0 members).
    schemas = [m.schema for m in cat.member_signals]
    s2 = s2_schema(file, schemas)

    s3 = s3_time(file, cat.time_range, cat.duration)
    s4 = 0.0 if reclassify_mode else s4_path(file, cat.parent_paths)

    target_body_pn: frozenset[str] = cat.cat_pn
    if cat.member_signals:
        target_body_pn = cat.cat_pn.union(*(m.body_pn for m in cat.member_signals))
    s5 = s5_body(file, target_body_pn)

    # S6: file's extension overlap with the category members' exts.
    target_exts = frozenset(m.ext for m in cat.member_signals if m.ext)
    s6 = s6_extension(file, target_exts)

    # S7: literal name-prefix overlap.
    target_stems = [m.raw_stem for m in cat.member_signals if m.raw_stem]
    s7 = s7_literal_prefix(file, target_stems)

    return min(
        1.0,
        max(0.0,
            w.s1 * s1 + w.s2 * s2 + w.s3 * s3
            + w.s4 * s4 + w.s5 * s5 + w.s6 * s6 + w.s7 * s7,
            ),
    )


def pair_compat(a: Signals, b: Signals, *, reclassify_mode: bool = False,
                weights: Optional[Weights] = None) -> float:
    """File-vs-file compatibility — used for clustering / outlier
    expulsion.  Time range is treated as a single point with span 0."""
    w = (weights or _DEFAULT_WEIGHTS)
    if reclassify_mode:
        w = w.reclassify()
    s1 = _jaccard(a.name_pn, b.name_pn)
    s2 = _normalised_lev(a.schema, b.schema)
    if a.modified and b.modified:
        delta = abs((a.modified - b.modified).days)
        import math
        s3 = math.exp(-delta / 365)
    else:
        s3 = 0.0
    if reclassify_mode or a.path is None or b.path is None:
        s4 = 0.0
    else:
        try:
            ap = str(Path(a.path).parent)
            bp = str(Path(b.path).parent)
            if ap == bp:
                s4 = 1.0
            elif str(Path(ap).parent) == str(Path(bp).parent):
                s4 = 0.5
            else:
                s4 = 0.0
        except Exception:
            s4 = 0.0
    s5 = _jaccard(a.body_pn, b.body_pn)
    # S6: extension match — pair version of s6_extension.
    if a.ext and b.ext and a.ext == b.ext:
        s6 = 0.5 if a.ext in _GENERIC_EXTS else 1.0
    else:
        s6 = 0.0
    # S7: literal prefix overlap.
    s7 = s7_literal_prefix(a, [b.raw_stem]) if b.raw_stem else 0.0
    return min(
        1.0,
        max(0.0,
            w.s1 * s1 + w.s2 * s2 + w.s3 * s3
            + w.s4 * s4 + w.s5 * s5 + w.s6 * s6 + w.s7 * s7,
            ),
    )


# --- thresholds (callers can override) ------------------------------------
# Tuned against the user-reported corpus (학생 발표자료 vs 사업 폴더):
# at 0.15 the rescue rejects every PN-less file that just shares a
# time-window match (S3 alone scores ≈ 0.095) while still letting a
# real 의약품 / 행안부 / 한양대 file through (S1 contribution ≈ 0.135
# brings the total to ≈ 0.23).  See test_compatibility_*.
THRESHOLD_GUESS_BY_TIME = 0.12   # ↓ from 0.15 after S1 weight redistribution
THRESHOLD_FILENAME_VETO = 0.20
THRESHOLD_OUTLIER = 0.20
THRESHOLD_CLUSTER_MERGE = 0.35
