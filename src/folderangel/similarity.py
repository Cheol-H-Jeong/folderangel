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

    return Signals(
        path=getattr(entry, "path", None),
        raw_stem=raw_stem,
        core_stem=core_stem,
        schema=_schema_sequence(raw_stem),
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


# --- composite ------------------------------------------------------------

@dataclass
class Weights:
    s1: float = 0.27
    s2: float = 0.27
    s3: float = 0.095
    s4: float = 0.27
    s5: float = 0.095

    def reclassify(self) -> "Weights":
        """S4 is unreliable when the user is escaping the existing
        layout — disable it and renormalise.
        """
        keep = self.s1 + self.s2 + self.s3 + self.s5
        if keep <= 0:
            return Weights(s1=0.4, s2=0.4, s3=0.1, s4=0.0, s5=0.1)
        return Weights(
            s1=self.s1 / keep,
            s2=self.s2 / keep,
            s3=self.s3 / keep,
            s4=0.0,
            s5=self.s5 / keep,
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

    return min(
        1.0,
        max(0.0,
            w.s1 * s1 + w.s2 * s2 + w.s3 * s3 + w.s4 * s4 + w.s5 * s5,
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
    return min(
        1.0,
        max(0.0,
            w.s1 * s1 + w.s2 * s2 + w.s3 * s3 + w.s4 * s4 + w.s5 * s5,
            ),
    )


# --- thresholds (callers can override) ------------------------------------
# Tuned against the user-reported corpus (학생 발표자료 vs 사업 폴더):
# at 0.15 the rescue rejects every PN-less file that just shares a
# time-window match (S3 alone scores ≈ 0.095) while still letting a
# real 의약품 / 행안부 / 한양대 file through (S1 contribution ≈ 0.135
# brings the total to ≈ 0.23).  See test_compatibility_*.
THRESHOLD_GUESS_BY_TIME = 0.15
THRESHOLD_FILENAME_VETO = 0.20
THRESHOLD_OUTLIER = 0.20
THRESHOLD_CLUSTER_MERGE = 0.35
