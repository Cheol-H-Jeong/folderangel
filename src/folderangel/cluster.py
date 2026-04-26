"""Filename signature + clustering for the large-corpus planner.

Goal: collapse a 5,000-file corpus down to a few hundred *clusters*
without any LLM call, so that downstream we only need to ask the LLM
about a representative sample of each cluster instead of paying per
file.

The signature strips the kinds of variance that distinguish *members
of the same logical document family* (versions, dates, sequence
numbers, "copy of" prefixes, file extension, surrounding decoration)
while keeping the project / customer / system core that the LLM
actually uses to assign a folder.

Pure-Python, deterministic, OS-agnostic, no LLM tokens spent.
"""
from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

from .models import FileEntry


# ----- normalisation -------------------------------------------------------

# Use lookarounds (not \b) because Python's \b doesn't fire between
# underscore and a word char — and these patterns are nearly always
# wrapped in underscores in real-world filenames.
_BOUND = r"(?<![A-Za-z0-9])"
_BOUND_END = r"(?![A-Za-z0-9])"
_VERSION_RE = re.compile(
    rf"{_BOUND}(?:v|ver|version|rev|revision|draft|fin|final|"
    rf"r|R|최종|확정|초안|수정|\d?차)\s*[._-]?\s*\d+(?:[._.\-]\d+)*{_BOUND_END}",
    re.IGNORECASE,
)
_DATE_RE = re.compile(
    rf"(?:"
    rf"{_BOUND}\d{{4}}[-_/.]?\d{{2}}[-_/.]?\d{{2}}{_BOUND_END}"   # 2024-03-21 / 20240321
    rf"|{_BOUND}\d{{2}}\d{{2}}\d{{2}}{_BOUND_END}"                # 240321
    rf"|{_BOUND}\d{{4}}[-_/.]?\d{{2}}{_BOUND_END}"                # 2024-03 / 202403
    rf"|{_BOUND}\d{{2}}[-_/.]?\d{{2}}[-_/.]?\d{{2}}{_BOUND_END}"  # 24-03-21
    rf")"
)
_SEQ_RE = re.compile(r"\((\s*\d+\s*)\)|_\d{1,3}$|copy(?:\s*of)?", re.IGNORECASE)
_DECORATION_RE = re.compile(r"^(?:★|※|◎|◆|■|●|○|▶|▷)+\s*")
# Words we know carry no project-identity signal — drop them so two
# variants like "...최종본" and "...작성요청 (1)" still collapse.
_NOISE_TOKENS = {
    "복사본", "복사", "사본", "copy", "of", "수정본", "변경본",
    "최종본", "최종판", "최종", "확정본", "발표용", "작성요청",
    "임시", "원본", "공유용", "draft", "final", "fin",
}
_TOKEN_SPLIT_RE = re.compile(r"[\s_\-\.,()\[\]{}<>]+")
# Number of leading tokens that form the canonical project signature.
# Real-world business filenames put the project / customer name first,
# so the first 2 meaningful tokens are nearly always the right key
# (e.g. "한국지역정보개발원 제안발표 …" or "AVOCA 특허 …").  Three
# tokens overfit and split a single project across many clusters.
_SIG_PREFIX_LEN = 2


def _strip(name: str) -> str:
    """Remove version / date / seq / decoration noise from a filename."""
    s = unicodedata.normalize("NFC", name)
    s = _DECORATION_RE.sub("", s)
    s = re.sub(r"\.[A-Za-z0-9]{1,5}$", "", s)  # extension
    s = _DATE_RE.sub(" ", s)
    s = _VERSION_RE.sub(" ", s)
    s = _SEQ_RE.sub(" ", s)
    return s


def _tokenise(stripped: str) -> list[str]:
    raw = _TOKEN_SPLIT_RE.split(stripped)
    out: list[str] = []
    for t in raw:
        t = t.strip().casefold()
        if not t:
            continue
        # Drop pure-numeric tokens (sequence numbers we missed) and
        # known low-signal noise words.
        if t.isdigit():
            continue
        if t in _NOISE_TOKENS:
            continue
        if len(t) < 2:
            continue
        out.append(t)
    return out


def signature(name: str) -> str:
    """Stable hashable key that collapses members of the same family.

    Two filenames that only differ by version / date / sequence /
    decoration / writer-suffix produce the same signature.  We use
    the first ``_SIG_PREFIX_LEN`` meaningful tokens — real-world
    business filenames lead with the project / customer name and
    follow it with version / writer / period detail that we want to
    discard for clustering.
    """
    tokens = _tokenise(_strip(name))
    if not tokens:
        return ""
    return " ".join(tokens[:_SIG_PREFIX_LEN])


# ----- clustering ----------------------------------------------------------

@dataclass
class Cluster:
    signature: str
    members: list[FileEntry] = field(default_factory=list)

    @property
    def size(self) -> int:
        return len(self.members)

    @property
    def time_range(self) -> Optional[tuple[datetime, datetime]]:
        if not self.members:
            return None
        ts = sorted(m.modified for m in self.members)
        return ts[0], ts[-1]

    def representatives(self, k: int = 2) -> list[FileEntry]:
        """Pick ``k`` member files most useful for the LLM:
        the latest by mtime + the longest excerpt available."""
        if not self.members:
            return []
        if self.size <= k:
            return list(self.members)
        # Always include the newest (most likely the latest version).
        by_mtime = sorted(self.members, key=lambda m: m.modified, reverse=True)
        chosen: list[FileEntry] = [by_mtime[0]]
        # Then the one with the richest excerpt (most signal for the LLM).
        rest = [m for m in self.members if m is not chosen[0]]
        rest.sort(key=lambda m: len(m.content_excerpt or ""), reverse=True)
        for m in rest[: k - 1]:
            chosen.append(m)
        return chosen


def cluster_files(
    entries: Iterable[FileEntry], min_cluster_size: int = 3
) -> tuple[list[Cluster], list[FileEntry]]:
    """Group entries by their filename signature.

    Returns ``(clusters, long_tail)``:
      * clusters    — every signature with ≥ ``min_cluster_size``
                      members, ordered largest-first.
      * long_tail   — all other entries (the planner runs them through
                      the regular per-file path so a singleton's
                      classification is never compromised).
    """
    buckets: dict[str, list[FileEntry]] = defaultdict(list)
    for e in entries:
        sig = signature(e.name) or "_singleton_"
        buckets[sig].append(e)

    clusters: list[Cluster] = []
    long_tail: list[FileEntry] = []
    for sig, members in buckets.items():
        if len(members) >= min_cluster_size:
            clusters.append(Cluster(signature=sig, members=members))
        else:
            long_tail.extend(members)
    clusters.sort(key=lambda c: c.size, reverse=True)
    return clusters, long_tail


def collapse_ratio(total_files: int, clusters: list[Cluster], long_tail_n: int) -> float:
    """Returns the (representatives + long-tail) / total ratio.

    Used by the planner to decide whether the hierarchical path is
    worthwhile (collapse < 0.4 means we're saving real money).
    """
    if total_files <= 0:
        return 1.0
    reps = sum(min(c.size, 2) for c in clusters)
    return (reps + long_tail_n) / total_files
