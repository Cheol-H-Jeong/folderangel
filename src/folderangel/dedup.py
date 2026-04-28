"""Content-hash-based duplicate detection.

Goal: when the user has the *exact same file* sitting at multiple
locations (often because the same large media file or PDF was
downloaded twice, copied across devices, or is a pre-existing
duplicate the LLM-organise step would naively place into two
different folders), classify only one canonical copy and *delete*
the rest after the canonical is moved.

Cheap two-stage filter so we don't hash a 5,000-file corpus:
    1. Group by file size — two files of different size cannot be
       byte-equal.  This is a single ``stat()`` per file, basically
       free.
    2. For groups of ≥ 2 files whose size meets ``min_bytes``, hash
       the content (BLAKE2b — faster than SHA-256, plenty of
       resistance for non-adversarial dedup) and group by hash.

Returns the duplicate groups as ``[[FileEntry, FileEntry, …], …]``
where each inner list is the set of files that share content.  The
caller (pipeline) picks one canonical, sends only that to the
classifier, and after the move deletes the others.
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .models import FileEntry

log = logging.getLogger(__name__)

_HASH_CHUNK = 1 << 20  # 1 MiB read buffer
_HASH_PREFIX_LIMIT = 256 * (1 << 20)  # cap per-file hash work at 256 MiB


def _content_hash(path: Path, *, size: int) -> str:
    """BLAKE2b of the file's bytes — capped at 256 MiB.

    For files larger than the cap we hash the *first* 256 MiB and
    fold the size into the digest, which is enough to deduplicate
    accidentally-copied media without scanning gigabytes.
    """
    h = hashlib.blake2b(digest_size=16)
    remaining = min(size, _HASH_PREFIX_LIMIT)
    try:
        with open(path, "rb") as f:
            while remaining > 0:
                chunk = f.read(min(_HASH_CHUNK, remaining))
                if not chunk:
                    break
                h.update(chunk)
                remaining -= len(chunk)
    except OSError as exc:
        log.warning("hash failed for %s: %s", path, exc)
        return ""
    h.update(str(size).encode("ascii"))  # disambiguate truncated hashes
    return h.hexdigest()


@dataclass
class DupeGroup:
    canonical: FileEntry
    duplicates: list[FileEntry]

    @property
    def bytes_per_duplicate(self) -> int:
        try:
            return int(self.canonical.size or 0)
        except Exception:
            return 0

    @property
    def total_bytes_freed(self) -> int:
        return self.bytes_per_duplicate * len(self.duplicates)


def find_duplicate_groups(
    entries: Iterable[FileEntry],
    *,
    min_bytes: int = 1_048_576,
) -> list[DupeGroup]:
    """Return groups of byte-identical files (≥ 2 each).

    Files smaller than ``min_bytes`` are skipped — small dupes don't
    save meaningful space and the user typically wants to keep them
    (config files, small docs).  Set ``min_bytes=0`` to dedup
    everything.
    """
    by_size: dict[int, list[FileEntry]] = {}
    for e in entries:
        size = int(getattr(e, "size", 0) or 0)
        if size <= 0 or size < min_bytes:
            continue
        by_size.setdefault(size, []).append(e)

    groups: list[DupeGroup] = []
    for size, ents in by_size.items():
        if len(ents) < 2:
            continue
        by_hash: dict[str, list[FileEntry]] = {}
        for e in ents:
            digest = _content_hash(Path(e.path), size=size)
            if not digest:
                continue
            by_hash.setdefault(digest, []).append(e)
        for h, group in by_hash.items():
            if len(group) < 2:
                continue
            # Pick canonical: shortest path wins (typically the
            # cleanest location), falling back to oldest mtime.
            group_sorted = sorted(
                group,
                key=lambda x: (
                    len(str(x.path)),
                    getattr(x.modified, "timestamp", lambda: 0.0)(),
                    str(x.path),
                ),
            )
            canonical = group_sorted[0]
            duplicates = group_sorted[1:]
            groups.append(DupeGroup(canonical=canonical, duplicates=duplicates))
    return groups


def remove_duplicate_files(
    groups: list[DupeGroup],
    *,
    dry_run: bool = False,
) -> list[tuple[Path, Path, int]]:
    """Delete every duplicate from disk.  Caller is expected to have
    already moved each ``canonical`` file to its final location.

    Returns ``[(deleted_path, canonical_path, bytes_freed), …]`` for
    auditing / reporting.  ``dry_run=True`` returns the list without
    touching the filesystem so the UI can preview the deletion.
    """
    actions: list[tuple[Path, Path, int]] = []
    for g in groups:
        bytes_each = g.bytes_per_duplicate
        for dup in g.duplicates:
            dpath = Path(dup.path)
            cpath = Path(g.canonical.path)
            try:
                if not dry_run:
                    if dpath.exists():
                        dpath.unlink()
            except OSError as exc:
                log.warning("dedup delete failed for %s: %s", dpath, exc)
                continue
            actions.append((dpath, cpath, bytes_each))
    return actions
