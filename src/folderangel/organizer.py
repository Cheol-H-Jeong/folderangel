"""Apply a :class:`Plan` to the filesystem.

The organiser is the only module allowed to mutate files.  It is deliberately
tolerant: one failing file must not abort the whole run.  The outcome is
captured in an :class:`OperationResult` which downstream (index, reporter)
consume.
"""
from __future__ import annotations

import logging
import os
import re
import shutil
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

from .config import Config
from .models import (
    Assignment,
    Category,
    MovedFile,
    OperationResult,
    Plan,
    SkippedFile,
)
from .shortcuts import create_shortcut

log = logging.getLogger(__name__)

ProgressCB = Callable[[str, float], None]

_INVALID_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_TRAILING = re.compile(r"[. ]+$")


def sanitize_folder_name(name: str, fallback: str = "folder") -> str:
    """Return a filesystem-safe version of *name* usable on both Linux and Windows."""
    if not name:
        return fallback
    cleaned = unicodedata.normalize("NFC", name.strip())
    cleaned = _INVALID_CHARS.sub("_", cleaned)
    cleaned = _TRAILING.sub("", cleaned)
    # Windows reserved device names
    reserved = {
        "CON", "PRN", "AUX", "NUL",
        *(f"COM{i}" for i in range(1, 10)),
        *(f"LPT{i}" for i in range(1, 10)),
    }
    if cleaned.upper() in reserved:
        cleaned = f"_{cleaned}"
    if not cleaned:
        return fallback
    return cleaned[:120]


_GROUP_PREFIX_RE = re.compile(r"^\s*(\d)\.\s+")
_TIME_SUFFIX_RE = re.compile(r"\s*\([^()]+\)\s*$")


def compose_folder_name(cat: Category) -> str:
    """Build the on-disk folder name from a :class:`Category`.

    Convention (always applied): ``"{group}. {name} ({time_label})"`` —
    every category gets a 1..9 group prefix, and the time suffix appears
    only when the LLM provided one.  Examples::

        Category("a", "AVOCA 시스템", group=2, time_label="2024-Q3")
            → "2. AVOCA 시스템 (2024-Q3)"
        Category("a", "프로젝트 외 자료", group=9, time_label="")
            → "9. 프로젝트 외 자료"
    """
    g = max(1, min(9, int(cat.group or 9)))
    pieces = [f"{g}.", cat.name or cat.id]
    if cat.time_label:
        pieces.append(f"({cat.time_label})")
    return sanitize_folder_name(" ".join(pieces))


def _normalize_for_match(folder_name: str) -> str:
    """Reduce a folder name to its 'core' for fuzzy comparison.

    Drops any leading ``"N. "`` group prefix and trailing ``" (...)"`` time
    suffix, then casefolds + collapses whitespace so that
    ``"1. AVOCA 시스템 (2024-Q3)"`` and ``"AVOCA 시스템"`` collide.
    """
    s = folder_name.strip()
    s = _GROUP_PREFIX_RE.sub("", s)
    s = _TIME_SUFFIX_RE.sub("", s).strip()
    s = re.sub(r"\s+", " ", s)
    return s.casefold()


_TOKEN_RE = re.compile(r"[0-9A-Za-z가-힣]+")


def _tokens(core: str) -> set[str]:
    return {t for t in _TOKEN_RE.findall(core) if len(t) >= 2}


def _fuzzy_match_score(existing_core: str, new_core: str) -> float:
    """Heuristic score in 0..1 for two normalized folder cores.

    Used to reuse a pre-existing folder when the LLM produced a slightly
    different but clearly-related new name (e.g. existing "AVOCA" vs new
    "AVOCA 특허 및 분석 모듈").  We require that the *shorter* side's
    distinctive tokens are mostly contained in the longer side.
    """
    if not existing_core or not new_core:
        return 0.0
    if existing_core == new_core:
        return 1.0
    a = _tokens(existing_core)
    b = _tokens(new_core)
    if not a or not b:
        return 0.0
    smaller, larger = (a, b) if len(a) <= len(b) else (b, a)
    inter = smaller & larger
    if not inter:
        return 0.0
    coverage = len(inter) / len(smaller)
    # Substring shortcut: pre-existing core fully appears as a phrase.
    if existing_core in new_core or new_core in existing_core:
        coverage = max(coverage, 0.85)
    return coverage


# ----- Time-label helpers ------------------------------------------------

def _parse_time_label(label: str) -> Optional[datetime]:
    """Heuristic: turn '2024', '2024-Q1', '2024-03' into a representative dt.

    Returns None when the label is empty or unparseable.  We pick the middle
    of the period so it sorts naturally in file-manager date columns.
    """
    if not label:
        return None
    s = label.strip()
    m = re.fullmatch(r"(\d{4})-Q([1-4])", s)
    if m:
        year = int(m.group(1))
        q = int(m.group(2))
        month = (q - 1) * 3 + 2  # mid-quarter month
        return datetime(year, month, 15)
    m = re.fullmatch(r"(\d{4})-(\d{1,2})", s)
    if m:
        return datetime(int(m.group(1)), max(1, min(12, int(m.group(2)))), 15)
    m = re.fullmatch(r"(\d{4})", s)
    if m:
        return datetime(int(m.group(1)), 6, 15)
    return None


def _set_dir_mtime(path: Path, dt: datetime) -> None:
    try:
        ts = dt.timestamp()
        os.utime(path, (ts, ts))
    except (OSError, OverflowError, ValueError) as exc:
        log.debug("set mtime failed for %s: %s", path, exc)


def _walk_dirs(root: Path):
    for entry in os.scandir(root):
        if entry.is_dir(follow_symlinks=False):
            yield entry.path
            yield from _walk_dirs(Path(entry.path))


def _unique_path(target: Path) -> Path:
    if not target.exists():
        return target
    stem = target.stem
    suffix = target.suffix
    parent = target.parent
    idx = 2
    while True:
        candidate = parent / f"{stem} ({idx}){suffix}"
        if not candidate.exists():
            return candidate
        idx += 1


class Organizer:
    def __init__(self, config: Config):
        self.config = config

    def execute(
        self,
        target_root: Path,
        plan: Plan,
        dry_run: bool = False,
        progress: Optional[ProgressCB] = None,
    ) -> OperationResult:
        target_root = Path(target_root).resolve()
        started_at = datetime.now().astimezone()

        # Pre-compute safe folder paths for each category id, but *defer*
        # directory creation until we actually place a file (so empty
        # categories don't clutter the target root).  Sort by group/name so
        # the on-disk order reflects the LLM's relevance grouping.
        ordered = sorted(
            plan.categories,
            key=lambda c: (c.group or 99, c.time_label or "~", c.name or c.id),
        )

        # Map of normalized-name → existing folder Path.  We use this to
        # reuse a pre-existing folder whose core name matches a planned
        # category instead of creating a sibling with a slightly different
        # group prefix or time suffix.  Pre-existing folders are also
        # *renamed* to the canonical "N. name (period)" pattern so the
        # whole target root ends up with consistent folder naming.
        existing_dirs = self._list_existing_dirs(target_root)

        dir_for: dict[str, Path] = {}
        used_paths: set[Path] = set()
        for cat in ordered:
            canonical = compose_folder_name(cat)
            canonical_path = target_root / canonical
            cat_core = _normalize_for_match(canonical)

            chosen: Optional[Path] = None
            best_score = 0.0
            for d in existing_dirs:
                if d in used_paths:
                    continue
                score = _fuzzy_match_score(_normalize_for_match(d.name), cat_core)
                if score >= 0.85 and score > best_score:
                    chosen = d
                    best_score = score
            if chosen is None:
                # No existing folder with the same core name — pick the
                # canonical path, deduping if necessary.
                target_path = canonical_path
                counter = 2
                while target_path in used_paths or (
                    target_path.exists() and target_path != canonical_path
                ):
                    target_path = target_root / f"{canonical} ({counter})"
                    counter += 1
                chosen = target_path
            else:
                # Rename the existing folder to the canonical form so the
                # whole target root follows one naming convention.
                if chosen.name != canonical and not canonical_path.exists():
                    if not dry_run:
                        try:
                            chosen.rename(canonical_path)
                            chosen = canonical_path
                        except OSError as exc:
                            log.warning(
                                "rename %s → %s failed: %s",
                                chosen, canonical_path, exc,
                            )
                    else:
                        chosen = canonical_path

            used_paths.add(chosen)
            dir_for[cat.id] = chosen

        created_dirs: set[Path] = set()

        def ensure_dir(cid: str) -> Path:
            d = dir_for[cid]
            if not dry_run and d not in created_dirs:
                d.mkdir(parents=True, exist_ok=True)
                created_dirs.add(d)
            return d

        moved: list[MovedFile] = []
        skipped: list[SkippedFile] = []
        used_category_ids: set[str] = set()

        total = max(1, len(plan.assignments))
        for idx, assign in enumerate(plan.assignments, 1):
            cat_for_msg = next((c for c in plan.categories if c.id == assign.primary_category_id), None)
            cat_label = cat_for_msg.name if cat_for_msg else assign.primary_category_id
            if progress:
                progress(
                    f"move [{idx}/{total}] {assign.file_path.name} → {cat_label}",
                    idx / total,
                )
            try:
                moved_entry = self._apply_one(
                    assign, plan, dir_for, target_root, dry_run, ensure_dir
                )
                if moved_entry is not None:
                    moved.append(moved_entry)
                    used_category_ids.add(moved_entry.category_id)
                    for sp in moved_entry.shortcuts:
                        if progress:
                            progress(f"  ↳ 바로가기: {sp.name}", idx / total)
                        # Track categories that received shortcuts too.
                        for cid, cdir in dir_for.items():
                            try:
                                if sp.parent.resolve() == cdir.resolve():
                                    used_category_ids.add(cid)
                                    break
                            except FileNotFoundError:
                                continue
            except Exception as exc:
                log.warning("organize failed for %s: %s", assign.file_path, exc)
                if progress:
                    progress(f"  ⚠ 스킵: {assign.file_path.name} ({exc})", idx / total)
                skipped.append(SkippedFile(path=assign.file_path, reason=str(exc)))

        # Adopt files that were already sitting inside a (now reused) category
        # folder.  These don't need to be moved, but we record them so the
        # report and stats reflect the *final* contents of each category.
        moved_paths = {mf.new_path.resolve() for mf in moved if mf.new_path.exists()}
        for cid, cdir in dir_for.items():
            if not cdir.exists() or not cdir.is_dir():
                continue
            for entry in os.scandir(cdir):
                if not entry.is_file(follow_symlinks=False):
                    continue
                p = Path(entry.path)
                try:
                    rp = p.resolve()
                except OSError:
                    rp = p
                if rp in moved_paths:
                    continue
                moved.append(
                    MovedFile(
                        original_path=p,
                        new_path=p,
                        category_id=cid,
                        reason="기존 폴더 잔존 파일 흡수",
                        score=1.0,
                    )
                )
                used_category_ids.add(cid)
                moved_paths.add(rp)

        # Report only the categories that ended up hosting at least one file.
        surviving_categories = [c for c in plan.categories if c.id in used_category_ids]

        # Stamp folder mtimes from the LLM-provided time_label so file
        # managers can sort folders chronologically.
        if not dry_run:
            if progress:
                progress("organize: 폴더 시기(mtime) 적용", 0.97)
            for cat in surviving_categories:
                dt = _parse_time_label(cat.time_label)
                d = dir_for.get(cat.id)
                if dt is not None and d is not None and d.is_dir():
                    _set_dir_mtime(d, dt)

            # Empty-folder cleanup: any subdirectory under the target root
            # that is now empty (whether we created it or it pre-existed)
            # gets removed so the result is tidy.
            if progress:
                progress("organize: 빈 폴더 정리", 0.99)
            self._sweep_empty_dirs(target_root)

        finished_at = datetime.now().astimezone()
        return OperationResult(
            target_root=target_root,
            started_at=started_at,
            finished_at=finished_at,
            dry_run=dry_run,
            categories=surviving_categories,
            moved=moved,
            skipped=skipped,
            total_scanned=len(plan.assignments),
        )

    # -----------------------------------------------------------------
    def _list_existing_dirs(self, root: Path) -> list[Path]:
        out: list[Path] = []
        try:
            for entry in os.scandir(root):
                if entry.is_dir(follow_symlinks=False):
                    out.append(Path(entry.path))
        except FileNotFoundError:
            return []
        return out

    # -----------------------------------------------------------------
    def _sweep_empty_dirs(self, root: Path) -> None:
        """Remove empty subdirectories under *root*, depth-first.

        Only directories are touched; the root itself is preserved.  We sort
        deepest-first so that nested empty trees collapse correctly.
        """
        try:
            all_dirs = [Path(d) for d in _walk_dirs(root)]
        except FileNotFoundError:
            return
        for d in sorted(all_dirs, key=lambda p: len(p.parts), reverse=True):
            if d == root:
                continue
            try:
                if d.is_dir() and not any(d.iterdir()):
                    d.rmdir()
            except OSError as exc:
                log.debug("rmdir skipped %s: %s", d, exc)

    # -----------------------------------------------------------------
    def _apply_one(
        self,
        assign: Assignment,
        plan: Plan,
        dir_for: dict[str, Path],
        target_root: Path,
        dry_run: bool,
        ensure_dir,
    ) -> Optional[MovedFile]:
        src = assign.file_path
        if not src.exists():
            raise FileNotFoundError(src)

        cat_id = assign.primary_category_id
        if cat_id not in dir_for:
            # Fall back to misc if it exists, else first available category.
            if "misc" in dir_for:
                cat_id = "misc"
            elif dir_for:
                cat_id = next(iter(dir_for))
            else:
                raise KeyError(f"unknown category {assign.primary_category_id}")
        primary_dir = ensure_dir(cat_id)

        # Skip move if the file is already sitting in the destination folder.
        target_path = primary_dir / src.name
        if src.resolve() == target_path.resolve():
            return MovedFile(
                original_path=src,
                new_path=src,
                category_id=cat_id,
                reason=assign.reason,
                score=assign.primary_score,
            )

        new_path = _unique_path(target_path)
        if not dry_run:
            shutil.move(str(src), str(new_path))

        shortcut_paths: list[Path] = []
        # Only create shortcuts for secondary categories whose score is close enough.
        primary_score = assign.primary_score
        for sec in assign.secondary:
            if primary_score - sec.score > self.config.ambiguity_threshold:
                continue
            if sec.category_id == cat_id:
                continue
            if sec.category_id not in dir_for:
                continue
            sec_dir = ensure_dir(sec.category_id)
            if dry_run:
                shortcut_paths.append(sec_dir / new_path.name)
                continue
            try:
                sp = create_shortcut(new_path, sec_dir)
                shortcut_paths.append(sp)
            except Exception as exc:
                log.warning("shortcut create failed (%s → %s): %s", new_path, sec_dir, exc)

        return MovedFile(
            original_path=src,
            new_path=new_path,
            category_id=cat_id,
            reason=assign.reason,
            score=primary_score,
            shortcuts=shortcut_paths,
        )
