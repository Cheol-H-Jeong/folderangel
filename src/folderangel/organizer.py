"""Apply a :class:`Plan` to the filesystem.

The organiser is the only module allowed to mutate files.  It is deliberately
tolerant: one failing file must not abort the whole run.  The outcome is
captured in an :class:`OperationResult` which downstream (index, reporter)
consume.
"""
from __future__ import annotations

import logging
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
        # categories don't clutter the target root).
        dir_for: dict[str, Path] = {}
        used_names: set[str] = set()
        for cat in plan.categories:
            name = sanitize_folder_name(cat.name or cat.id)
            base = name
            counter = 2
            while name.lower() in used_names:
                name = f"{base} ({counter})"
                counter += 1
            used_names.add(name.lower())
            dir_for[cat.id] = target_root / name

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
            if progress:
                progress(assign.file_path.name, idx / total)
            try:
                moved_entry = self._apply_one(
                    assign, plan, dir_for, target_root, dry_run, ensure_dir
                )
                if moved_entry is not None:
                    moved.append(moved_entry)
                    used_category_ids.add(moved_entry.category_id)
                    for sp in moved_entry.shortcuts:
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
                skipped.append(SkippedFile(path=assign.file_path, reason=str(exc)))

        # Report only the categories that ended up hosting at least one file.
        surviving_categories = [c for c in plan.categories if c.id in used_category_ids]

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
