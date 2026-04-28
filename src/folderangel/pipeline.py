"""High-level orchestration used by both CLI and UI.

This module pulls the scanner/parser/planner/organizer together so that
callers don't need to know about individual stages.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Optional

import concurrent.futures as _futures
import os

from .config import Config, default_paths, get_api_key
from .index import IndexDB
from .llm import make_llm_client
from .metadata import collect
from .models import FileEntry, LLMUsage, OperationResult, Plan
from .organizer import Organizer
from .parser_cache import ParserCache
from .parsers import extract_excerpt
from .planner import Planner
from .reporter import emit_markdown
from .scanner import scan

log = logging.getLogger(__name__)

ProgressCB = Callable[[str, float], None]


def gather_entries(
    root: Path,
    config: Config,
    recursive: bool,
    progress: Optional[ProgressCB] = None,
    cancel_check=None,
) -> list[FileEntry]:
    if progress:
        progress("scan: 폴더 검사 시작", 0.0)
    paths = scan(
        root,
        recursive=recursive,
        ignore_patterns=config.ignore_patterns if not config.include_hidden else [],
        max_files=config.max_files,
    )
    if progress:
        progress(f"scan: {len(paths)}개 파일 발견", 0.05)
    # Persistent excerpt cache so unchanged files skip parsing on
    # subsequent runs.  Keyed by (path, mtime, size).
    cache = ParserCache(default_paths().root / "parser_cache.db")

    # Parallel parser pool: parsing is IO + CPU bound and the existing
    # extract_excerpt already uses its own time-bounded ThreadPool, so a
    # modest worker count here parallelises across files cleanly.
    workers = max(2, min(8, (os.cpu_count() or 4)))

    def _parse_one(idx_p: tuple[int, "Path"]) -> Optional[FileEntry]:
        idx, p = idx_p
        if cancel_check is not None and cancel_check():
            raise RuntimeError("canceled by user")
        if progress:
            progress(f"parse [{idx}/{len(paths)}] {p.name}", idx / max(1, len(paths)))
        try:
            entry = collect(p)
        except Exception as exc:
            log.warning("metadata failed %s: %s", p, exc)
            if progress:
                progress(
                    f"  ⚠ 메타데이터 실패: {p.name} ({exc})",
                    idx / max(1, len(paths)),
                )
            return None
        try:
            entry.content_excerpt = cache.get_or_parse(
                entry.path, entry.modified.timestamp(), entry.size,
                lambda: extract_excerpt(
                    entry.path,
                    max_chars=config.max_excerpt_chars,
                    timeout=config.parse_timeout_s,
                ),
            )
        except Exception as exc:
            log.debug("cache lookup failed for %s: %s", p, exc)
            entry.content_excerpt = extract_excerpt(
                entry.path,
                max_chars=config.max_excerpt_chars,
                timeout=config.parse_timeout_s,
            )
        return entry

    entries: list[FileEntry] = []
    try:
        with _futures.ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix="folderangel-pipeline"
        ) as pool:
            for entry in pool.map(_parse_one, enumerate(paths, 1), chunksize=4):
                if entry is not None:
                    entries.append(entry)
    finally:
        cache.close()
    return entries


def run(
    target_root: Path,
    config: Config,
    recursive: bool,
    dry_run: bool,
    index_db: Optional[IndexDB] = None,
    progress: Optional[ProgressCB] = None,
    force_mock: bool = False,
    cancel_check=None,
) -> OperationResult:
    target_root = Path(target_root)

    def _check():
        if cancel_check is not None and cancel_check():
            raise RuntimeError("canceled by user")

    _check()
    entries = gather_entries(target_root, config, recursive, progress, cancel_check)
    _check()

    # ------------------------------------------------------------------
    # Mode resolution.  ``organize_mode`` was added when we split the
    # legacy ``reclassify_mode`` boolean into a 3-state choice on the
    # start screen.  Old configs may only have ``reclassify_mode``;
    # treat that as the new "신규 분류" mode.
    # ------------------------------------------------------------------
    mode = (getattr(config, "organize_mode", "") or "").lower()
    if mode not in ("new", "incremental"):
        mode = "new" if getattr(config, "reclassify_mode", False) else "new"
    # Both modes anonymise sub-folder paths in the prompt — the LLM
    # decides categories from filename + content + (in incremental
    # mode) the seed catalogue, never from broken or already-classified
    # parent folder names.
    config.reclassify_mode = True

    # Incremental mode pre-seeds the rolling planner's catalogue from
    # the existing top-level sub-folders of ``target_root``.  This
    # forces the LLM to *re-use* those folders for new files instead
    # of inventing parallel categories.
    seed_categories: list[dict] = []
    if mode == "incremental":
        seed_categories = _seed_categories_from_disk(target_root)
        if progress:
            progress(
                f"plan: 재분류 모드 — 기존 폴더 {len(seed_categories)}개를 카테고리로 활용",
                0.06,
            )

    # ------------------------------------------------------------------
    # Duplicate detection: skip the LLM round-trip for non-canonical
    # copies of the same file and queue them for deletion after the
    # canonical is placed.
    # ------------------------------------------------------------------
    dedup_groups = []
    canonical_only = entries
    min_bytes = int(getattr(config, "dedup_min_bytes", 1_048_576) or 0)
    if dry_run:
        if progress:
            progress("dedup: Dry-Run 모드 — 중복 검사 건너뜀", 0.07)
    elif min_bytes < 0:
        if progress:
            progress("dedup: 비활성 (dedup_min_bytes < 0)", 0.07)
    else:
        from . import dedup as _dedup
        if progress:
            mb_thr = min_bytes / (1 << 20)
            progress(
                f"dedup: 중복 검사 시작 — 임계값 {mb_thr:.1f} MB / "
                f"{len(entries)} 파일 검사",
                0.06,
            )
        dedup_groups = _dedup.find_duplicate_groups(entries, min_bytes=min_bytes)
        if dedup_groups:
            n_dupes = sum(len(g.duplicates) for g in dedup_groups)
            mb_save = sum(g.total_bytes_freed for g in dedup_groups) / (1 << 20)
            if progress:
                progress(
                    f"dedup: 중복 그룹 {len(dedup_groups)}개 / "
                    f"삭제 예정 {n_dupes}개 / ≈ {mb_save:.1f} MB 회수 예정",
                    0.07,
                )
            duplicate_paths = {
                str(d.path) for g in dedup_groups for d in g.duplicates
            }
            canonical_only = [
                e for e in entries if str(e.path) not in duplicate_paths
            ]
        else:
            if progress:
                progress(
                    f"dedup: 임계값 {min_bytes / (1 << 20):.1f} MB 이상 "
                    f"중복 파일 없음",
                    0.07,
                )

    client = None
    if not force_mock:
        key = get_api_key(config, provider=config.llm_provider)
        if key:
            try:
                client = make_llm_client(config, key)
            except Exception as exc:
                log.warning("llm init failed: %s", exc)
                client = None

    if progress:
        if client is not None:
            from .config import provider_label

            progress(
                f"plan: {provider_label(config)} ({config.model}) 호출 준비", 0.0
            )
        else:
            progress("plan: Mock 휴리스틱 모드", 0.0)
    planner = Planner(
        config, gemini=client, cancel_check=cancel_check,
        seed_categories=seed_categories,
    )
    plan: Plan = planner.plan(canonical_only, progress=progress)
    _check()

    # Add the duplicates back to the plan, each pointing at the same
    # category as its canonical.  The organizer will skip the actual
    # file move (we'll delete them after) but the report shows them.
    if dedup_groups:
        canon_cat: dict[str, str] = {
            str(a.file_path): a.primary_category_id for a in plan.assignments
        }
        from .models import Assignment, SecondaryAssignment
        for g in dedup_groups:
            cid = canon_cat.get(str(g.canonical.path))
            if not cid:
                continue
            for d in g.duplicates:
                plan.assignments.append(Assignment(
                    file_path=d.path,
                    primary_category_id=cid,
                    primary_score=1.0,
                    secondary=[],
                    reason=f"중복 — 정본: {Path(g.canonical.path).name}",
                ))

    if progress:
        progress(f"plan: 카테고리 {len(plan.categories)}개 결정됨", 0.95)
        progress(f"organize: 파일 이동 시작 ({len(plan.assignments)}개)", 0.0)
    organizer = Organizer(config)
    excerpts_map = {str(e.path): (e.content_excerpt or "") for e in entries}
    duplicate_paths = (
        {str(d.path) for g in dedup_groups for d in g.duplicates}
        if dedup_groups else set()
    )
    op = organizer.execute(
        target_root, plan, dry_run=dry_run, progress=progress,
        cancel_check=cancel_check, excerpts=excerpts_map,
        skip_paths=duplicate_paths,
    )

    # ------------------------------------------------------------------
    # After the canonical files are in their final folders, delete the
    # duplicates that no longer earn their disk space.
    # ------------------------------------------------------------------
    if dedup_groups and not dry_run:
        from . import dedup as _dedup
        actions = _dedup.remove_duplicate_files(dedup_groups, dry_run=False)
        op.dupes_removed = [(str(d), str(c), b) for d, c, b in actions]
        op.bytes_freed = sum(b for _d, _c, b in actions)
        if progress:
            mb = op.bytes_freed / (1 << 20)
            progress(
                f"dedup: 중복 파일 {len(actions)}개 삭제 완료 — {mb:.1f} MB 회수",
                0.98,
            )

    if client is not None:
        op.llm_usage = LLMUsage(
            request_count=client.request_count,
            prompt_chars=client.prompt_chars,
            response_chars=client.response_chars,
            model=config.model,
            total_duration_s=getattr(client, "total_duration_s", 0.0),
            calls=list(getattr(client, "calls", [])),
        )
    else:
        op.llm_usage = LLMUsage(model="mock")

    # Write the markdown report FIRST so its path is available to
    # ``record_operation`` for storage in stats_json — that lets the
    # History tab open the report on double-click without globbing.
    try:
        op.report_path = emit_markdown(op)
    except Exception as exc:
        log.warning("report failed: %s", exc)

    if index_db is not None and not dry_run:
        try:
            index_db.record_operation(op)
        except Exception as exc:
            log.warning("index record failed: %s", exc)

    return op


def _seed_categories_from_disk(target_root: Path) -> list[dict]:
    """Build a seed catalogue from the existing top-level folders of
    ``target_root`` — used by the *재분류 (incremental)* mode so the
    LLM places new files into folders the user already approved
    instead of inventing parallel categories.

    Naming convention recognised: "{n}. {name} 〈{period}〉" or
    "{n}. {name} ({period})" or just plain "{name}".  We strip the
    leading "{n}." sort prefix to keep the LLM-facing id stable.
    """
    import re
    if not target_root.is_dir():
        return []
    seeds: list[dict] = []
    for entry in sorted(target_root.iterdir()):
        if not entry.is_dir():
            continue
        if entry.name.startswith(".") or entry.name.startswith("__"):
            continue
        raw = entry.name
        # Strip "1. " / "2-" / "3) " sort prefixes.
        core = re.sub(r"^\s*\d+[\.\-_)\]\s]+", "", raw).strip()
        # Pull out a trailing 〈2025-2026〉 / (2024) period if present.
        m = re.search(r"[〈(]([^〉)]{1,30})[〉)]\s*$", core)
        time_label = m.group(1).strip() if m else ""
        if m:
            core = core[:m.start()].strip()
        slug = re.sub(r"[^A-Za-z0-9가-힣]+", "-", core).strip("-").lower()[:40]
        if not slug:
            slug = f"existing-{len(seeds)+1}"
        seeds.append({
            "id": slug,
            "name": core or raw,
            "description": f"기존 폴더: {raw}",
            "time_label": time_label,
            "duration": "mixed",
            "group": (len(seeds) % 8) + 1,
            "_existing_folder": str(entry),  # informational
        })
    return seeds

    return op
