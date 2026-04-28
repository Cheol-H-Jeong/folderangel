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
    planner = Planner(config, gemini=client, cancel_check=cancel_check)
    plan: Plan = planner.plan(entries, progress=progress)
    _check()

    if progress:
        progress(f"plan: 카테고리 {len(plan.categories)}개 결정됨", 0.95)
        progress(f"organize: 파일 이동 시작 ({len(plan.assignments)}개)", 0.0)
    organizer = Organizer(config)
    excerpts_map = {str(e.path): (e.content_excerpt or "") for e in entries}
    op = organizer.execute(
        target_root, plan, dry_run=dry_run, progress=progress,
        cancel_check=cancel_check, excerpts=excerpts_map,
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
