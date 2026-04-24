"""High-level orchestration used by both CLI and UI.

This module pulls the scanner/parser/planner/organizer together so that
callers don't need to know about individual stages.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Optional

from .config import Config, get_api_key
from .index import IndexDB
from .llm import GeminiClient
from .metadata import collect
from .models import FileEntry, LLMUsage, OperationResult, Plan
from .organizer import Organizer
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
) -> list[FileEntry]:
    paths = scan(
        root,
        recursive=recursive,
        ignore_patterns=config.ignore_patterns if not config.include_hidden else [],
        max_files=config.max_files,
    )
    entries: list[FileEntry] = []
    for idx, p in enumerate(paths, 1):
        if progress:
            progress(p.name, idx / max(1, len(paths)))
        try:
            entry = collect(p)
        except Exception as exc:
            log.warning("metadata failed %s: %s", p, exc)
            continue
        entry.content_excerpt = extract_excerpt(
            p, max_chars=config.max_excerpt_chars, timeout=config.parse_timeout_s
        )
        entries.append(entry)
    return entries


def run(
    target_root: Path,
    config: Config,
    recursive: bool,
    dry_run: bool,
    index_db: Optional[IndexDB] = None,
    progress: Optional[ProgressCB] = None,
    force_mock: bool = False,
) -> OperationResult:
    target_root = Path(target_root)
    if progress:
        progress("scan", 0.0)
    entries = gather_entries(target_root, config, recursive, progress)

    client: Optional[GeminiClient] = None
    if not force_mock:
        key = get_api_key(config)
        if key:
            try:
                client = GeminiClient(api_key=key, model=config.model)
            except Exception as exc:
                log.warning("gemini init failed: %s", exc)
                client = None

    if progress:
        progress("plan", 0.0)
    planner = Planner(config, gemini=client)
    plan: Plan = planner.plan(entries, progress=progress)

    if progress:
        progress("organize", 0.0)
    organizer = Organizer(config)
    op = organizer.execute(target_root, plan, dry_run=dry_run, progress=progress)

    if client is not None:
        op.llm_usage = LLMUsage(
            request_count=client.request_count,
            prompt_chars=client.prompt_chars,
            response_chars=client.response_chars,
            model=config.model,
        )
    else:
        op.llm_usage = LLMUsage(model="mock")

    if index_db is not None and not dry_run:
        try:
            index_db.record_operation(op)
        except Exception as exc:
            log.warning("index record failed: %s", exc)

    try:
        emit_markdown(op)
    except Exception as exc:
        log.warning("report failed: %s", exc)

    return op
