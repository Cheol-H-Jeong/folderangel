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
    if mode not in ("new", "incremental", "additive"):
        mode = "new"
    config.reclassify_mode = True

    seed_categories: list[dict] = []
    if mode == "incremental":
        # 재분류 — 기존 최상위 폴더 *전체* 를 카테고리로 활용.
        seed_categories = _seed_categories_from_disk(target_root, fa_only=False)
        if progress:
            progress(
                f"plan: 재분류 모드 — 기존 폴더 {len(seed_categories)}개를 카테고리로 활용",
                0.06,
            )
    elif mode == "additive":
        # 추가 분류 — FolderAngel 가 만들어준 폴더만 카테고리로 활용.
        # 그 안의 파일들은 이미 분류된 것으로 간주, 재분류 안 함.
        # 외부에 떨어진 새 파일들만 FA 폴더(혹은 신규 폴더)로 보냄.
        seed_categories = _seed_categories_from_disk(target_root, fa_only=True)
        from .organizer import is_folderangel_folder_name
        fa_paths: list[Path] = []
        if target_root.is_dir():
            for d in target_root.iterdir():
                if d.is_dir() and is_folderangel_folder_name(d.name):
                    fa_paths.append(d.resolve())
        # Drop any entry whose absolute path lies inside an FA folder.
        if fa_paths:
            kept: list[FileEntry] = []
            skipped_in_fa = 0
            for e in entries:
                try:
                    rp = Path(e.path).resolve()
                except OSError:
                    rp = Path(e.path)
                # Use string-prefix match — a child of an FA dir starts
                # with its directory + os.sep.
                if any(
                    str(rp).startswith(str(fa) + ("/" if "/" in str(fa) else "\\"))
                    or str(rp) == str(fa)
                    for fa in fa_paths
                ):
                    skipped_in_fa += 1
                    continue
                kept.append(e)
            entries = kept
            if progress:
                progress(
                    f"plan: 추가 분류 — 기존 FA 폴더 {len(fa_paths)}개 / "
                    f"이미 분류된 파일 {skipped_in_fa}개 건너뜀 / "
                    f"새 분류 대상 {len(entries)}개",
                    0.06,
                )
        else:
            if progress:
                progress(
                    "plan: 추가 분류 — FA 시그니처 폴더가 없어 신규 분류처럼 동작",
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
    key = None
    if not force_mock:
        key = get_api_key(config, provider=config.llm_provider)
        # Try to build the client even when no key — make_llm_client
        # accepts local URLs (Ollama / vLLM / LM Studio) without auth.
        try:
            client = make_llm_client(config, key)
        except Exception as exc:
            log.warning("llm init failed: %s", exc)
            client = None

    if progress:
        from .config import provider_label
        if client is not None:
            key_state = (
                "키 등록됨" if key else "키 없음(로컬 LLM)"
            )
            progress(
                f"plan: {provider_label(config)} ({config.model}) — "
                f"{key_state} / {config.llm_base_url or '(기본 endpoint)'}",
                0.0,
            )
        else:
            # Tell the user *why* we fell to mock — usually means no
            # key for the provider they just switched to.
            reason = (
                "API 키가 등록되지 않음 — 설정에서 현재 provider 의 키를 등록하세요"
                if not key else "LLM 클라이언트 초기화 실패 — 로그 확인"
            )
            progress(
                f"plan: Mock 휴리스틱 모드 ({provider_label(config)} {reason})",
                0.0,
            )
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


def _seed_categories_from_disk(
    target_root: Path, *, fa_only: bool = False,
) -> list[dict]:
    """Build a seed catalogue from the existing top-level folders of
    ``target_root``.

    ``fa_only=False`` (재분류): every readable sub-folder becomes a
    seed category — convenient when the user has manually curated the
    layout and only wants the LLM to place new files into existing
    bins.

    ``fa_only=True`` (추가 분류): only folders whose name carries the
    ``[FA·xxxxxx]`` signature added by :func:`folder_signature` are
    used.  Anything the user (or another tool) made by hand is left
    out of the catalogue and its contents will be re-evaluated as
    loose files.
    """
    import re
    from .organizer import (
        is_folderangel_folder_name,
        parse_fa_folder_name,
    )
    if not target_root.is_dir():
        return []
    seeds: list[dict] = []
    for entry in sorted(target_root.iterdir()):
        if not entry.is_dir():
            continue
        if entry.name.startswith(".") or entry.name.startswith("__"):
            continue
        raw = entry.name
        if fa_only and not is_folderangel_folder_name(raw):
            continue
        parsed = parse_fa_folder_name(raw)
        if parsed:
            core = parsed["clean_name"] or raw
            time_label = parsed["period"]
            sig = parsed["signature"]
        else:
            # Strip "1. " / "2-" / "3) " sort prefixes.
            core = re.sub(r"^\s*\d+[\.\-_)\]\s]+", "", raw).strip()
            m = re.search(r"[〈(]([^〉)]{1,30})[〉)]\s*$", core)
            time_label = m.group(1).strip() if m else ""
            if m:
                core = core[:m.start()].strip()
            sig = ""
        slug = re.sub(r"[^A-Za-z0-9가-힣]+", "-", core).strip("-").lower()[:40]
        if not slug:
            slug = f"existing-{len(seeds)+1}"
        # If we recovered a FA signature, prefer it as the slug suffix
        # so a future signature() call regenerates the same tag and
        # the folder is reused on disk instead of being created anew.
        if sig:
            slug = f"{slug}-{sig}"
        seeds.append({
            "id": slug,
            "name": core or raw,
            "description": f"기존 폴더: {raw}",
            "time_label": time_label,
            "duration": "mixed",
            "group": (len(seeds) % 8) + 1,
            "_existing_folder": str(entry),
        })
    return seeds

    return op
