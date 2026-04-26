"""Top-level classification planner.

Given a list of ``FileEntry`` objects, coordinate:
  Stage A – per-batch candidate discovery
  Stage A-merge – consolidate to a final category list
  Stage B – per-batch assignment using the final list

Every stage falls back to the heuristic :mod:`folderangel.llm.mock` planner if
the LLM call fails or is unavailable, so the pipeline always yields a usable
plan.  That fallback is the reason we don't hard-fail the user on API errors.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

from .cluster import Cluster, cluster_files, collapse_ratio, signature
from .config import Config
from .llm import LLMError, mock as mock_planner, prompts
from .models import Assignment, Category, FileEntry, Plan, SecondaryAssignment

log = logging.getLogger(__name__)

ProgressCB = Callable[[str, float], None]


def _batched(items: list, size: int):
    for i in range(0, len(items), size):
        yield items[i : i + size]


def _parse_time_label_range(label: str):
    """Parse a Category.time_label into ``(start, end)`` ``date`` pair.

    Recognises::
        "2024"          → 2024-01-01 .. 2024-12-31
        "2024-Q1"       → 2024-01-01 .. 2024-03-31
        "2024-H1"       → 2024-01-01 .. 2024-06-30
        "2024-03"       → 2024-03-01 .. 2024-03-31
        "2023–2025"     → 2023-01-01 .. 2025-12-31
        "2023~2025"     → same
        "2023-2025"     → same (only when both are 4-digit years)
    """
    import re as _re
    from datetime import date

    s = (label or "").strip()
    if not s:
        return None
    m = _re.fullmatch(r"(\d{4})[\-–~](\d{4})", s)
    if m:
        y1, y2 = int(m.group(1)), int(m.group(2))
        if y1 <= y2 <= 9999:
            return date(y1, 1, 1), date(y2, 12, 31)
    m = _re.fullmatch(r"(\d{4})-Q([1-4])", s)
    if m:
        y, q = int(m.group(1)), int(m.group(2))
        m0 = (q - 1) * 3 + 1
        last_day = [31, 31, 30, 30][q - 1] if False else 31  # safe upper
        return date(y, m0, 1), date(y, m0 + 2, 28)
    m = _re.fullmatch(r"(\d{4})-H([12])", s)
    if m:
        y, h = int(m.group(1)), int(m.group(2))
        return (date(y, 1, 1), date(y, 6, 30)) if h == 1 else (date(y, 7, 1), date(y, 12, 31))
    m = _re.fullmatch(r"(\d{4})-(\d{1,2})", s)
    if m:
        y, mo = int(m.group(1)), max(1, min(12, int(m.group(2))))
        return date(y, mo, 1), date(y, mo, 28)
    m = _re.fullmatch(r"(\d{4})", s)
    if m:
        y = int(m.group(1))
        return date(y, 1, 1), date(y, 12, 31)
    return None


def _guess_by_time(entry, cats):
    """Pick the project category whose time-window covers the file's
    modified date.  Returns the category id, or ``None`` if nothing
    matches (in which case the caller falls back to misc).

    Only project-style categories are considered — anything tagged as
    the catch-all ``misc`` is excluded.  When several windows match,
    the *narrowest* (shortest span in days) wins, since burst /
    short-period buckets are stronger evidence than a multi-year
    umbrella programme.
    """
    try:
        d = entry.modified.date()
    except Exception:
        return None
    best = None
    best_span = None
    for c in cats:
        if c.id == "misc":
            continue
        rng = _parse_time_label_range(c.time_label or "")
        if rng is None:
            continue
        start, end = rng
        if start <= d <= end:
            span = (end - start).days
            if best is None or span < best_span:
                best = c.id
                best_span = span
    return best


def _doc_for_cluster_member(entry: FileEntry) -> str:
    body = (entry.content_excerpt or "")[:600]
    return f"{entry.name}\n{body}"


def _cosine_to_ref(ref_doc: str, docs: list[str]) -> list[Optional[float]]:
    """Cosine similarity of every doc against the ref.  Returns ``None``
    for entries when no embedding backend is available — caller treats
    those as "skip outlier check"."""
    from . import embed as _embed
    if _embed.backend_label() == "none" or not docs:
        return [None] * len(docs)
    vecs = _embed.embed([ref_doc] + docs)
    if vecs is None or len(vecs) < 2:
        return [None] * len(docs)
    import numpy as _np
    ref = vecs[0]
    rest = vecs[1:]
    n_ref = _np.linalg.norm(ref) or 1.0
    out: list[Optional[float]] = []
    for r in rest:
        n_r = _np.linalg.norm(r) or 1.0
        out.append(float((ref @ r) / (n_ref * n_r)))
    return out


def _safe_path_repr(path_str: str, is_mojibake, *, anonymise_parents: bool = False) -> str:
    """Redact pre-existing mojibake folder names from a path before we
    show it to the LLM.

    Keeps the leaf filename intact, replaces any parent path component
    that looks like Latin-1-of-UTF-8 garbage with a neutral placeholder
    so the model has no incentive to reuse the broken name.  Also
    strips any leading "{n}." prefix from prior runs.

    When ``anonymise_parents=True`` (re-classify mode), every parent
    component is replaced with ``[folder]`` regardless of mojibake —
    used when the user has explicitly asked to re-classify a corpus
    whose existing folder structure shouldn't anchor the LLM.
    """
    if not path_str:
        return path_str
    p = Path(path_str)
    if not p.parts:
        return path_str
    redacted: list[str] = []
    parts = list(p.parts)
    last_idx = len(parts) - 1
    for i, part in enumerate(parts):
        # Drop a leading "{n}." prefix from parent dirs so the model
        # sees the descriptive part only.
        core = re.sub(r"^\s*\d\.\s+", "", part)
        if anonymise_parents and i < last_idx:
            redacted.append("[folder]")
        elif is_mojibake(core, strict=True):
            redacted.append("[unknown-folder]")
        else:
            redacted.append(part)
    return str(Path(*redacted))


def _strip_payload(files: list[dict]) -> list[dict]:
    """Drop the heaviest fields so a small-context model has room.

    Keeps path + name + ext + a *short* excerpt; drops mime / size /
    accessed timestamps that don't help categorisation.
    """
    out = []
    for f in files:
        out.append(
            {
                "path": f.get("path"),
                "name": f.get("name"),
                "ext": f.get("ext", ""),
                "modified": f.get("modified", ""),
                "excerpt": (f.get("excerpt") or "")[:600],
            }
        )
    return out


def _unique_categories(cat_list: list[dict]) -> list[dict]:
    seen: dict[str, dict] = {}
    for c in cat_list:
        cid = c.get("id") or ""
        cid = cid.strip()
        if not cid:
            continue
        if cid in seen:
            # keep the one with a longer name (more descriptive)
            if len(c.get("name", "")) > len(seen[cid].get("name", "")):
                seen[cid] = c
        else:
            seen[cid] = c
    return list(seen.values())


class Planner:
    def __init__(
        self,
        config: Config,
        gemini: Optional[Any] = None,
        cancel_check: Optional[Callable[[], bool]] = None,
    ) -> None:
        # ``gemini`` is named that way for backwards compatibility, but it
        # accepts any object exposing ``generate_json(prompt, *,
        # heartbeat=None, cancel_check=None)`` and the usage counters —
        # i.e. either :class:`GeminiClient` or :class:`OpenAICompatClient`.
        self.config = config
        self.gemini = gemini
        self.cancel_check = cancel_check

    def _llm_call(
        self,
        prompt: str,
        *,
        heartbeat=None,
        stream_label: str = "LLM 응답 스트림",
        progress: Optional["ProgressCB"] = None,
    ) -> dict:
        """All LLM calls go through here so we can uniformly attach the
        current cancel_check, the progress heartbeat, *and* a
        token-streaming preview callback that surfaces what the model is
        currently producing.

        Tolerant of clients that don't yet support every keyword
        (Gemini's REST client has no stream_text support — we fall back
        to plain non-streaming behaviour for it).
        """
        stream_state = {
            "chars": 0,
            "preview": "",
            "warned": False,
            "buffer": "",        # rolling tail to recover whole chars across chunks
            "last_emit_ts": 0.0,
            "last_shown": "",
        }

        import re as _re
        # Cosmetic-only filter: strip JSON syntax noise that makes the
        # streaming preview look like an encoding error to a human even
        # though the underlying data is fine.  We never mutate the
        # data we send to json.loads — this only affects what's shown
        # in the progress log.
        _json_noise = _re.compile(
            r'(?:\\["nrtu/]|\\u[0-9A-Fa-f]{0,4}|[{}\[\]"\\]|,\s*|:\s*)+'
        )
        _ws = _re.compile(r"\s+")

        def _humanise_preview(window: str) -> str:
            t = window.replace("\n", " ").replace("\r", " ").replace("\t", " ")
            t = _json_noise.sub(" ", t)
            t = "".join(ch if ch.isprintable() else " " for ch in t)
            t = _ws.sub(" ", t).strip()
            # Last 80 chars are enough to feel "alive" without crowding
            # the log line with stale tokens.
            if len(t) > 80:
                t = "…" + t[-80:]
            return t

        import time as _time

        # The token preview line only needs to *feel* alive, not flicker.
        # Throttle UI updates to once every ~1.5 s, and skip emits when
        # the visible portion hasn't actually changed.  The character
        # counter itself is always carried in the line so the user can
        # still see progress.
        _MIN_EMIT_INTERVAL = 1.5

        def _on_stream(chunk: str, total: int):
            stream_state["chars"] = total
            if progress is None:
                return
            from .llm.client import _looks_like_mojibake

            stream_state["buffer"] = (stream_state["buffer"] + chunk)[-200:]
            window = stream_state["buffer"]

            if _looks_like_mojibake(window, strict=True):
                if not stream_state["warned"]:
                    stream_state["warned"] = True
                    progress(
                        "⚠ 응답이 모지바케로 보입니다 — 서버 chat template 또는 양자화 모델 호환 문제일 수 있습니다.",
                        -1.0,
                    )
                shown = "●" * 40
            else:
                shown = _humanise_preview(window)

            now = _time.monotonic()
            if (
                now - stream_state["last_emit_ts"] < _MIN_EMIT_INTERVAL
                and shown == stream_state["last_shown"]
            ):
                # Visually identical line within the throttle window —
                # skip to avoid the per-second flicker the user
                # complained about.
                return
            stream_state["last_emit_ts"] = now
            stream_state["last_shown"] = shown
            progress(f"{stream_label}: {total}자 수신 중 — {shown}", -1.0)

        try:
            return self.gemini.generate_json(
                prompt,
                heartbeat=heartbeat,
                cancel_check=self.cancel_check,
                stream_text=_on_stream,
            )
        except TypeError:
            try:
                return self.gemini.generate_json(
                    prompt, heartbeat=heartbeat, cancel_check=self.cancel_check
                )
            except TypeError:
                return self.gemini.generate_json(prompt, heartbeat=heartbeat)

    def _check_cancel(self) -> None:
        if self.cancel_check is not None and self.cancel_check():
            raise LLMError("canceled by user")

    # -----------------------------------------------------------------
    def plan(
        self,
        entries: list[FileEntry],
        progress: Optional[ProgressCB] = None,
    ) -> Plan:
        if not entries:
            return Plan(categories=[], assignments=[])

        # Build LLM payloads, capping the per-file excerpt so a single
        # giant slide deck cannot blow the request size or read timeout.
        # Also: scrub any mojibake that already exists in the file's
        # *parent directory names* (left over from prior runs on this
        # corpus).  The LLM should ignore those broken folder names and
        # rebuild the folder structure from the file metadata + content
        # only — see ``_safe_path_repr`` for the redaction logic.
        from .llm.client import _looks_like_mojibake

        anonymise = bool(getattr(self.config, "reclassify_mode", False))
        # Re-classify mode hides parent folder names, so the LLM has
        # less context to work with.  Compensate by lifting the per-file
        # excerpt cap to the full max_excerpt_chars so file *content*
        # picks up the slack.
        per_file_cap = (
            self.config.max_excerpt_chars
            if anonymise
            else min(self.config.max_excerpt_chars, 1200)
        )
        payloads = []
        for e in entries:
            d = e.to_summary_dict()
            excerpt = d.get("excerpt", "") or ""
            d["excerpt"] = excerpt[:per_file_cap]
            d["path"] = _safe_path_repr(
                d.get("path") or "", _looks_like_mojibake, anonymise_parents=anonymise
            )
            payloads.append(d)

        # Short-circuit if there's no Gemini client — everything is mock.
        if self.gemini is None:
            if progress:
                # Tell the user which tier *would* have been picked even
                # in mock mode, so the file-count vs strategy mapping
                # is visible regardless of whether a key is configured.
                tier = self._pick_tier(entries)
                progress(self._tier_announcement(tier, len(entries)), 0.16)
                progress("mock-planner: API 키 없음 — 휴리스틱으로 분류합니다.", 0.5)
            plan_dict = mock_planner.plan(payloads, self.config.ambiguity_threshold)
            return _plan_from_dict(plan_dict, entries)

        # ------------------------------------------------------------------
        # Tiered planning policy — choose the cheapest path that still
        # gives accurate folders for *this* corpus size.  Ordered from
        # most LLM-intensive (best quality, fine for ≤ a few hundred
        # files) to most LLM-frugal (best for thousands).  The user is
        # told upfront which tier was picked.
        #
        #   1. ``small``         < small_corpus_files   single LLM call,
        #                                              everything goes
        #                                              into the prompt.
        #                                              Best quality.
        #   2. ``medium``        small ≤ N < large     micro-batch — LLM
        #                                              chunks the corpus
        #                                              and merges; still
        #                                              every file gets
        #                                              individually
        #                                              looked at.
        #   3. ``large``         large ≤ N             hierarchical
        #                                              (signature
        #                                              clusters →
        #                                              representatives →
        #                                              propagate).
        # ------------------------------------------------------------------
        tier = self._pick_tier(entries)
        if progress:
            progress(self._tier_announcement(tier, len(entries)), 0.16)

        # ------------------------------------------------------------------
        # Re-classify filename-first pass.  Only fires when the user
        # explicitly enabled "재분류 모드" AND the corpus is big enough
        # that it would otherwise hit the medium/large tier.  We send a
        # *filename-only* payload (no body, no path tail) so the LLM can
        # cheaply pick out the obvious project/사업/기관 buckets and
        # tell us which files it can't classify by name alone — those
        # get deferred to the body-aware tier pipeline below.
        # ------------------------------------------------------------------
        if (
            anonymise
            and tier in ("medium", "large")
            and len(entries) > int(getattr(self.config, "small_corpus_files", 60))
        ):
            try:
                pass1 = self._filename_first_pass(entries, payloads, progress)
            except Exception as exc:
                log.warning("filename-first pass failed; falling through: %s", exc)
                pass1 = None
            if pass1 is not None:
                seed_cats = pass1["categories"]
                seed_assigns = pass1["assignments"]
                deferred_paths = set(pass1["deferred"])
                deferred_entries = [
                    e for e in entries if str(e.path) in deferred_paths
                ]
                if not deferred_entries:
                    if progress:
                        progress(
                            f"plan: 파일명 패스로 전체 {len(entries)}개 분류 완료",
                            0.95,
                        )
                    return _plan_from_dict(
                        {"categories": seed_cats, "assignments": seed_assigns},
                        entries,
                    )
                if progress:
                    progress(
                        f"plan: 파일명 패스 — 자신있는 분류 {len(seed_assigns)}개, "
                        f"본문/시그니처 후속 분류 {len(deferred_entries)}개",
                        0.3,
                    )
                # Re-tier the deferred subset and run the normal pipeline
                # on it.  The deferred set is strictly smaller, so we do
                # this in-process by recursing into a helper that skips
                # the filename pass to avoid an infinite loop.
                sub_dict = self._plan_deferred(deferred_entries, progress)
                # Merge: dedupe categories on id (seed wins).
                seed_ids = {c.get("id") for c in seed_cats}
                merged_cats = list(seed_cats)
                cap = int(self.config.max_categories)
                for c in sub_dict.get("categories") or []:
                    cid = c.get("id")
                    if not cid or cid in seed_ids:
                        continue
                    if len(merged_cats) >= cap:
                        break
                    merged_cats.append(c)
                    seed_ids.add(cid)
                merged_assigns = list(seed_assigns) + list(
                    sub_dict.get("assignments") or []
                )
                return _plan_from_dict(
                    {"categories": merged_cats, "assignments": merged_assigns},
                    entries,
                )

        if tier == "large":
            try:
                if progress:
                    progress(
                        f"plan: 대규모 모드 — 파일명+본문 시그니처로 묶어 대표만 LLM 호출",
                        0.2,
                    )
                plan_dict = self._hierarchical_plan(entries, progress)
                return _plan_from_dict(plan_dict, entries)
            except Exception as exc:
                log.warning("hierarchical plan failed; falling through: %s", exc)

        # ------------------------------------------------------------------
        # Local LLM (small-context) micro-batch path.  When the provider is
        # an OpenAI-compat backend (Qwen / Llama on Ollama / LM Studio /
        # vLLM ...), a single 100-file prompt usually overflows the
        # context window, takes minutes, and times out.  Instead we:
        #   Pass A  – split files into small chunks, ask each chunk for
        #             *category candidates* only.  Each call is short.
        #   Pass M  – one tiny merge call consolidates candidates into
        #             the final folder list (with group + time_label).
        #   Pass B  – per-chunk assignment using the final categories.
        # Total inference count is bounded by 2 × ceil(N / chunk) + 1,
        # but every single call fits comfortably in 4–8k context.
        if self._should_use_microbatch(payloads):
            try:
                if progress:
                    progress("plan: micro-batch 모드 (컨텍스트 초과 분할)…", 0.2)
                plan_dict = self._microbatch_plan(payloads, progress)
                return _plan_from_dict(plan_dict, entries)
            except Exception as exc:
                log.warning("micro-batch plan failed; falling back: %s", exc)
                # Fall through to economy or the legacy two-stage path.

        # ------------------------------------------------------------------
        # Economy mode: a single LLM call covers categorisation + assignment.
        # This minimises inference count (1 call for ≤ ``economy_max_files``
        # files; 2–3 small calls if we have to chunk) and lets the model see
        # every filename at once, which is critical for spotting recurring
        # project / customer / system names.
        if getattr(self.config, "economy_mode", True):
            try:
                if progress:
                    progress("plan", 0.2)
                plan_dict = self._single_call_plan(payloads, progress)
                return _plan_from_dict(plan_dict, entries)
            except Exception as exc:
                log.warning("economy single-call plan failed; falling back: %s", exc)
                # Fall through to the original two-stage path on hard failure.

        # ---------- Stage A ----------
        candidate_sets: list[list[dict]] = []
        batches = list(_batched(payloads, self.config.batch_size))
        for idx, batch in enumerate(batches, 1):
            if progress:
                progress(
                    f"stage-a [{idx}/{len(batches)}] LLM 호출 ({len(batch)} 파일)…",
                    (idx - 1) / max(1, len(batches)) * 0.4,
                )
            prompt = prompts.build_stage_a(
                batch, reclassify_mode=bool(getattr(self.config, "reclassify_mode", False))
            )
            try:
                resp = self._llm_call(
                    prompt,
                    heartbeat=self._heartbeat_for(
                        f"stage-a [{idx}/{len(batches)}] LLM 응답 대기", progress
                    ),
                    stream_label=f"stage-a [{idx}/{len(batches)}] 토큰 수신",
                    progress=progress,
                )
                cands = resp.get("candidates") or []
                if not isinstance(cands, list):
                    raise LLMError("candidates not a list")
                candidate_sets.append(cands)
            except Exception as exc:
                log.warning("stage-A fallback to mock batch %d: %s", idx, exc)
                mock_out = mock_planner.plan(batch, self.config.ambiguity_threshold)
                candidate_sets.append(mock_out["categories"])

        # ---------- Stage A-merge ----------
        if progress:
            progress("stage-merge: 후보 카테고리 통합 중…", 0.45)
        try:
            merge_prompt = prompts.build_stage_merge(
                candidate_sets,
                self.config.min_categories,
                self.config.max_categories,
                reclassify_mode=bool(getattr(self.config, "reclassify_mode", False)),
            )
            merged = self._llm_call(
                merge_prompt,
                heartbeat=self._heartbeat_for("stage-merge: LLM 응답 대기", progress),
                stream_label="stage-merge 토큰 수신",
                progress=progress,
            )
            categories_raw = merged.get("categories") or []
            if not categories_raw:
                raise LLMError("empty merged categories")
            categories_raw = _unique_categories(categories_raw)[: self.config.max_categories]
        except Exception as exc:
            log.warning("stage-merge fallback to mock: %s", exc)
            # Flatten candidates and deduplicate by id
            flat = [c for cs in candidate_sets for c in cs]
            categories_raw = _unique_categories(flat)[: self.config.max_categories]
            if not categories_raw:
                mock_out = mock_planner.plan(payloads, self.config.ambiguity_threshold)
                categories_raw = mock_out["categories"]

        # ---------- Stage B ----------
        categories_payload = [
            {"id": c["id"], "name": c.get("name", c["id"]), "description": c.get("description", "")}
            for c in categories_raw
        ]
        category_ids = {c["id"] for c in categories_payload}
        assignments_raw: list[dict] = []
        for idx, batch in enumerate(batches, 1):
            if progress:
                progress(
                    f"stage-b [{idx}/{len(batches)}] LLM 호출 ({len(batch)} 파일)…",
                    0.5 + (idx / max(1, len(batches))) * 0.4,
                )
            try:
                assigns = self._stage_b_call(batch, categories_payload, progress)
                assignments_raw.extend(assigns)
            except Exception as exc:
                log.warning("stage-B fallback to mock batch %d: %s", idx, exc)
                mock_out = mock_planner.plan(batch, self.config.ambiguity_threshold)
                for a in mock_out["assignments"]:
                    if a["primary"] not in category_ids:
                        a["primary"] = _closest_category(a["primary"], categories_payload)
                assignments_raw.extend(mock_out["assignments"])

        # Build the final Plan, coercing unknown ids to the best available fallback.
        plan_dict = {"categories": categories_payload, "assignments": assignments_raw}
        return _plan_from_dict(plan_dict, entries)


    # ------------------------------------------------------------------
    def _should_use_microbatch(self, payloads: Optional[list[dict]] = None) -> bool:
        """Decide between single-call economy and 3-pass micro-batch.

        Policy:
          ``on``   → always micro-batch.
          ``off``  → never micro-batch.
          ``auto`` → estimate the prompt size in tokens and compare to the
                     model's context window minus a response budget.
                     Fits = single call (much faster); doesn't fit =
                     micro-batch.

        We need the actual file payloads for the auto path, so the auto
        branch returns a tuple-friendly bool here and the planner passes
        them in.  Callers without payload context get the legacy
        provider-based default.
        """
        mode = (getattr(self.config, "local_microbatch_mode", "auto") or "auto").lower()
        if mode == "on":
            return True
        if mode == "off":
            return False
        # auto — size-based decision.  If we don't yet have payloads to
        # estimate, fall back to the provider hint (Gemini → no
        # micro-batch; OpenAI-compat → tentatively yes, will be revisited
        # once payloads exist).
        if not payloads:
            return (getattr(self.config, "llm_provider", "gemini") or "gemini").lower() in (
                "openai_compat", "openai", "compat",
            )

        # Build the prompt we *would* send and estimate its token cost.
        prompt = prompts.build_single_call(
            payloads,
            self.config.min_categories,
            self.config.max_categories,
            self.config.ambiguity_threshold,
            reclassify_mode=bool(getattr(self.config, "reclassify_mode", False)),
        )
        # Korean+JSON mixed: ~3 chars/token is a safe upper bound.
        prompt_tokens_est = max(1, len(prompt) // 3)

        ctx = getattr(self.gemini, "context_length", lambda *_: None)()
        if not ctx:
            ctx = getattr(self.config, "assumed_ctx_tokens", 8192)
        budget = getattr(self.config, "response_token_budget", 4096)
        usable = max(1024, ctx - budget)
        decision = prompt_tokens_est > usable
        log.info(
            "auto microbatch decision: prompt≈%d tok, ctx=%d, usable=%d → %s",
            prompt_tokens_est, ctx, usable,
            "micro-batch" if decision else "single-call",
        )
        return decision

    def _microbatch_plan(
        self, payloads: list[dict], progress: Optional[ProgressCB]
    ) -> dict:
        """Three-pass plan that fits in small (4–8k) local-LLM contexts."""
        chunk = max(4, int(getattr(self.config, "local_chunk_size", 12)))
        chunks = list(_batched(payloads, chunk))

        # ---- Pass A: per-chunk candidate discovery -----------------
        candidate_sets: list[list[dict]] = []
        for idx, batch in enumerate(chunks, 1):
            self._check_cancel()
            if progress:
                progress(
                    f"micro-batch A [{idx}/{len(chunks)}] 후보 추출 ({len(batch)} 파일)…",
                    0.2 + (idx - 1) / max(1, len(chunks)) * 0.35,
                )
            prompt = prompts.build_compact_discover(
                _strip_payload(batch),
                reclassify_mode=bool(getattr(self.config, "reclassify_mode", False)),
            )
            try:
                resp = self._llm_call(
                    prompt,
                    heartbeat=self._heartbeat_for(
                        f"micro-batch A [{idx}/{len(chunks)}] 응답 대기", progress
                    ),
                    stream_label=f"micro-batch A [{idx}/{len(chunks)}] 토큰 수신",
                    progress=progress,
                )
            except LLMError:
                # If even a small chunk fails, halve it and retry.
                if len(batch) <= 1:
                    raise
                mid = len(batch) // 2
                candidate_sets.extend(
                    self._microbatch_discover_split(batch[:mid], progress)
                )
                candidate_sets.extend(
                    self._microbatch_discover_split(batch[mid:], progress)
                )
                continue
            cands = resp.get("candidates") or []
            if isinstance(cands, list):
                candidate_sets.append(cands)
            else:
                candidate_sets.append([])

        # ---- Pass M: merge candidates into final categories --------
        self._check_cancel()
        if progress:
            progress("micro-batch M: 후보 통합 중…", 0.55)
        merge_prompt = prompts.build_compact_merge(
            candidate_sets,
            self.config.min_categories,
            self.config.max_categories,
            reclassify_mode=bool(getattr(self.config, "reclassify_mode", False)),
        )
        merged = self._llm_call(
            merge_prompt,
            heartbeat=self._heartbeat_for("micro-batch M 응답 대기", progress),
            stream_label="micro-batch M 토큰 수신",
            progress=progress,
        )
        categories_raw = merged.get("categories") or []
        categories_raw = _unique_categories(categories_raw)[: self.config.max_categories]
        if not categories_raw:
            # very last fallback — flatten the candidate list ourselves
            flat = [c for cs in candidate_sets for c in cs]
            categories_raw = _unique_categories(flat)[: self.config.max_categories]
        if not categories_raw:
            raise LLMError("micro-batch produced no categories")

        # ---- Pass B: per-chunk assignment using the final categories
        category_ids = {c["id"] for c in categories_raw}
        assignments_raw: list[dict] = []
        for idx, batch in enumerate(chunks, 1):
            self._check_cancel()
            if progress:
                progress(
                    f"micro-batch B [{idx}/{len(chunks)}] 분류 ({len(batch)} 파일)…",
                    0.6 + (idx / max(1, len(chunks))) * 0.35,
                )
            prompt = prompts.build_compact_assign(
                _strip_payload(batch),
                categories_raw,
                self.config.ambiguity_threshold,
                reclassify_mode=bool(getattr(self.config, "reclassify_mode", False)),
            )
            try:
                resp = self._llm_call(
                    prompt,
                    heartbeat=self._heartbeat_for(
                        f"micro-batch B [{idx}/{len(chunks)}] 응답 대기", progress
                    ),
                    stream_label=f"micro-batch B [{idx}/{len(chunks)}] 토큰 수신",
                    progress=progress,
                )
                assigns = resp.get("assignments") or []
                if not isinstance(assigns, list):
                    raise LLMError("assignments not a list")
                # Coerce unknown ids to misc/closest.
                for a in assigns:
                    if a.get("primary") not in category_ids:
                        a["primary"] = _closest_category(a.get("primary", ""), categories_raw)
                assignments_raw.extend(assigns)
            except LLMError as exc:
                log.warning("micro-batch B chunk %d → mock: %s", idx, exc)
                m = mock_planner.plan(batch, self.config.ambiguity_threshold)
                for a in m["assignments"]:
                    if a["primary"] not in category_ids:
                        a["primary"] = _closest_category(a["primary"], categories_raw)
                assignments_raw.extend(m["assignments"])
        return {"categories": categories_raw, "assignments": assignments_raw}

    def _microbatch_discover_split(
        self, batch: list[dict], progress: Optional[ProgressCB]
    ) -> list[list[dict]]:
        """Split-and-retry helper for Pass A when a chunk is still too big."""
        if not batch:
            return []
        prompt = prompts.build_compact_discover(_strip_payload(batch))
        try:
            resp = self._llm_call(
                prompt,
                heartbeat=self._heartbeat_for("micro-batch A (분할) 응답 대기", progress),
                stream_label="micro-batch A (분할) 토큰 수신",
                progress=progress,
            )
            cands = resp.get("candidates") or []
            return [cands if isinstance(cands, list) else []]
        except LLMError:
            if len(batch) <= 1:
                return [[]]
            mid = len(batch) // 2
            return (
                self._microbatch_discover_split(batch[:mid], progress)
                + self._microbatch_discover_split(batch[mid:], progress)
            )

    # ------------------------------------------------------------------
    def _pick_tier(self, entries: list[FileEntry]) -> str:
        """Return ``"small"``, ``"medium"``, or ``"large"`` based on
        the file count alone.

        File count is the user-visible knob; we honour it strictly so
        that "100개 이상부터 새로운 모드" is what actually happens.
        Even when signature clustering collapses poorly, the
        hierarchical path still works: every member that doesn't fit
        a cluster falls into long-tail and the LLM looks at it
        individually.  The hierarchical body itself logs the
        collapse ratio and any internal fallback.
        """
        n = len(entries)
        small_threshold = int(getattr(self.config, "small_corpus_files", 60) or 60)
        large_threshold = int(getattr(self.config, "hierarchical_min_files", 100) or 100)
        if n < small_threshold:
            tier = "small"
        elif n < large_threshold:
            tier = "medium"
        else:
            tier = "large"
        log.info("tier decision: %d files → %s", n, tier)
        return tier

    def _tier_announcement(self, tier: str, n: int) -> str:
        """A single human-readable line that explains what the planner
        is about to do — surfaced to the user via the progress log so
        they can see the chosen mode and why."""
        if tier == "small":
            return (
                f"plan: 소규모 모드 — {n}개 파일을 한 번의 LLM 호출로 분류합니다 "
                f"(가장 정확)"
            )
        if tier == "medium":
            return (
                f"plan: 중간 규모 모드 — {n}개 파일을 micro-batch 로 나눠 "
                f"여러 번 LLM 호출 (균형형)"
            )
        return (
            f"plan: 대규모 모드 — {n}개 파일을 시그니처로 묶어 대표만 LLM 호출 "
            f"(가장 비용 효율적)"
        )

    def _should_use_hierarchical(
        self, entries: list[FileEntry], payloads: list[dict]
    ) -> bool:
        """Pick the hierarchical path when the corpus is large *and*
        signature analysis would actually save LLM budget.
        """
        threshold = int(getattr(self.config, "hierarchical_min_files", 500) or 500)
        if len(entries) < threshold:
            return False
        # Cheap probe — same signature pass we'd run anyway.
        clusters, long_tail = cluster_files(
            entries, min_cluster_size=int(
                getattr(self.config, "cluster_min_size", 3) or 3
            ),
        )
        ratio = collapse_ratio(len(entries), clusters, len(long_tail))
        # Worth it only if collapse drops cost meaningfully (≤ 60 %).
        decision = ratio <= 0.6
        log.info(
            "hierarchical decision: %d files → %d clusters + %d long-tail "
            "(ratio %.2f) → %s",
            len(entries), len(clusters), len(long_tail), ratio,
            "hierarchical" if decision else "fallthrough",
        )
        return decision

    def _hierarchical_plan(
        self, entries: list[FileEntry], progress: Optional[ProgressCB]
    ) -> dict:
        """Three steps: cluster → ask LLM about reps → propagate."""
        cluster_min = int(getattr(self.config, "cluster_min_size", 3) or 3)
        reps_per = int(getattr(self.config, "reps_per_cluster", 2) or 2)
        clusters, long_tail = cluster_files(entries, min_cluster_size=cluster_min)

        # 1) Build the representative bag (a stand-in mini-corpus).
        # Long-tail entries are NOT added here on purpose — sending them
        # into the first call dilutes the project/사업/기관 axis the LLM
        # is supposed to learn from cleanly-clustered reps.  They are
        # picked up by the dedicated long-tail discover call below
        # (Step 4), which also receives the outliers we expel from
        # clusters via the purity check.
        rep_entries: list[FileEntry] = []
        rep_to_cluster: dict[Path, Cluster] = {}
        for c in clusters:
            for r in c.representatives(reps_per):
                rep_entries.append(r)
                rep_to_cluster[r.path] = c
        if not rep_entries and long_tail:
            # Degenerate corpus — every file landed in the long tail.
            # Fall through to the long-tail discover call directly.
            rep_entries = []

        if not rep_entries and not long_tail:
            # nothing to do
            return {"categories": [], "assignments": []}

        # 2) Ask the LLM (single call when reps fit; micro-batch fallback
        #    if the rep bag itself is too big).
        from .llm.client import _looks_like_mojibake
        anonymise = bool(getattr(self.config, "reclassify_mode", False))
        per_file_cap = (
            self.config.max_excerpt_chars
            if anonymise
            else min(self.config.max_excerpt_chars, 1200)
        )
        rep_payloads: list[dict] = []
        for e in rep_entries:
            d = e.to_summary_dict()
            d["excerpt"] = (d.get("excerpt") or "")[:per_file_cap]
            d["path"] = _safe_path_repr(
                d.get("path") or "", _looks_like_mojibake, anonymise_parents=anonymise
            )
            rep_payloads.append(d)

        categories_raw: list[dict] = []
        rep_assignments: list[dict] = []
        if rep_payloads:
            if progress:
                progress(
                    f"plan: 대표 {len(rep_entries)}개로 단일 호출 시도 "
                    f"(원본 {len(entries)}개, 롱테일 {len(long_tail)}개는 추가 호출에서)",
                    0.25,
                )
            try:
                llm_out = self._single_call_plan(rep_payloads, progress)
            except Exception as exc:
                log.warning("hierarchical: single-call on reps failed (%s) — micro-batch", exc)
                if progress:
                    progress("plan: 대표 micro-batch fallback", 0.5)
                llm_out = self._microbatch_plan(rep_payloads, progress)

            categories_raw = list(llm_out.get("categories") or [])
            rep_assignments = list(llm_out.get("assignments") or [])

        # 3) Build a quick lookup from rep path → assignment so we can
        #    propagate to every cluster member.
        assign_by_rep_path: dict[str, dict] = {}
        for a in rep_assignments:
            p = str(a.get("path") or "")
            if p:
                assign_by_rep_path[p] = a
        # Fallback: lookup by basename when an LLM rewrote the path.
        assign_by_basename: dict[str, dict] = {}
        for a in rep_assignments:
            name = Path(str(a.get("path") or "")).name
            if name:
                assign_by_basename.setdefault(name, a)

        all_assignments: list[dict] = []
        outliers: list[FileEntry] = []
        from . import embed as _embed
        outlier_thresh = float(getattr(self.config, "outlier_min_similarity", 0.30) or 0.30)

        for c in clusters:
            # Pick the strongest rep assignment for this cluster.
            chosen: Optional[dict] = None
            chosen_member: Optional[FileEntry] = None
            for r in c.members:
                a = assign_by_rep_path.get(str(r.path)) or assign_by_basename.get(r.name)
                if a is not None:
                    chosen = a
                    chosen_member = r
                    break
            if chosen is None:
                continue
            cat_id = chosen.get("primary") or "misc"
            score = float(chosen.get("primary_score", 0.6) or 0.6)
            secondary = chosen.get("secondary") or []
            reason = chosen.get("reason") or "클러스터 대표 분류 상속"

            # Outlier check: members whose body looks materially
            # different from the rep's body get demoted to long-tail
            # so the LLM classifies them individually.  Skips when no
            # embedding backend is available (degrades gracefully).
            ref_doc = _doc_for_cluster_member(chosen_member or c.members[0])
            check_docs = [_doc_for_cluster_member(m) for m in c.members]
            sims = _cosine_to_ref(ref_doc, check_docs)
            for m, sim in zip(c.members, sims):
                if sim is not None and sim < outlier_thresh and len(c.members) > 1:
                    outliers.append(m)
                    continue
                all_assignments.append({
                    "path": str(m.path),
                    "primary": cat_id,
                    "primary_score": score,
                    "secondary": secondary,
                    "reason": reason if (chosen_member and str(m.path) == str(chosen_member.path))
                              else f"동일 패턴 클러스터 자동 상속 — {reason}",
                })

        # Long-tail discover+assign: the original long-tail (files that
        # never made it into a signature/embedding cluster) plus the
        # outliers we just expelled from impure clusters go to a single
        # dedicated LLM call that may EITHER reuse an existing category
        # OR propose a *new* one (사업/과제/프로젝트/기관/목적/용도
        # axis).  This is the user-facing "관련이 없으면 롱테일로 넘기고
        # LLM이 추가 카테고리 생성" behaviour.
        longtail_bag: list[FileEntry] = list(long_tail) + list(outliers)
        if longtail_bag:
            if progress:
                progress(
                    f"plan: 롱테일 {len(longtail_bag)}개 추가 분류 "
                    f"(원본 {len(long_tail)} + 클러스터 outlier {len(outliers)})",
                    0.88,
                )
            try:
                payloads_lt: list[dict] = []
                for e in longtail_bag:
                    d = e.to_summary_dict()
                    d["excerpt"] = (d.get("excerpt") or "")[:per_file_cap]
                    d["path"] = _safe_path_repr(
                        d.get("path") or "",
                        _looks_like_mojibake,
                        anonymise_parents=anonymise,
                    )
                    payloads_lt.append(d)
                lt_out = self._longtail_discover_call(
                    payloads_lt, categories_raw, progress
                )
                # Merge any newly-proposed categories, capped at
                # max_categories.  Existing ids win — never replace.
                existing_ids = {c.get("id") for c in categories_raw}
                cap = int(self.config.max_categories)
                for nc in lt_out.get("new_categories") or []:
                    nid = (nc.get("id") or "").strip()
                    if not nid or nid in existing_ids:
                        continue
                    if len(categories_raw) >= cap:
                        break
                    categories_raw.append(nc)
                    existing_ids.add(nid)
                for a in lt_out.get("assignments") or []:
                    a.setdefault("reason", "")
                    a["reason"] = (a.get("reason") or "") + " · 롱테일 재분류"
                    all_assignments.append(a)
            except Exception as exc:
                log.warning(
                    "long-tail discover failed (%s) — leaving to misc fallback", exc
                )
                # leave the long-tail to be picked up by _plan_from_dict's
                # time-based / misc rescue.

        if progress:
            propagated = len(entries) - len(outliers) - len(long_tail)
            progress(
                f"plan: 클러스터 {len(clusters)} → 대표 분류 → 멤버 {propagated}명 자동 상속, "
                f"롱테일/outlier {len(longtail_bag)}명 추가 호출",
                0.92,
            )
        return {"categories": categories_raw, "assignments": all_assignments}

    # ------------------------------------------------------------------
    def _heartbeat_for(self, label: str, progress: Optional[ProgressCB]):
        """Build a heartbeat callback that streams ``label … Ns`` lines.

        We only emit when the integer-second value advances by ≥ 3 to
        keep the live-status line from re-rendering every single second
        (the user reported a "flickering" effect).  The first beat
        always fires so the user sees the call started.
        """
        if progress is None:
            return None
        state = {"last": -10.0}

        def _beat(elapsed: float):
            if elapsed - state["last"] < 3.0:
                return
            state["last"] = elapsed
            progress(f"{label} … {elapsed:.0f}s 경과", -1.0)

        return _beat

    # ------------------------------------------------------------------
    def _single_call_plan(
        self,
        payloads: list[dict],
        progress: Optional[ProgressCB],
    ) -> dict:
        """One Gemini call covers both folder design and file assignment.

        If the file count exceeds ``economy_max_files`` we (a) ask the LLM
        once on a representative sample to discover the project-level
        categories, then (b) do per-chunk assignment using those fixed
        categories.  Even in the chunked path we use **at most**
        ``ceil(N / economy_max_files) + 1`` calls.
        """
        cap = max(20, int(getattr(self.config, "economy_max_files", 120)))

        if len(payloads) <= cap:
            prompt = prompts.build_single_call(
                payloads,
                self.config.min_categories,
                self.config.max_categories,
                self.config.ambiguity_threshold,
                reclassify_mode=bool(getattr(self.config, "reclassify_mode", False)),
            )
            if progress:
                progress(f"plan: LLM 호출 중 ({len(payloads)} 파일)…", -1.0)
            try:
                resp = self._llm_call(
                    prompt,
                    heartbeat=self._heartbeat_for(
                        f"plan: LLM 응답 대기 중 ({len(payloads)} 파일)", progress
                    ),
                    stream_label=f"plan 토큰 수신 ({len(payloads)} 파일)",
                    progress=progress,
                )
            except LLMError as exc:
                if "context exceeded" in str(exc):
                    if progress:
                        progress(
                            "plan: 컨텍스트 초과 — micro-batch 로 자동 전환합니다.",
                            -1.0,
                        )
                    log.warning(
                        "single-call context exceeded — switching to micro-batch"
                    )
                    return self._microbatch_plan(payloads, progress)
                raise
            cats = resp.get("categories") or []
            assigns = resp.get("assignments") or []
            if not cats or not isinstance(assigns, list):
                raise LLMError("single-call response missing categories/assignments")
            if progress:
                progress(
                    f"plan: 응답 수신 — 카테고리 {len(cats)}, 분류 {len(assigns)}",
                    -1.0,
                )
            return {"categories": cats, "assignments": assigns}

        # Too many files for one call — design categories from a representative
        # slice (every Nth file across the corpus), then assign in chunks.
        step = max(1, len(payloads) // cap)
        sample = payloads[::step][:cap]
        if progress:
            progress(
                f"plan-design: 폴더 설계 (LLM, 샘플 {len(sample)} 파일)…",
                -1.0,
            )
        design_prompt = prompts.build_single_call(
            sample,
            self.config.min_categories,
            self.config.max_categories,
            self.config.ambiguity_threshold,
            reclassify_mode=bool(getattr(self.config, "reclassify_mode", False)),
        )
        design = self._llm_call(
            design_prompt,
            heartbeat=self._heartbeat_for("plan-design: LLM 응답 대기", progress),
            stream_label="plan-design 토큰 수신",
            progress=progress,
        )
        categories = design.get("categories") or []
        if not categories:
            raise LLMError("design pass returned no categories")

        # Reuse the (relatively) cheap Stage-B prompt for the per-chunk
        # assignment so the LLM doesn't waste tokens redesigning categories.
        assignments_raw: list[dict] = []
        chunks = list(_batched(payloads, cap))
        for idx, chunk in enumerate(chunks, 1):
            if progress:
                progress(
                    f"plan-assign [{idx}/{len(chunks)}] LLM 호출 ({len(chunk)} 파일)…",
                    0.3 + 0.6 * (idx / len(chunks)),
                )
            try:
                assigns = self._stage_b_call(chunk, categories, progress)
                assignments_raw.extend(assigns)
            except Exception as exc:
                log.warning("economy assign chunk %d fell back to mock: %s", idx, exc)
                m = mock_planner.plan(chunk, self.config.ambiguity_threshold)
                category_ids = {c["id"] for c in categories}
                for a in m["assignments"]:
                    if a["primary"] not in category_ids:
                        a["primary"] = _closest_category(a["primary"], categories)
                assignments_raw.extend(m["assignments"])
        return {"categories": categories, "assignments": assignments_raw}

    # ------------------------------------------------------------------
    def _filename_first_pass(
        self,
        entries: list[FileEntry],
        payloads: list[dict],
        progress: Optional[ProgressCB],
    ) -> Optional[dict]:
        """Filename-only LLM call used by the re-classify mode.

        Sends just ``[{"path", "name", "ext"}]`` per file — no body,
        no metadata.  The LLM produces an initial set of categories
        plus assignments for files it can confidently classify by
        filename alone, and a ``deferred`` list of paths for everything
        else.  Deferred files get the regular body-aware tier
        pipeline downstream.

        Returns ``{"categories", "assignments", "deferred"}`` or
        ``None`` if the LLM call hard-failed (caller falls through
        to the regular pipeline).
        """
        # Strip every payload down to filename-only for the prompt.
        # The path field is preserved (already anonymised by the
        # caller) so we can match assignments back to entries.
        slim = [
            {
                "path": d.get("path") or "",
                "name": d.get("name") or "",
                "ext": d.get("ext") or "",
            }
            for d in payloads
        ]
        if not slim:
            return None

        # Cap chunk size so a corpus of ~5,000 filenames stays within
        # any modern context window.  ~50 chars/file × 200 files ≈
        # 10K chars of payload — safe for 8K-context local models too.
        chunk_size = max(50, int(getattr(self.config, "economy_max_files", 120)))
        chunks = [slim[i : i + chunk_size] for i in range(0, len(slim), chunk_size)]
        if progress:
            progress(
                f"plan: 파일명 패스 — {len(slim)}개 파일을 {len(chunks)} 청크로 분리",
                0.18,
            )

        agg_cats: list[dict] = []
        agg_assigns: list[dict] = []
        agg_deferred: list[str] = []
        seen_cat_ids: set[str] = set()
        for idx, chunk in enumerate(chunks, 1):
            prompt = prompts.build_filename_first_pass(
                chunk,
                self.config.min_categories,
                self.config.max_categories,
                self.config.ambiguity_threshold,
                reclassify_mode=True,
            )
            if progress:
                progress(
                    f"plan: 파일명 패스 [{idx}/{len(chunks)}] LLM 호출 ({len(chunk)} 파일)…",
                    0.18 + (idx - 1) / max(1, len(chunks)) * 0.1,
                )
            try:
                resp = self._llm_call(
                    prompt,
                    heartbeat=self._heartbeat_for(
                        f"파일명 패스 [{idx}/{len(chunks)}] 응답 대기", progress
                    ),
                    stream_label=f"파일명 패스 [{idx}/{len(chunks)}] 토큰 수신",
                    progress=progress,
                )
            except LLMError as exc:
                log.warning("filename-first chunk %d failed: %s — defer all", idx, exc)
                # Defer the entire chunk so nothing is silently dropped.
                agg_deferred.extend(d.get("path") or "" for d in chunk)
                continue
            cats = resp.get("categories") or []
            assigns = resp.get("assignments") or []
            deferred = resp.get("deferred") or []
            if isinstance(cats, list):
                for c in cats:
                    cid = (c.get("id") or "").strip()
                    if not cid or cid in seen_cat_ids:
                        continue
                    agg_cats.append(c)
                    seen_cat_ids.add(cid)
            if isinstance(assigns, list):
                for a in assigns:
                    a.setdefault("reason", "")
                    a["reason"] = (a.get("reason") or "") + " · 파일명 패스"
                    agg_assigns.append(a)
            if isinstance(deferred, list):
                agg_deferred.extend(str(p) for p in deferred if p)

        if not agg_cats and not agg_assigns:
            return None  # complete miss → caller falls through
        return {
            "categories": agg_cats,
            "assignments": agg_assigns,
            "deferred": agg_deferred,
        }

    # ------------------------------------------------------------------
    def _plan_deferred(
        self,
        deferred_entries: list[FileEntry],
        progress: Optional[ProgressCB],
    ) -> dict:
        """Run the regular tier pipeline on the deferred subset only.

        We rebuild minimal payloads from the deferred FileEntry objects
        and dispatch through the same tier picker as ``plan()`` — but
        skip the filename-first pass to avoid recursion.
        """
        from .llm.client import _looks_like_mojibake

        anonymise = bool(getattr(self.config, "reclassify_mode", False))
        per_file_cap = (
            self.config.max_excerpt_chars
            if anonymise
            else min(self.config.max_excerpt_chars, 1200)
        )
        payloads: list[dict] = []
        for e in deferred_entries:
            d = e.to_summary_dict()
            d["excerpt"] = (d.get("excerpt") or "")[:per_file_cap]
            d["path"] = _safe_path_repr(
                d.get("path") or "", _looks_like_mojibake, anonymise_parents=anonymise
            )
            payloads.append(d)

        tier = self._pick_tier(deferred_entries)
        if tier == "large":
            try:
                return self._hierarchical_plan(deferred_entries, progress)
            except Exception as exc:
                log.warning("deferred hierarchical failed (%s) — falling through", exc)
        if self._should_use_microbatch(payloads):
            try:
                return self._microbatch_plan(payloads, progress)
            except Exception as exc:
                log.warning("deferred micro-batch failed (%s) — falling through", exc)
        try:
            return self._single_call_plan(payloads, progress)
        except Exception as exc:
            log.warning("deferred single-call failed (%s) — empty", exc)
            return {"categories": [], "assignments": []}

    # ------------------------------------------------------------------
    def _longtail_discover_call(
        self,
        files: list[dict],
        categories_payload: list[dict],
        progress: Optional[ProgressCB] = None,
    ) -> dict:
        """Discover-or-assign call for the long-tail bag.

        The LLM gets the *existing* categories AND files that didn't
        fit any of them.  For each file it may either pick an existing
        category id or propose a brand-new one (사업/과제/프로젝트/
        기관/목적/용도 axis).  Returns ``{"new_categories": [...],
        "assignments": [...]}`` — the caller merges new categories
        into the running list (capped by max_categories) and appends
        the assignments.

        On JSON / network failure we halve the batch and retry, same
        recovery shape as ``_stage_b_call``.
        """
        if not files:
            return {"new_categories": [], "assignments": []}
        prompt = prompts.build_longtail_discover(
            files,
            categories_payload,
            self.config.ambiguity_threshold,
            reclassify_mode=bool(getattr(self.config, "reclassify_mode", False)),
        )
        try:
            resp = self._llm_call(
                prompt,
                heartbeat=self._heartbeat_for(
                    f"long-tail: LLM 응답 대기 ({len(files)} 파일)", progress
                ),
                stream_label=f"long-tail 토큰 수신 ({len(files)} 파일)",
                progress=progress,
            )
        except LLMError as exc:
            if len(files) <= 1:
                raise
            log.warning(
                "long-tail split (%d → %d+%d) after error: %s",
                len(files), len(files) // 2, len(files) - len(files) // 2, exc,
            )
            mid = len(files) // 2
            left = self._longtail_discover_call(
                files[:mid], categories_payload, progress
            )
            right = self._longtail_discover_call(
                files[mid:], categories_payload, progress
            )
            return {
                "new_categories": list(left.get("new_categories") or [])
                + list(right.get("new_categories") or []),
                "assignments": list(left.get("assignments") or [])
                + list(right.get("assignments") or []),
            }
        new_cats = resp.get("new_categories") or []
        assigns = resp.get("assignments") or []
        if not isinstance(new_cats, list):
            new_cats = []
        if not isinstance(assigns, list):
            raise LLMError("long-tail assignments not a list")
        return {"new_categories": new_cats, "assignments": assigns}

    # ------------------------------------------------------------------
    def _stage_b_call(
        self,
        batch: list[dict],
        categories_payload: list[dict],
        progress: Optional[ProgressCB] = None,
    ) -> list[dict]:
        """Run Stage B for one batch.

        On timeout/JSON failure we split the batch in halves and retry, which
        usually clears the read timeout because the prompt becomes shorter.
        Recurses down to single-file batches before giving up.
        """
        prompt = prompts.build_stage_b(
            batch,
            categories_payload,
            self.config.ambiguity_threshold,
            reclassify_mode=bool(getattr(self.config, "reclassify_mode", False)),
        )
        try:
            resp = self._llm_call(
                prompt,
                heartbeat=self._heartbeat_for(
                    f"stage-b: LLM 응답 대기 ({len(batch)} 파일)", progress
                ),
                stream_label=f"stage-b 토큰 수신 ({len(batch)} 파일)",
                progress=progress,
            )
        except LLMError as exc:
            if len(batch) <= 1:
                raise
            log.warning(
                "stage-B split (%d → %d+%d) after error: %s",
                len(batch), len(batch) // 2, len(batch) - len(batch) // 2, exc,
            )
            mid = len(batch) // 2
            return self._stage_b_call(
                batch[:mid], categories_payload, progress
            ) + self._stage_b_call(batch[mid:], categories_payload, progress)
        assigns = resp.get("assignments") or []
        if not isinstance(assigns, list):
            raise LLMError("assignments not a list")
        return assigns


def _closest_category(unknown_id: str, categories: list[dict]) -> str:
    if not categories:
        return unknown_id
    # Simple heuristic: first category.  Callers then still get deterministic results.
    return categories[0]["id"]


def _plan_from_dict(data: dict, entries: list[FileEntry]) -> Plan:
    by_path = {str(e.path): e for e in entries}
    from .llm.client import _looks_like_mojibake, _try_repair_mojibake

    cats: list[Category] = []
    for c in data.get("categories", []):
        try:
            group_val = int(c.get("group", 0) or 0)
        except (TypeError, ValueError):
            group_val = 0
        if group_val < 1 or group_val > 9:
            group_val = 9
        raw_name = str(c.get("name") or c.get("id") or "").strip()
        raw_desc = str(c.get("description", "") or "")

        # Per-field mojibake detection + best-effort repair.  This must be
        # strict because only a single category may be corrupt in an
        # otherwise clean response — the document-level check would miss it.
        if _looks_like_mojibake(raw_name, strict=True):
            repaired = _try_repair_mojibake(raw_name, strict=True)
            if not _looks_like_mojibake(repaired, strict=True) and repaired != raw_name:
                log.info("repaired mojibake category name: %r → %r", raw_name, repaired)
                raw_name = repaired
            else:
                log.warning(
                    "dropping category with mojibake name we cannot repair: %r",
                    raw_name,
                )
                continue
        if _looks_like_mojibake(raw_desc, strict=True):
            raw_desc = _try_repair_mojibake(raw_desc, strict=True)
            if _looks_like_mojibake(raw_desc, strict=True):
                raw_desc = ""  # don't write garbage to the report

        # Replacement char / BOM checks are still a hard reject.
        if any(ch in raw_name for ch in ("�", "﻿")):
            log.warning("dropping category with corrupt name: %r", raw_name)
            continue

        # Duration type, normalised to one of the known buckets so the
        # folder-name composer can rely on a closed vocabulary.
        raw_dur = str(c.get("duration", "") or "").strip().lower()
        if raw_dur not in {"burst", "short", "annual", "multi-year", "multiyear", "mixed"}:
            raw_dur = ""
        if raw_dur == "multiyear":
            raw_dur = "multi-year"
        cats.append(
            Category(
                id=str(c.get("id") or "").strip() or raw_name[:24] or f"cat-{len(cats)+1}",
                name=raw_name or str(c.get("id") or ""),
                description=raw_desc,
                time_label=str(c.get("time_label", "") or "").strip(),
                duration=raw_dur,
                group=group_val,
            )
        )
    cat_ids = {c.id for c in cats}
    # Collapse any LLM-supplied catch-all category to a single canonical
    # "기타" bucket so we never end up with both "기타", "프로젝트 외
    # 자료", and "기타 (2)" living side-by-side.
    catchall_keywords = ("기타", "그 외", "분류되지 않은", "프로젝트 외", "기타 자료")
    canonical: list[Category] = []
    misc_seen = False
    for c in cats:
        is_catchall = (
            c.id == "misc"
            or any(k in (c.name or "") for k in catchall_keywords)
        )
        if is_catchall:
            if misc_seen:
                # Drop duplicate catch-alls — assignments to them get
                # remapped further down.
                continue
            misc_seen = True
            c.id = "misc"
            c.name = "기타"
            c.group = 9
        canonical.append(c)
    cats = canonical
    cat_ids = {c.id for c in cats}
    if "misc" not in cat_ids:
        cats.append(Category(id="misc", name="기타", description="분류하기 어려운 파일", group=9))
        cat_ids.add("misc")

    assignments: list[Assignment] = []
    for a in data.get("assignments", []):
        path_str = str(a.get("path") or "")
        primary = a.get("primary") or "misc"
        if primary not in cat_ids:
            primary = "misc"
        secondary_list: list[SecondaryAssignment] = []
        for s in a.get("secondary", []) or []:
            sid = s.get("id")
            if not sid or sid == primary or sid not in cat_ids:
                continue
            try:
                score = float(s.get("score", 0.0))
            except (TypeError, ValueError):
                score = 0.0
            secondary_list.append(SecondaryAssignment(category_id=sid, score=score))

        entry = by_path.get(path_str)
        if entry is None:
            # LLM may have returned basename; try to find by name suffix
            for p, e in by_path.items():
                if p.endswith(path_str) or path_str.endswith(e.name):
                    entry = e
                    path_str = str(e.path)
                    break
        if entry is None:
            log.debug("assignment for unknown path skipped: %s", path_str)
            continue
        try:
            ps = float(a.get("primary_score", 0.0))
        except (TypeError, ValueError):
            ps = 0.0
        assignments.append(
            Assignment(
                file_path=entry.path,
                primary_category_id=primary,
                primary_score=ps,
                secondary=secondary_list,
                reason=(a.get("reason") or "").strip()[:140],
            )
        )

    # Ensure every entry has an assignment.  For misses (or assignments
    # the LLM dropped to "misc") we make a *project-time* attempt first:
    # if a project category's time_label range contains the file's
    # modified date, place the file there with reason="시기로 추정".
    # Only files that match no project bucket fall to actual 기타.
    covered_paths = {a.file_path for a in assignments}
    for entry in entries:
        if entry.path in covered_paths:
            continue
        guess = _guess_by_time(entry, cats)
        if guess is not None:
            assignments.append(
                Assignment(
                    file_path=entry.path,
                    primary_category_id=guess,
                    primary_score=0.45,
                    secondary=[],
                    reason="시기로 추정 (사업 기간 일치)",
                )
            )
        else:
            assignments.append(
                Assignment(
                    file_path=entry.path,
                    primary_category_id="misc",
                    primary_score=0.3,
                    secondary=[],
                    reason="명확한 단서 없음 — 기타로 분류",
                )
            )

    # Also rescue LLM-supplied "misc" assignments when their modified
    # date sits inside a project category's window — the LLM tends to
    # over-use misc for documents whose project name isn't spelled out.
    by_path = {str(e.path): e for e in entries}
    for a in assignments:
        if a.primary_category_id != "misc":
            continue
        entry = by_path.get(str(a.file_path))
        if entry is None:
            continue
        guess = _guess_by_time(entry, cats)
        if guess is not None and guess != "misc":
            a.primary_category_id = guess
            a.primary_score = max(a.primary_score, 0.45)
            a.reason = "시기로 추정 (사업 기간 일치)"

    return Plan(categories=cats, assignments=assignments)
