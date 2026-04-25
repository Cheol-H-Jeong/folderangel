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
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

from .config import Config
from .llm import LLMError, mock as mock_planner, prompts
from .models import Assignment, Category, FileEntry, Plan, SecondaryAssignment

log = logging.getLogger(__name__)

ProgressCB = Callable[[str, float], None]


def _batched(items: list, size: int):
    for i in range(0, len(items), size):
        yield items[i : i + size]


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
        stream_state = {"chars": 0, "preview": "", "warned": False}

        def _on_stream(chunk: str, total: int):
            stream_state["chars"] = total
            # Don't show raw token previews in the live progress log —
            # multi-byte boundaries split mid-character, JSON escapes look
            # like garbage out of context, and even a clean stream looks
            # noisy.  Just show progress count + the active stage label so
            # the user knows it's still moving.
            if progress is not None:
                progress(f"{stream_label}: {total}자 수신 중…", -1.0)
                if not stream_state["warned"]:
                    from .llm.client import _looks_like_mojibake

                    # Only sample the *cumulative* preview to detect
                    # mojibake — never display it.
                    stream_state["preview"] = (stream_state["preview"] + chunk)[-512:]
                    if _looks_like_mojibake(stream_state["preview"]):
                        stream_state["warned"] = True
                        progress(
                            "⚠ 응답이 모지바케로 보입니다 — "
                            "서버 chat template 또는 양자화 모델 호환 문제일 수 있습니다.",
                            -1.0,
                        )

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
        per_file_cap = min(self.config.max_excerpt_chars, 1200)
        payloads = []
        for e in entries:
            d = e.to_summary_dict()
            excerpt = d.get("excerpt", "") or ""
            d["excerpt"] = excerpt[:per_file_cap]
            payloads.append(d)

        # Short-circuit if there's no Gemini client — everything is mock.
        if self.gemini is None:
            if progress:
                progress("mock-planner", 0.5)
            plan_dict = mock_planner.plan(payloads, self.config.ambiguity_threshold)
            return _plan_from_dict(plan_dict, entries)

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
            prompt = prompts.build_stage_a(batch)
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
            prompt = prompts.build_compact_discover(_strip_payload(batch))
            try:
                resp = self._llm_call(
                    prompt,
                    heartbeat=self._heartbeat_for(
                        f"micro-batch A [{idx}/{len(chunks)}] 응답 대기", progress
                    ),
                    stream_label=f"micro-batch A [{idx}/{len(chunks)}] 토큰",
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
        )
        merged = self._llm_call(
            merge_prompt,
            heartbeat=self._heartbeat_for("micro-batch M 응답 대기", progress),
            stream_label="micro-batch M 토큰",
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
            )
            try:
                resp = self._llm_call(
                    prompt,
                    heartbeat=self._heartbeat_for(
                        f"micro-batch B [{idx}/{len(chunks)}] 응답 대기", progress
                    ),
                    stream_label=f"micro-batch B [{idx}/{len(chunks)}] 토큰",
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
                stream_label="micro-batch A (분할) 토큰",
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
    def _heartbeat_for(self, label: str, progress: Optional[ProgressCB]):
        """Build a heartbeat callback that streams ``label … Ns`` lines."""
        if progress is None:
            return None

        def _beat(elapsed: float):
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
            )
            if progress:
                progress(f"plan: LLM 호출 중 ({len(payloads)} 파일)…", -1.0)
            resp = self._llm_call(
                prompt,
                heartbeat=self._heartbeat_for(
                    f"plan: LLM 응답 대기 중 ({len(payloads)} 파일)", progress
                ),
                stream_label=f"plan 토큰 수신 ({len(payloads)} 파일)",
                progress=progress,
            )
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
            batch, categories_payload, self.config.ambiguity_threshold
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
    cats: list[Category] = []
    for c in data.get("categories", []):
        try:
            group_val = int(c.get("group", 0) or 0)
        except (TypeError, ValueError):
            group_val = 0
        # Force a numeric group on every category so naming stays consistent.
        # 0/missing → 9 (catch-all bucket); valid range is 1..9.
        if group_val < 1 or group_val > 9:
            group_val = 9
        # Defensive: filter out garbage LLM output before it reaches disk.
        raw_name = str(c.get("name") or c.get("id") or "").strip()
        # Drop anything containing the Unicode replacement char or BOM —
        # those signal a truncated UTF-8 sequence.
        if any(ch in raw_name for ch in ("�", "﻿")):
            log.warning("dropping category with corrupt name (mojibake): %r", raw_name)
            continue
        cats.append(
            Category(
                id=str(c.get("id") or "").strip() or raw_name[:24] or f"cat-{len(cats)+1}",
                name=raw_name or str(c.get("id") or ""),
                description=str(c.get("description", "") or ""),
                time_label=str(c.get("time_label", "") or "").strip(),
                group=group_val,
            )
        )
    cat_ids = {c.id for c in cats}
    if "misc" not in cat_ids:
        cats.append(Category(id="misc", name="기타", description="분류되지 않은 파일"))
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

    # Ensure every entry has an assignment — fallback to misc for any missed.
    covered = {a.file_path for a in assignments}
    for entry in entries:
        if entry.path in covered:
            continue
        assignments.append(
            Assignment(
                file_path=entry.path,
                primary_category_id="misc",
                primary_score=0.3,
                secondary=[],
                reason="플랜에서 누락되어 기타로 분류",
            )
        )

    return Plan(categories=cats, assignments=assignments)
