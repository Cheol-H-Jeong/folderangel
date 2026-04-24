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
from typing import Callable, Iterable, Optional

from .config import Config
from .llm import GeminiClient, LLMError, mock as mock_planner, prompts
from .models import Assignment, Category, FileEntry, Plan, SecondaryAssignment

log = logging.getLogger(__name__)

ProgressCB = Callable[[str, float], None]


def _batched(items: list, size: int):
    for i in range(0, len(items), size):
        yield items[i : i + size]


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
        gemini: Optional[GeminiClient] = None,
    ) -> None:
        self.config = config
        self.gemini = gemini

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
                progress(f"stage-a {idx}/{len(batches)}", (idx - 1) / max(1, len(batches)) * 0.4)
            prompt = prompts.build_stage_a(batch)
            try:
                resp = self.gemini.generate_json(prompt)
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
            progress("stage-merge", 0.45)
        try:
            merge_prompt = prompts.build_stage_merge(
                candidate_sets,
                self.config.min_categories,
                self.config.max_categories,
            )
            merged = self.gemini.generate_json(merge_prompt)
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
                    f"stage-b {idx}/{len(batches)}",
                    0.5 + (idx / max(1, len(batches))) * 0.4,
                )
            try:
                assigns = self._stage_b_call(batch, categories_payload)
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
            resp = self.gemini.generate_json(prompt)
            cats = resp.get("categories") or []
            assigns = resp.get("assignments") or []
            if not cats or not isinstance(assigns, list):
                raise LLMError("single-call response missing categories/assignments")
            return {"categories": cats, "assignments": assigns}

        # Too many files for one call — design categories from a representative
        # slice (every Nth file across the corpus), then assign in chunks.
        step = max(1, len(payloads) // cap)
        sample = payloads[::step][:cap]
        if progress:
            progress("plan-design", 0.25)
        design_prompt = prompts.build_single_call(
            sample,
            self.config.min_categories,
            self.config.max_categories,
            self.config.ambiguity_threshold,
        )
        design = self.gemini.generate_json(design_prompt)
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
                    f"plan-assign {idx}/{len(chunks)}",
                    0.3 + 0.6 * (idx / len(chunks)),
                )
            try:
                assigns = self._stage_b_call(chunk, categories)
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
    def _stage_b_call(self, batch: list[dict], categories_payload: list[dict]) -> list[dict]:
        """Run Stage B for one batch.

        On timeout/JSON failure we split the batch in halves and retry, which
        usually clears the read timeout because the prompt becomes shorter.
        Recurses down to single-file batches before giving up.
        """
        prompt = prompts.build_stage_b(
            batch, categories_payload, self.config.ambiguity_threshold
        )
        try:
            resp = self.gemini.generate_json(prompt)
        except LLMError as exc:
            if len(batch) <= 1:
                raise
            log.warning(
                "stage-B split (%d → %d+%d) after error: %s",
                len(batch), len(batch) // 2, len(batch) - len(batch) // 2, exc,
            )
            mid = len(batch) // 2
            return self._stage_b_call(batch[:mid], categories_payload) + self._stage_b_call(
                batch[mid:], categories_payload
            )
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
    cats = [
        Category(id=c["id"], name=c.get("name", c["id"]), description=c.get("description", ""))
        for c in data.get("categories", [])
    ]
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
