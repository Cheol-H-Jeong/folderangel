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
from functools import lru_cache as _lru_cache
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


def _guess_by_time(entry, cats, *, members_by_cat=None, reclassify_mode=False):
    """Pick the project category whose time-window covers the file's
    modified date AND whose multi-axis compatibility with the file
    clears :data:`similarity.THRESHOLD_GUESS_BY_TIME`.

    The composite score (S1 filename-core proper-noun Jaccard, S2
    filename-schema similarity, S3 time proximity, S4 path
    co-residence, S5 body-head proper-noun Jaccard) replaces the
    previous bare-set-overlap check.  S4 is forced to 0 in re-classify
    mode so the rescue does not re-anchor on the layout the user is
    escaping.

    ``members_by_cat`` is an optional ``{cat_id: list[Signals]}`` map
    of files already assigned by upstream stages, used to compute S2
    (schema sim against existing members) and S5 / S1 (proper-noun
    Jaccard against the union of member nouns).  When omitted, only
    the category's name/description proper-nouns drive S1/S5.

    When several categories pass the threshold, the *highest-score*
    one wins (with narrower time windows acting as a tiebreaker).
    """
    from . import similarity as _sim

    try:
        _ = entry.modified.date()
    except Exception:
        return None

    file_sig = _sim.signals_for_entry(entry)
    if not file_sig.name_pn and not file_sig.body_pn:
        # No identity anchor at all → refuse to guess.
        return None

    best_cid: Optional[str] = None
    best_score = -1.0
    best_span: Optional[int] = None
    for c in cats:
        if c.id == "misc":
            continue
        rng = _parse_time_label_range(c.time_label or "")
        if rng is None:
            continue
        start, end = rng
        if not (start <= file_sig.modified <= end):
            continue
        members = (members_by_cat or {}).get(c.id, [])
        cat_sig = _sim.category_signals(c, members=members, time_range=rng)
        score = _sim.compatibility(
            file_sig, cat_sig, reclassify_mode=reclassify_mode
        )
        if score < _sim.THRESHOLD_GUESS_BY_TIME:
            continue
        span = (end - start).days
        # Pick max score; on tie, prefer narrower window (burst > multi-year).
        if (
            score > best_score + 1e-6
            or (abs(score - best_score) <= 1e-6 and (best_span is None or span < best_span))
        ):
            best_cid = c.id
            best_score = score
            best_span = span
    return best_cid


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


# ----- opaque-filename detector + keyword-overlap veto --------------------
#
# Pre-Pass-1 safety net.  Filenames that carry zero project identity
# (pure numerics, random hashes, IMG_*, raw camera/screen captures)
# must never be classified by the filename-only LLM pass — there is
# nothing in the name for the LLM to anchor to, and past runs showed
# the LLM happily lumping these into the most-active project category.
# We force them straight into the deferred bag so the body-aware
# pipeline gets to look at content (or, for unparseable media, route
# them to a dedicated 미디어 자료 bucket).
_OPAQUE_NAME_PATTERNS = [
    re.compile(r"^\s*\d{2,}\s*$"),                       # 1152, 1767000341906
    re.compile(r"^[A-Za-z0-9_\-]{18,}$"),                # random hashes / base64-ish
    re.compile(r"^IMG[_\-]?\d+", re.IGNORECASE),         # phone camera
    re.compile(r"^DSC[_\-]?\d+", re.IGNORECASE),         # DSLR camera
    re.compile(r"^Screenshot[_\- ]", re.IGNORECASE),
    re.compile(r"^Screen[_\- ]?Shot[_\- ]", re.IGNORECASE),
    re.compile(r"^Recording[_\- ]?\d+", re.IGNORECASE),
    re.compile(r"^Untitled[_\- ]?\d*", re.IGNORECASE),
    re.compile(r"^새\s*문서", re.IGNORECASE),
    re.compile(r"^무제[_\- ]?\d*", re.IGNORECASE),
]
_MEDIA_EXTS = {
    ".mp4", ".mov", ".avi", ".mkv", ".webm",
    ".jpg", ".jpeg", ".png", ".gif", ".heic", ".webp", ".bmp", ".tiff",
}


def _is_opaque_filename(name: str, ext: str = "") -> bool:
    """A filename so generic / opaque that the filename-only LLM pass
    would just guess.  Includes pure-numeric, random-hash, camera
    auto-names, and media files whose stem has no meaningful tokens."""
    if not name:
        return True
    stem = re.sub(r"\.[A-Za-z0-9]{1,5}$", "", name).strip()
    if not stem:
        return True
    for pat in _OPAQUE_NAME_PATTERNS:
        if pat.match(stem):
            return True
    e = (ext or "").lower()
    if not e:
        m = re.search(r"\.([A-Za-z0-9]{1,5})$", name)
        e = "." + m.group(1).lower() if m else ""
    if e in _MEDIA_EXTS:
        # Media file: only trust the filename if the stem contains a
        # meaningful (non-numeric, non-IMG_*) token that could carry
        # project identity.  Bare "IMG_xxxx" / "video123" → opaque.
        meaningful = re.findall(r"[A-Za-z가-힣]{3,}", stem)
        meaningful = [
            t for t in meaningful
            if not re.fullmatch(r"(?:IMG|DSC|VID|REC|MOV|SCR|TMP)[A-Za-z]*", t, re.IGNORECASE)
        ]
        if not meaningful:
            return True
    return False


def _filename_tokens(name: str) -> set[str]:
    """Substantive tokens from a filename for the keyword-overlap veto.

    Strip the extension, split on common separators, lowercase, drop
    pure-numeric and 1-char tokens, and drop the filename-pattern
    noise (``IMG``, ``DSC``, ``v1``, etc.).  Korean tokens are kept
    only when they are 2+ chars.
    """
    stem = re.sub(r"\.[A-Za-z0-9]{1,5}$", "", name or "")
    raw = re.split(r"[\s_\-\.,()\[\]{}<>]+", stem)
    out: set[str] = set()
    for t in raw:
        t = t.strip().casefold()
        if not t or t.isdigit() or len(t) < 2:
            continue
        if re.fullmatch(r"v\d+|r\d+|[ivxlcdm]+", t):
            continue
        if t in {"img", "dsc", "vid", "rec", "mov", "scr", "tmp", "pdf",
                 "docx", "pptx", "xlsx", "hwp", "txt", "csv", "json", "xml",
                 "copy", "of", "fin", "final", "draft", "ver", "version"}:
            continue
        out.add(t)
    return out


def _category_tokens(category) -> set[str]:
    """Substantive tokens from a category's name + keywords + description.

    Accepts either a plain ``dict`` (used during the LLM-call hot path
    where categories are still raw JSON) OR an object with ``.name`` /
    ``.description`` / ``.keywords`` attributes (the dataclass form).
    """
    parts: list[str] = []
    if isinstance(category, dict):
        for k in ("name", "description"):
            v = category.get(k)
            if isinstance(v, str):
                parts.append(v)
        kws = category.get("keywords") or []
        if isinstance(kws, list):
            parts.extend(str(k) for k in kws if k)
    else:
        for k in ("name", "description"):
            v = getattr(category, k, "") or ""
            if isinstance(v, str):
                parts.append(v)
        kws = getattr(category, "keywords", None) or []
        if isinstance(kws, list):
            parts.extend(str(k) for k in kws if k)
    return _filename_tokens(" ".join(parts))


@_lru_cache(maxsize=4096)
def _proper_noun_tokens(text: str) -> frozenset[str]:
    """Named-entity-shaped tokens only: NNP (proper noun) + SL≥3 +
    SH, with person-name and clerical filters applied.

    Used by the conservative ``시기로 추정`` rescue path
    (:func:`_guess_by_time`) where we must NOT snap a file into a
    project category just because their text shares generic NNG nouns
    (``지원``/``운영``/``체계``) or 2-char ASCII abbreviations
    (``AI``/``ML``).  See :func:`folderangel.morph.extract_proper_nouns`
    for the underlying tag rules.
    """
    if not text:
        return frozenset()
    try:
        from . import morph as _morph
        return frozenset(_morph.extract_proper_nouns(text))
    except Exception:
        return frozenset()


def _proper_nouns_for_entry(entry) -> frozenset[str]:
    name = getattr(entry, "name", "") or ""
    excerpt = (getattr(entry, "content_excerpt", "") or "")[:1200]
    text = f"{name}\n{excerpt}".strip()
    return _proper_noun_tokens(text)


def cat_sig_names_substantive(category) -> bool:
    """A brand-new LLM-proposed category is acceptable as a target
    for its very first member iff its name carries at least one
    substantive identifier — a category named only with abstract
    labels ("문서"/"보고서"/"기타") fails this check and the file
    gets deferred regardless of LLM confidence.
    """
    pn = _proper_nouns_for_category(category)
    return bool(pn)


def _proper_nouns_for_category(category) -> frozenset[str]:
    parts: list[str] = []
    if isinstance(category, dict):
        for k in ("name", "description"):
            v = category.get(k)
            if isinstance(v, str):
                parts.append(v)
        kws = category.get("keywords") or []
        if isinstance(kws, list):
            parts.extend(str(k) for k in kws if k)
    else:
        for k in ("name", "description"):
            v = getattr(category, k, "") or ""
            if isinstance(v, str):
                parts.append(v)
        kws = getattr(category, "keywords", None) or []
        if isinstance(kws, list):
            parts.extend(str(k) for k in kws if k)
    return _proper_noun_tokens(" ".join(parts))


def _is_substantive_token(t: str) -> bool:
    """Whether a single token carries enough specificity to count as
    evidence of category membership.

    Two-letter ASCII tokens (AI / ML / VR / UX / NLP-style abbreviations)
    appear in almost every filename AND almost every category name in
    a tech-leaning Korean corpus, so they are not evidence of any
    *specific* project.  Past leak: every student presentation matched
    every AI project category through the bare "ai" token.

    A 2-character *Korean* token, by contrast, is a full word
    (감정 / 분석 / 로봇 / 발표) — denser than a 2-char ASCII abbreviation
    — and stays admissible.
    """
    if len(t) >= 3:
        return True
    return any("가" <= ch <= "힣" for ch in t)


def _tokens_overlap(file_toks: set[str], cat_toks: set[str]) -> bool:
    """Permissive overlap: exact match OR substring containment OR
    common 4-character prefix.  Tolerates real-world variants like
    "projx" vs "projectx" and "한국지역정보개발원" vs "한국지역" while
    still vetoing unrelated tokens like "rtx" vs "한양대".

    *Generic 2-char ASCII tokens never count as overlap.*  In an
    AI-heavy Korean corpus, "ai" appears in essentially every filename
    AND every category name; allowing it as evidence collapses every
    file into the largest AI project bucket.  Korean 2-char tokens
    remain admissible — they are full words, not acronyms.
    """
    if not file_toks or not cat_toks:
        return False
    common = file_toks & cat_toks
    if any(_is_substantive_token(t) for t in common):
        return True
    for ft in file_toks:
        if not _is_substantive_token(ft):
            continue
        for ct in cat_toks:
            if not _is_substantive_token(ct):
                continue
            if ft in ct or ct in ft:
                return True
            if len(ft) >= 4 and len(ct) >= 4 and ft[:4] == ct[:4]:
                return True
    return False


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
            return _plan_from_dict(plan_dict, entries, reclassify_mode=anonymise)

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
        # Rolling-window planner.  Replaces the old hierarchical /
        # filename-first / outlier-discover stack for every provider
        # whose effective context window can hold at least
        # MIN_CHUNK_FILES per call.  See :mod:`folderangel.rolling`.
        # ------------------------------------------------------------------
        try:
            from . import rolling as _rolling
        except Exception as exc:
            log.warning("rolling module unavailable (%s); falling through", exc)
            _rolling = None

        if _rolling is not None and _rolling.should_use_rolling(self.config, len(entries)):
            try:
                if progress:
                    cap = _rolling.estimate_files_capacity(self.config)
                    eff = _rolling.estimate_effective_ctx(self.config)
                    n_chunks = max(1, (len(entries) + cap - 1) // max(1, cap))
                    extra = " + consolidation 1콜" if n_chunks >= 2 else ""
                    progress(
                        f"plan: rolling 모드 — 실효 ctx {eff:,} 토큰, "
                        f"1콜당 최대 {cap}파일 → 예상 {n_chunks}콜{extra}",
                        0.18,
                    )
                plan_dict = self._rolling_plan(entries, payloads, progress)
                return _plan_from_dict(plan_dict, entries, reclassify_mode=anonymise)
            except Exception as exc:
                log.warning("rolling plan failed; falling through: %s", exc)

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
                return _plan_from_dict(plan_dict, entries, reclassify_mode=anonymise)
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
                return _plan_from_dict(plan_dict, entries, reclassify_mode=anonymise)
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
        return _plan_from_dict(plan_dict, entries, reclassify_mode=anonymise)


    # ------------------------------------------------------------------
    def _rolling_plan(
        self,
        entries: list[FileEntry],
        payloads: list[dict],
        progress: Optional[ProgressCB],
    ) -> dict:
        """Linear single-pass planner.  See :mod:`folderangel.rolling`."""
        from . import rolling as _rolling

        anonymise = bool(getattr(self.config, "reclassify_mode", False))
        rows = _rolling.build_rows(entries, reclassify_mode=anonymise)
        eff_ctx = _rolling.estimate_effective_ctx(self.config)

        # Initial chunk size assumes a small (10-cat) catalogue; later
        # chunks recompute based on the actual cumulative catalogue.
        cum_cats: list[dict] = []
        cum_assigns: list[dict] = []   # raw {"i":fid, "c":cid, "p":..., "r":...}
        seen_cat_ids: set[str] = set()

        idx = 0
        total_rows = len(rows)
        while idx < total_rows:
            chunk_size = _rolling.compute_chunk_size(eff_ctx, len(cum_cats))
            chunk = rows[idx : idx + chunk_size]
            chunk_payload = [_rolling.row_to_payload(r) for r in chunk]
            prompt = _rolling.build_rolling_prompt(
                cum_cats,
                chunk_payload,
                ambiguity_threshold=self.config.ambiguity_threshold,
                reclassify_mode=anonymise,
            )
            chunk_no = (idx // max(1, chunk_size)) + 1
            n_chunks_est = (total_rows + chunk_size - 1) // chunk_size
            if progress:
                progress(
                    f"plan: rolling [{chunk_no}/{n_chunks_est}] LLM 호출 "
                    f"({len(chunk)} 시그니처 / 누적 카테고리 {len(cum_cats)})…",
                    0.2 + (idx / max(1, total_rows)) * 0.7,
                )
            try:
                resp = self._llm_call(
                    prompt,
                    heartbeat=self._heartbeat_for(
                        f"rolling [{chunk_no}/{n_chunks_est}] 응답 대기", progress
                    ),
                    stream_label=f"rolling [{chunk_no}/{n_chunks_est}] 토큰 수신",
                    progress=progress,
                )
            except LLMError as exc:
                log.warning("rolling chunk %d failed: %s", chunk_no, exc)
                # Mark every file in this chunk as misc — the
                # ``_plan_from_dict`` rescue will try to recover them.
                for r in chunk:
                    cum_assigns.append({"i": r.fid, "c": "misc",
                                         "p": 0.3, "r": "rolling 호출 실패"})
                idx += len(chunk)
                continue

            for c in (resp.get("new_categories") or []):
                cid = (c.get("id") or "").strip()
                if not cid or cid in seen_cat_ids:
                    continue
                cum_cats.append(c)
                seen_cat_ids.add(cid)
            for a in (resp.get("assignments") or []):
                if not isinstance(a, dict):
                    continue
                cum_assigns.append(a)

            idx += len(chunk)

        # ------------------------------------------------------------------
        # Consolidation pass — only when we actually had to chunk.  The
        # LLM gets just the catalogue (no files) and is asked to merge
        # near-duplicate categories.
        # ------------------------------------------------------------------
        n_chunks = max(1, (total_rows + 1) // max(1, _rolling.compute_chunk_size(eff_ctx, 0)))
        if len(cum_cats) > 0 and n_chunks >= 2:
            if progress:
                progress(
                    f"plan: rolling consolidation — 카테고리 {len(cum_cats)}개 통합 검토",
                    0.93,
                )
            try:
                cprompt = _rolling.build_consolidation_prompt(cum_cats)
                cresp = self._llm_call(
                    cprompt,
                    heartbeat=self._heartbeat_for("rolling consolidation 응답 대기", progress),
                    stream_label="rolling consolidation 토큰 수신",
                    progress=progress,
                )
                merges = cresp.get("merges") or []
                # Apply merges: drop[*] → keep
                drop_to_keep: dict[str, str] = {}
                for m in merges:
                    if not isinstance(m, dict):
                        continue
                    keep = (m.get("keep") or "").strip()
                    if not keep or keep not in seen_cat_ids:
                        continue
                    for d in (m.get("drop") or []):
                        d = (d or "").strip()
                        if d and d in seen_cat_ids and d != keep:
                            drop_to_keep[d] = keep
                if drop_to_keep:
                    cum_cats = [c for c in cum_cats
                                if (c.get("id") or "") not in drop_to_keep]
                    seen_cat_ids = {c.get("id") for c in cum_cats}
                    for a in cum_assigns:
                        cid = (a.get("c") or "").strip()
                        if cid in drop_to_keep:
                            a["c"] = drop_to_keep[cid]
            except Exception as exc:
                log.warning("consolidation pass failed (%s); using raw catalogue", exc)

        # ------------------------------------------------------------------
        # Expand fid-keyed assignments back to per-file (path-keyed) ones,
        # since duplicate signatures collapsed multiple files into one row.
        # ------------------------------------------------------------------
        rows_by_fid = {r.fid: r for r in rows}
        seen_paths: set[str] = set()
        out_assigns: list[dict] = []
        for a in cum_assigns:
            try:
                fid = int(a.get("i"))
            except (TypeError, ValueError):
                continue
            row = rows_by_fid.get(fid)
            if row is None:
                continue
            cid = (a.get("c") or "misc").strip() or "misc"
            try:
                pscore = float(a.get("p") or 0.0)
            except (TypeError, ValueError):
                pscore = 0.0
            reason = (a.get("r") or "").strip() or "rolling 분류"
            for member in row.members:
                p = str(member.path)
                if p in seen_paths:
                    continue
                seen_paths.add(p)
                out_assigns.append({
                    "path": p,
                    "primary": cid,
                    "primary_score": pscore,
                    "secondary": [],
                    "reason": reason,
                })

        # Convert cumulative categories into the same shape ``_plan_from_dict``
        # expects.  Group is mandatory — coerce to 9 if missing.
        out_cats: list[dict] = []
        for c in cum_cats:
            d = dict(c)
            try:
                d["group"] = int(d.get("group") or 9) or 9
            except (TypeError, ValueError):
                d["group"] = 9
            out_cats.append(d)

        return {"categories": out_cats, "assignments": out_assigns}

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


def _plan_from_dict(
    data: dict,
    entries: list[FileEntry],
    *,
    reclassify_mode: bool = False,
) -> Plan:
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

    # Build a {cat_id → list[Signals]} index of files the LLM already
    # assigned with confidence (primary != misc).  ``_guess_by_time``
    # uses these as cluster members for S2 (schema sim) and S5 (body
    # PN) — which is how a 학생 발표자료 that has no clear project
    # name still finds its way to the right student/lecture folder
    # if the LLM placed enough sibling files there.
    from . import similarity as _sim
    members_by_cat: dict[str, list] = {}
    for a in assignments:
        if a.primary_category_id == "misc":
            continue
        e = by_path.get(str(a.file_path))
        if e is None:
            continue
        members_by_cat.setdefault(a.primary_category_id, []).append(
            _sim.signals_for_entry(e)
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
        guess = _guess_by_time(
            entry, cats,
            members_by_cat=members_by_cat,
            reclassify_mode=reclassify_mode,
        )
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
        guess = _guess_by_time(
            entry, cats,
            members_by_cat=members_by_cat,
            reclassify_mode=reclassify_mode,
        )
        if guess is not None and guess != "misc":
            a.primary_category_id = guess
            a.primary_score = max(a.primary_score, 0.45)
            a.reason = "시기로 추정 (사업 기간 일치)"

    return Plan(categories=cats, assignments=assignments)
