"""Rolling-window planner.

Replaces the older clustering / hierarchical / filename-first-pass path
with a single linear sweep:

  · probe the model's effective context window;
  · sort files by signature (collapsing duplicates) and slice into
    chunks that fit comfortably under the *effective* — not advertised
    — ctx;
  · for each chunk, send (categories_so_far + files) to the LLM and ask
    it to (a) propose new categories needed for this batch, (b) assign
    every file to a category by **file-id only** (so neither prompt
    nor response re-emits long Korean filenames);
  · accumulate the catalogue across chunks;
  · if the run actually had to chunk (≥ 2 chunks), make a final
    consolidation call to merge near-duplicate categories that drifted
    apart between chunks.

A small corpus that already fits inside the effective ctx skips chunking
*and* consolidation — single LLM call, full catalogue + full assignment.

The tunable inputs are entirely declarative:

    EFFECTIVE_CTX_RATIO    advertised → effective ctx multiplier per
                           model family (RULER / NIAH-derived)
    DEFAULT_CHUNK_FILES    fallback when ctx probe fails (600)
    MIN_CHUNK_FILES        floor — under this we fall through to the
                           micro-batch path instead
    MAX_CHUNK_FILES        ceiling so a 1M-ctx model doesn't try to
                           reason over 10 000 files at once
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable, Optional

from .config import Config
from .models import FileEntry


# ----- filename signature ------------------------------------------------
#
# Stable hashable key that collapses members of the same document
# family (versions / dates / sequence numbers).  Pulls project /
# agency / system nouns out of the filename via :mod:`folderangel.morph`
# (kiwi-based with a regex fallback) and uses up to ``_SIG_PREFIX_LEN``
# of them as the key.  Migrated here from the old ``cluster`` module
# whose hierarchical-clustering code was retired in favour of the
# rolling planner.
# -------------------------------------------------------------------------

_BOUND = r"(?<![A-Za-z0-9])"
_BOUND_END = r"(?![A-Za-z0-9])"
_VERSION_RE = re.compile(
    rf"{_BOUND}(?:v|ver|version|rev|revision|draft|fin|final|"
    rf"r|R|최종|확정|초안|수정|\d?차)\s*[._-]?\s*\d+(?:[._.\-]\d+)*{_BOUND_END}",
    re.IGNORECASE,
)
_DATE_RE = re.compile(
    rf"(?:"
    rf"{_BOUND}\d{{4}}[-_/.]?\d{{2}}[-_/.]?\d{{2}}{_BOUND_END}"
    rf"|{_BOUND}\d{{2}}\d{{2}}\d{{2}}{_BOUND_END}"
    rf"|{_BOUND}\d{{4}}[-_/.]?\d{{2}}{_BOUND_END}"
    rf"|{_BOUND}\d{{2}}[-_/.]?\d{{2}}[-_/.]?\d{{2}}{_BOUND_END}"
    rf")"
)
_SEQ_RE = re.compile(r"\((\s*\d+\s*)\)|_\d{1,3}$|copy(?:\s*of)?", re.IGNORECASE)
_DECORATION_RE = re.compile(r"^(?:★|※|◎|◆|■|●|○|▶|▷)+\s*")
_PARENS_RE = re.compile(r"\([^()]{1,16}\)")
_NOISE_TOKENS = {
    "복사본", "복사", "사본", "copy", "of", "수정본", "변경본",
    "최종본", "최종판", "최종", "확정본", "발표용", "작성요청",
    "임시", "원본", "공유용", "draft", "final", "fin",
}
_SIG_PREFIX_LEN = 2


def signature(name: str, body_excerpt: str = "") -> str:
    """Stable signature key — see module docstring for the rules."""
    from . import morph

    head = _DECORATION_RE.sub("", name or "")
    head = re.sub(r"\.[A-Za-z0-9]{1,5}$", "", head)
    head = _PARENS_RE.sub(" ", head)
    head = _DATE_RE.sub(" ", head)
    head = _VERSION_RE.sub(" ", head)
    head = _SEQ_RE.sub(" ", head)

    nouns = morph.extract_nouns(head)
    nouns = [n for n in nouns if n not in _NOISE_TOKENS]
    if len(nouns) < _SIG_PREFIX_LEN and body_excerpt:
        body_nouns = morph.extract_nouns(body_excerpt[:1000], top_k=8)
        body_nouns = [n for n in body_nouns if n not in _NOISE_TOKENS]
        for n in body_nouns:
            if n in nouns:
                continue
            nouns.append(n)
            if len(nouns) >= _SIG_PREFIX_LEN:
                break
    if not nouns:
        return ""
    return " ".join(nouns[:_SIG_PREFIX_LEN])

log = logging.getLogger(__name__)

ProgressCB = Callable[[str, float], None]


# Per-(provider, model-family) effective-ctx multiplier.  The values
# come from RULER / NIAH evaluations adjusted for our specific task —
# multi-document classification with a growing category catalogue —
# which is harder than pure needle retrieval.
EFFECTIVE_CTX_RATIO: dict[str, float] = {
    "gemini-2.5-pro": 0.12,        # 1M → ~120K
    "gemini-2.5-flash": 0.06,      # 1M → ~60K
    "gemini-1.5-pro": 0.10,        # 1M → ~100K
    "gemini-1.5-flash": 0.05,      # 1M → ~50K
    "gpt-4o": 0.40,                # 128K → ~50K
    "gpt-4.1": 0.08,               # 1M → ~80K
    "claude-opus": 0.50,           # 200K → ~100K
    "claude-sonnet": 0.40,         # 200K → ~80K
    "claude-haiku": 0.30,
    "qwen2.5-32b": 0.50,           # 32K → ~16K
    "qwen2.5": 0.50,
    "llama3": 0.50,
    "default": 0.50,               # unknown → trust half the ctx
}

# Default model context windows when the API does not report one.
ADVERTISED_CTX: dict[str, int] = {
    "gemini-2.5-pro": 1_048_576,
    "gemini-2.5-flash": 1_048_576,
    "gemini-1.5-pro": 1_048_576,
    "gemini-1.5-flash": 1_048_576,
    "gemini-1.0": 32_768,
    "gpt-4o": 128_000,
    "gpt-4.1": 1_000_000,
    "gpt-4-turbo": 128_000,
    "claude-opus": 200_000,
    "claude-sonnet": 200_000,
    "claude-haiku": 200_000,
    "qwen2.5-32b": 32_768,
    "qwen2.5": 32_768,
    "llama3": 8_192,
}

DEFAULT_CHUNK_FILES = 600       # fallback when probe + table both miss
MIN_CHUNK_FILES = 40            # below this → micro-batch path
MAX_CHUNK_FILES = 5_000         # safety cap (1M-ctx Gemini)
RESPONSE_TOKEN_BUDGET = 12_000  # output room — categories + assignments
SYS_OVERHEAD_TOKENS = 3_000     # system prompt + format spec
CAT_TOKENS_PER_ENTRY = 60       # rough — id + name + 1-line desc
TOKENS_PER_FILE_ROW = 100       # 1 row per signature group


# --- ctx detection ---------------------------------------------------------

def _model_family(model_id: str) -> str:
    """Map an arbitrary model id back to a family key in our tables.

    e.g. ``gemini-2.5-flash-002`` → ``gemini-2.5-flash``.
    """
    m = (model_id or "").lower().strip()
    if not m:
        return "default"
    candidates = sorted(EFFECTIVE_CTX_RATIO.keys(), key=len, reverse=True)
    for key in candidates:
        if key == "default":
            continue
        if key in m:
            return key
    return "default"


@lru_cache(maxsize=64)
def _probe_openai_compat_ctx(base_url: str, model: str) -> Optional[int]:
    """Hit ``/v1/models`` to read ``context_length``.  Cached per
    (base_url, model) so a single planning run only ever probes once.

    Returns ``None`` if the endpoint is auth-gated (cloud providers
    like Google's Gemini-via-OpenAI proxy return 403 without an API
    key in the URL or header) or the response is unparseable.
    """
    try:
        import urllib.request
        with urllib.request.urlopen(f"{base_url}/models", timeout=2.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        for entry in data.get("data") or []:
            if not isinstance(entry, dict):
                continue
            if (entry.get("id") or "").lower().startswith(model.lower()):
                for key in ("context_length", "max_context_length",
                            "n_ctx", "context_window"):
                    v = entry.get(key)
                    if isinstance(v, int) and v > 0:
                        return v
        return None
    except Exception as exc:
        log.debug("ctx probe failed: %s", exc)
        return None


def estimate_effective_ctx(cfg: Config) -> int:
    """Effective context window in tokens, accounting for accuracy
    degradation past the model's reasonable comprehension range.

    For *known* model families (Gemini / Claude / GPT / Qwen — anything
    in :data:`ADVERTISED_CTX`) we trust the table and skip the live
    probe.  Probing a cloud-hosted OpenAI-compat proxy without an auth
    header burns time and emits 403 noise into the log; the table is
    accurate enough.

    Only when the model family is *unknown* AND the ``base_url``
    smells like a local server (``/v1`` path) do we hit ``/v1/models``.
    """
    model = (getattr(cfg, "model", "") or "")
    family = _model_family(model)
    ratio = EFFECTIVE_CTX_RATIO.get(family, EFFECTIVE_CTX_RATIO["default"])
    advertised = ADVERTISED_CTX.get(family) or int(getattr(cfg, "assumed_ctx_tokens", 8192))

    raw_ctx: int = advertised
    if family == "default":
        base_url = (getattr(cfg, "llm_base_url", "") or "").rstrip("/")
        # Only probe if the URL looks like a local server — Ollama /
        # LM Studio / vLLM live at 127.0.0.1 or localhost and don't
        # require auth on /v1/models.  Cloud proxies (Google, Together,
        # Groq) return 403 without an API key, which is just noise.
        if base_url and "/v1" in base_url and (
            "127.0.0.1" in base_url or "localhost" in base_url
            or "0.0.0.0" in base_url
        ):
            probed = _probe_openai_compat_ctx(base_url, model)
            if probed is not None:
                raw_ctx = probed

    return max(2_048, int(raw_ctx * ratio))


def compute_chunk_size(effective_ctx: int, n_categories_estimate: int) -> int:
    """Files per chunk that comfortably fit in *effective* ctx with
    headroom for the system prompt, current category catalogue, and
    response budget.
    """
    overhead = (
        SYS_OVERHEAD_TOKENS
        + n_categories_estimate * CAT_TOKENS_PER_ENTRY
        + RESPONSE_TOKEN_BUDGET
    )
    available = effective_ctx - overhead
    if available <= 0:
        # Ctx is too small even for one file row of overhead — caller
        # (should_use_rolling) sees this as "below MIN_CHUNK_FILES" and
        # routes the corpus to the micro-batch path instead.
        return 0
    n = available // TOKENS_PER_FILE_ROW
    return int(min(n, MAX_CHUNK_FILES))


# --- prompt builders ------------------------------------------------------

ROLLING_SYSTEM = """너는 파일을 사업/과제/프로젝트/기관/목적·용도 단위 폴더로 정리하는 전문가다.

지금까지 합의된 폴더 목록(`categories_so_far`)과 새로 분류해야 할 파일(`files`)을 받는다.
files 의 각 파일은 정수 fid("i") + 파일명("n") + modified 일자("m") + 부모 단서("p") 로 표현된다.

각 파일에 대해 둘 중 하나만 한다:
  1) categories_so_far 중 정말 어울리는 것이 있다 → primary 에 그 id
  2) 어울리는 것이 없다 → new_categories 에 신규 폴더 *직접 제안* 하고 그 id 사용

원칙:
- 카테고리는 **구체적인 사업명/기관명/제품명/시스템명** (예: "한양대 인간-AI 협업 강의",
  "행안부 범정부 AI 공통기반"). 추상 라벨("문서"/"보고서"/"기타") 금지.
- 같은 사업이라도 *기능*(작업물 / 거래·계약 / 회계 / 회의·일정 / 원자료·미디어 /
  외부 참고)이 다르면 별도 폴더로 분리.
- 파일이 *생산된 맥락(발신/수신/청중)* 이 다르면 같은 주제라도 분리하라.
  (예: 같은 AI 주제라도 학생 발표자료와 기업 사업 산출물은 별개.)

**최소 폴더 크기 (HARD RULE)** — 자잘한 폴더가 너무 많이 생기면 사용자에게 쓸모 없다:
- 한 폴더에 들어갈 파일이 **최소 3개**가 되도록 묶어라.
- 단발성 1~2개 파일은 *비슷한 주제·기능의 더 큰 폴더*에 흡수하라
  (예: 단일 영수증 → "회계·세무" 폴더로, 단일 발표자료 → 그 사업의 "산출물" 폴더로).
- 주제·기능 모두 정말 매칭이 어려운 경우만 misc("기타")로 보낸다.
- 신규 카테고리를 만들 때는 **그 카테고리에 들어갈 파일이 이번 chunk + 이미 분류된
  목록 안에 3개 이상 있을 가능성**을 먼저 점검하라. 가능성이 낮으면 만들지 말고
  비슷한 기존 카테고리에 보낸다.
- 정말 단서 없는 파일만 "기타"(misc)로.
- 파일명을 응답에 다시 적지 마라 — fid("i")만 사용."""


ROLLING_INSTRUCTION = """응답 JSON 스키마(엄격, 다른 텍스트 금지):
{{
  "new_categories": [
    {{"id":"slug","name":"구체 폴더명","description":"한 줄","time_label":"2025","duration":"annual","group":1}}
  ],
  "assignments": [
    {{"i":42,"c":"category-id","p":0.92,"r":"한 줄 사유 (40자 이내)"}}
  ]
}}
duration ∈ {{burst, short, annual, multi-year, mixed}}.
group 은 1~9 정수 (잡파일=9).
primary_score("p")<{ambiguity_threshold} 인 모호 파일은 c="misc" 로 보내라.
"""


def build_rolling_prompt(
    categories_so_far: list[dict],
    files: list[dict],
    *,
    ambiguity_threshold: float,
    reclassify_mode: bool = False,
) -> str:
    body = json.dumps(
        {
            "categories_so_far": categories_so_far,
            "files": files,
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )
    instr = ROLLING_INSTRUCTION.format(ambiguity_threshold=ambiguity_threshold)
    hint = (
        "\n사용자가 *재분류*를 요청했다 — 'p' 부모 단서는 의도적으로 가려져 있으니"
        " 파일명만 보고 새 폴더 체계를 직접 설계하라."
        if reclassify_mode else ""
    )
    return f"{ROLLING_SYSTEM}{hint}\n\n{instr}\n\n데이터:\n{body}"


CONSOLIDATION_INSTRUCTION = """다음 카테고리 목록은 여러 chunk 에 걸쳐 누적된 것이라
의미가 매우 비슷한 카테고리가 중복되어 있을 수 있다.  너의 일은:

1) 진짜로 같은 사업/주제인 카테고리들을 **병합** 한다 (merges).
2) 너무 광범위해서 한 카테고리에 모든 파일이 몰리는 경우만 **분리** (splits).
   - 단순히 "큰 카테고리" 라는 이유로 쪼개지 마라.  사업/기능 축에서
     실제로 두 종류의 다른 묶음이 한 카테고리에 들어가 있을 때만.

응답 JSON 스키마(엄격):
{
  "merges": [
    {"keep":"category-id","drop":["category-id-1","category-id-2"],"reason":"한 줄"}
  ],
  "splits": [
    {"split_from":"category-id","into":[{"id":"new-slug","name":"...","description":"...","group":1,"duration":"..."}],"criterion":"한 줄"}
  ]
}
"""


def build_consolidation_prompt(categories: list[dict]) -> str:
    body = json.dumps({"categories": categories}, ensure_ascii=False, separators=(",", ":"))
    return f"{ROLLING_SYSTEM}\n\n{CONSOLIDATION_INSTRUCTION}\n\n데이터:\n{body}"


# --- file row collapsing --------------------------------------------------

@dataclass
class FileRow:
    fid: int
    name: str
    modified: str
    parent_hint: str
    members: list[FileEntry]   # all entries collapsed into this row


def build_rows(
    entries: list[FileEntry],
    *,
    reclassify_mode: bool = False,
) -> list[FileRow]:
    """Group identical-signature files into one row each.

    A signature is computed from filename + 80-char body slice (see
    :func:`folderangel.cluster.signature`) — many real corpora have
    runs of duplicates ("강의평가_*", "건강보험납입내역서_*") that
    collapse 100:1, dramatically shrinking the prompt.

    Parent hints are anonymised in re-classify mode so the LLM cannot
    just inherit the user's existing (broken) folder layout.
    """
    by_sig: dict[str, FileRow] = {}
    fid_counter = 0
    for e in entries:
        body = (e.content_excerpt or "")[:80]
        sig = signature(e.name, body)
        row = by_sig.get(sig)
        if row is None:
            fid_counter += 1
            parent = ""
            if e.path is not None:
                try:
                    parent = "[folder]" if reclassify_mode else Path(e.path).parent.name
                except Exception:
                    parent = ""
            row = FileRow(
                fid=fid_counter,
                name=e.name or "",
                modified=(
                    e.modified.strftime("%Y-%m-%d")
                    if getattr(e, "modified", None)
                    else ""
                ),
                parent_hint=parent,
                members=[],
            )
            by_sig[sig] = row
        row.members.append(e)
    # Stable order: sort by signature so similar files cluster in
    # adjacent rows (helps the LLM see "all 약관 files together").
    rows = list(by_sig.values())
    rows.sort(key=lambda r: r.name)
    # Reassign fids in stable order so a re-run produces the same ids.
    for new_fid, r in enumerate(rows, 1):
        r.fid = new_fid
    return rows


def row_to_payload(row: FileRow) -> dict:
    return {
        "i": row.fid,
        "n": row.name,
        "m": row.modified,
        "p": row.parent_hint,
    }


# --- main entry point -----------------------------------------------------

def estimate_files_capacity(cfg: Config) -> int:
    """Public helper — how many files this provider can handle in
    one rolling-window call before chunking kicks in."""
    eff = estimate_effective_ctx(cfg)
    # Use a small-catalogue assumption (10 categories) for the
    # initial chunk; later chunks have larger catalogues but by then
    # we're already chunking.
    return compute_chunk_size(eff, n_categories_estimate=10)


def should_use_rolling(cfg: Config, n_files: int) -> bool:
    """Use the rolling planner when:
      · the corpus has at least :data:`MIN_CHUNK_FILES` files (smaller
        than that is a one-shot 'small' tier anyway),
      · AND the model can take at least :data:`MIN_CHUNK_FILES` files
        per chunk (so chunking is meaningful — for ctx-starved local
        models the existing micro-batch path is still better).
    """
    if n_files < MIN_CHUNK_FILES:
        return False
    return estimate_files_capacity(cfg) >= MIN_CHUNK_FILES


__all__ = [
    "EFFECTIVE_CTX_RATIO",
    "ADVERTISED_CTX",
    "DEFAULT_CHUNK_FILES",
    "MIN_CHUNK_FILES",
    "MAX_CHUNK_FILES",
    "FileRow",
    "build_rows",
    "row_to_payload",
    "build_rolling_prompt",
    "build_consolidation_prompt",
    "estimate_effective_ctx",
    "compute_chunk_size",
    "estimate_files_capacity",
    "should_use_rolling",
]
