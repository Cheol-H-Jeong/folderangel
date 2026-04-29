"""App configuration, paths, and API-key storage."""
from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

_KEYRING_SERVICE = "folder1004"
_KEYRING_USER = "gemini_api_key"  # legacy, kept for migration

CLASSIFICATION_GUIDANCE_PRESETS = [
    {
        "label": "프로젝트 중심",
        "text": "프로젝트명, 고객사명, 사업명처럼 일이 진행된 단위를 가장 우선해서 묶어줘. 같은 프로젝트라도 계약·정산·회의·산출물은 필요하면 분리해줘.",
    },
    {
        "label": "업무/용도 중심",
        "text": "파일이 실제로 어떤 용도인지 우선해줘. 산출물, 계약, 정산, 회의, 참고자료, 미디어처럼 쓰임새가 다르면 같은 주제라도 다른 폴더로 나눠줘.",
    },
    {
        "label": "날짜/기간 중심",
        "text": "작성·수정 시기와 파일명 날짜를 중요하게 봐줘. 월별·분기별·학기별·연도별로 자연스러운 기간 라벨이 보이면 폴더명에 반영해줘.",
    },
    {
        "label": "사람/고객 중심",
        "text": "사람 이름, 고객명, 기관명, 반/팀 이름이 보이면 그 단위를 우선해줘. 사진·계약·피드백·납품본이 섞여 있어도 같은 고객 흐름을 찾기 쉽게 해줘.",
    },
    {
        "label": "최대한 세분화",
        "text": "큰 잡동사니 폴더보다 의미가 분명한 작은 폴더를 선호해. 단, 1~2개짜리 너무 자잘한 폴더는 비슷한 큰 폴더에 흡수해줘.",
    },
    {
        "label": "보수적으로 정리",
        "text": "확신이 낮으면 무리하게 새 폴더를 만들지 말고 기타 또는 검토 필요 쪽으로 보내줘. 중요한 파일이 엉뚱한 곳에 가지 않는 것을 우선해줘.",
    },
    {
        "label": "버림 후보 분리",
        "text": "깨진 다운로드, 임시 파일, 오래된 설치 파일, 자동저장/복구 파일, OS 부산물은 삭제하지 말고 검토용 버림 후보 폴더로 따로 모아줘.",
    },
    {
        "label": "수업/학기 중심",
        "text": "선생님이나 학생 폴더라면 학기, 과목, 수업자료, 과제, 평가, 행정자료 축을 우선해서 폴더를 만들어줘.",
    },
]


def _keyring_user_for(provider: str) -> str:
    p = (provider or "gemini").lower()
    if p in ("openai_compat", "openai", "compat"):
        return "openai_compat_api_key"
    return "gemini_api_key"


@dataclass
class AppPaths:
    root: Path
    config: Path
    index_db: Path
    logs_dir: Path

    def ensure(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)


def provider_label(cfg: "Config") -> str:
    """Human-friendly name for the currently selected LLM provider.

    Used by the UI, reporter, and CLI so we never hard-code "Gemini" once
    the user has switched to OpenAI-compatible providers (Qwen / OpenAI /
    Ollama / OpenRouter / …).
    """
    provider = (getattr(cfg, "llm_provider", "gemini") or "gemini").lower()
    if provider in ("openai_compat", "openai", "compat"):
        # Try to be a bit more specific from the URL when possible.
        url = (getattr(cfg, "llm_base_url", "") or "").lower()
        if "openai.com" in url:
            return "OpenAI"
        if "openrouter" in url:
            return "OpenRouter"
        if "together" in url:
            return "Together"
        if "groq" in url:
            return "Groq"
        if "anthropic" in url:
            return "Anthropic"
        if "ollama" in url or "localhost" in url or "127.0.0.1" in url:
            return "로컬 LLM"
        if "qwen" in url:
            return "Qwen"
        return "OpenAI 호환"
    return "Gemini"


def default_paths() -> AppPaths:
    """Pick a platform-appropriate data dir.

      Linux:    ``$XDG_DATA_HOME/folder1004`` if set, else
                ``~/.local/share/folder1004`` if it already exists,
                else legacy ``~/.folder1004``.
      macOS:    ``~/Library/Application Support/Folder1004``
                (legacy ``~/.folder1004`` is read transparently if the
                user was running an older build).
      Windows:  ``%LOCALAPPDATA%/Folder1004`` (then ``%APPDATA%``,
                then home).

    Override with ``FOLDER1004_HOME`` for tests / portable installs.
    """
    override = os.environ.get("FOLDER1004_HOME")
    if override:
        base = Path(override).expanduser()
    elif sys.platform.startswith("win"):
        base = Path(
            os.environ.get("LOCALAPPDATA")
            or os.environ.get("APPDATA")
            or Path.home()
        ) / "Folder1004"
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support" / "Folder1004"
        legacy = Path.home() / ".folder1004"
        if legacy.exists() and not base.exists():
            base = legacy   # respect existing data from older versions
    else:
        xdg = os.environ.get("XDG_DATA_HOME")
        if xdg:
            base = Path(xdg) / "folder1004"
        else:
            modern = Path.home() / ".local" / "share" / "folder1004"
            legacy = Path.home() / ".folder1004"
            base = modern if modern.exists() or not legacy.exists() else legacy
    return AppPaths(
        root=base,
        config=base / "config.json",
        index_db=base / "index.db",
        logs_dir=base / "logs",
    )


@dataclass
class Config:
    # LLM provider — "gemini" (Google AI Studio native), "openai_compat" (any
    # OpenAI Chat Completions compatible endpoint: OpenAI, Together, Groq,
    # OpenRouter, Ollama with `/v1`, vLLM, LM Studio, Anthropic via gateway,
    # Gemini via the OpenAI-compatible proxy at
    # https://generativelanguage.googleapis.com/v1beta/openai, ...).
    llm_provider: str = "gemini"
    # For openai_compat: required.  For gemini: optional, defaults to the
    # Google generative-language host.
    llm_base_url: str = ""
    model: str = "gemini-2.5-flash"
    # Per-provider remembered values so toggling between Gemini and an
    # OpenAI-compat backend in Settings restores the last configuration
    # used for each one (URL + model).  ``api_key`` is still kept in the
    # OS keyring under provider-specific service names; see config helpers.
    llm_settings_by_provider: dict = field(
        default_factory=lambda: {
            "gemini": {"base_url": "", "model": "gemini-2.5-flash"},
            "openai_compat": {"base_url": "", "model": "gpt-4o-mini"},
        }
    )
    # Named LLM presets — the user's most-recently-saved (provider,
    # base_url, model, reasoning_mode) snapshots.  Switching presets in
    # Settings restores all four fields at once.  An API key is stored
    # per *provider* in the OS keyring, so two presets that share a
    # provider share the key (usually what the user wants); presets on
    # different providers each have their own key.
    llm_presets: list = field(default_factory=list)
    # Name of the currently-active preset.  Empty string = freeform
    # (matches the flat fields above without belonging to any preset).
    active_preset: str = ""
    # Natural-language instructions the user wants to add to every LLM
    # planning prompt. Stored locally only; never pre-filled with secrets.
    classification_guidance: str = ""
    classification_guidance_preset_names: list[str] = field(default_factory=list)
    batch_size: int = 30
    max_files: int = 5000
    min_categories: int = 3
    max_categories: int = 30
    ambiguity_threshold: float = 0.15
    max_excerpt_chars: int = 1800
    parse_timeout_s: float = 5.0
    recursive_default: bool = False
    include_hidden: bool = False
    language: str = "ko"
    appearance: str = "auto"  # auto | light | dark
    # When True, ask the LLM exactly once for the whole corpus; only chunk
    # automatically if the prompt is too large.  This is much cheaper and
    # better for project-name discovery (the LLM sees every filename at once).
    economy_mode: bool = True
    # Soft cap on how many files we send in a single combined call.
    economy_max_files: int = 120
    # Micro-batch path for small-context local LLMs (Qwen / Llama / Phi
    # running on Ollama / LM Studio etc.).  In ``auto`` mode the planner
    # estimates the prompt size in tokens; if the estimate fits inside
    # the model's advertised context window (with a generous safety
    # margin reserved for the response), a single economy call is used
    # — same path Gemini takes.  If it doesn't fit, we fall through to
    # micro-batch.  Force ``on`` / ``off`` to override the heuristic.
    local_microbatch_mode: str = "auto"  # auto | on | off
    local_chunk_size: int = 12  # files per Pass-A and Pass-B call when micro-batch is used
    # Heuristic ceiling assumed when we cannot detect the model's
    # ``n_ctx`` automatically.  Conservative — most modern local models
    # advertise ≥ 8 192.
    assumed_ctx_tokens: int = 8192
    # How many tokens we leave unused for the response (and any chat
    # template / tool overhead).  A 1-call plan needs to fit the prompt
    # *and* the JSON it produces.
    response_token_budget: int = 4096
    # Tiered planning thresholds.  See docs/LARGE_CORPUS.md.
    #   < small_corpus_files          → "small" tier  (single LLM call)
    #   < hierarchical_min_files       → "medium" tier (micro-batch)
    #   ≥ hierarchical_min_files       → "large" tier  (hierarchical),
    #                                    falling back to "medium" if
    #                                    signature collapse < 40 %.
    # Defaults are intentionally aggressive on the user-test side so
    # the hierarchical path engages from ~100 files; tweak upward in
    # real usage if you'd rather pay the medium-tier cost on more
    # corpora.
    small_corpus_files: int = 60        # ≤ this: single LLM call
    hierarchical_min_files: int = 100   # ≥ this: try hierarchical first
    cluster_min_size: int = 3           # signature seen this many times → cluster
    reps_per_cluster: int = 2           # representatives sampled per cluster
    # When the rep's body and a cluster member's body fall below this
    # cosine similarity, the member is *not* propagated to the rep's
    # category — instead it is *expelled to the long-tail* and gets
    # categorised in a separate LLM call that can also propose NEW
    # categories.  Tighter than the old loose 0.30 because at 0.30 a
    # 약관/보험 file passes against a 범정부/사업 representative just
    # because they share clerical Korean nouns.  0 disables the check.
    outlier_min_similarity: float = 0.45

    # Minimum files per folder.  After the rolling planner has placed
    # every file, any category with fewer members than this gets
    # absorbed into the closest larger category by proper-noun overlap
    # — or to "기타" when no good match exists.  Set to 1 to disable.
    # 3 is the default because real users see 1- and 2-file folders as
    # clutter ("자잘한 폴더").
    min_category_size: int = 3

    # Reasoning / "thinking" mode for Qwen3 / DeepSeek-R1 / Magistral /
    # Phi-4-mini-reasoning style models.
    #   "off"  — disable thinking (default, much faster for our JSON task)
    #   "on"   — let the model reason; response will include <think>…</think>
    #            and we transparently strip it before parsing
    #   "auto" — pick based on the model: off for known reasoning models
    #            (matches the "off" behaviour); kept as an alias for
    #            future server-side defaults
    reasoning_mode: str = "off"
    # API key is stored separately (keyring) but mirrored here only if keyring fails
    api_key_fallback: str = ""
    ignore_patterns: list[str] = field(
        default_factory=lambda: [".*", "~$*", "Thumbs.db", ".DS_Store", "desktop.ini"]
    )
    # *Deprecated* — kept only for backwards compatibility while old
    # config files load.  The active control is now ``organize_mode``.
    reclassify_mode: bool = False

    # User-facing organize mode chosen on the start screen:
    #   "new"          신규 분류 — ignore every existing sub-folder,
    #                 build the category catalogue from scratch.  Use
    #                 the very first time the user organises a folder.
    #   "incremental"  재분류 — keep the existing top-level folders as
    #                 the canonical category list and only place new
    #                 / unsorted files into them, appending a brand-new
    #                 category only when no existing folder fits.  Use
    #                 every subsequent time the user pours new files
    #                 into a folder that Folder1004 has organised
    #                 before.
    organize_mode: str = "new"

    # Duplicate-file dedup threshold (bytes).  When ≥ 2 files share
    # the same content (size + hash), all but one are *deleted* after
    # the canonical copy is placed — but only if each duplicate is at
    # least this many bytes.  Setting it to 0 deduplicates every
    # duplicate regardless of size; raising it (e.g. 10 MB) limits
    # dedup to the files that actually save space.
    dedup_min_bytes: int = 1_048_576   # 1 MB — covers media, archives, big PDFs

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Config":
        cfg = cls()
        for k, v in data.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        return cfg


def load_config(paths: Optional[AppPaths] = None) -> Config:
    paths = paths or default_paths()
    paths.ensure()
    if not paths.config.exists():
        return Config()
    try:
        data = json.loads(paths.config.read_text(encoding="utf-8"))
        return Config.from_dict(data)
    except Exception as exc:  # corrupt config → fall back to defaults
        log.warning("config load failed (%s); using defaults", exc)
        return Config()


def save_config(cfg: Config, paths: Optional[AppPaths] = None) -> None:
    paths = paths or default_paths()
    paths.ensure()
    tmp = paths.config.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(cfg.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(paths.config)


# ---------------- API key ----------------

def _try_keyring():  # lazily imported so tests don't need it
    try:
        import keyring  # type: ignore

        return keyring
    except Exception:  # pragma: no cover
        return None


def get_api_key(cfg: Optional[Config] = None, provider: Optional[str] = None) -> Optional[str]:
    """Resolve the API key for the *current* (or specified) provider.

    Lookup order: env → keyring (provider-specific) → legacy fallback
    *only when looking up gemini* → config fallback.  Env vars
    checked depend on provider:

      gemini        → GEMINI_API_KEY, GOOGLE_API_KEY
      openai_compat → OPENAI_API_KEY, FOLDER1004_OPENAI_API_KEY

    Past bug: lookup for ``openai_compat`` would fall back to the
    legacy ``gemini_api_key`` slot when its own slot was empty,
    silently sending the Gemini key to a local Qwen/Ollama backend
    and triggering an auth error.  The legacy slot is only used as a
    fallback for the gemini provider it was originally minted for.
    """
    p = (provider or (cfg.llm_provider if cfg else "gemini")).lower()
    env_keys = (
        ["GEMINI_API_KEY", "GOOGLE_API_KEY"]
        if p == "gemini"
        else ["OPENAI_API_KEY", "FOLDER1004_OPENAI_API_KEY"]
    )
    for name in env_keys:
        v = os.environ.get(name)
        if v:
            return v.strip()
    kr = _try_keyring()
    if kr is not None:
        slots = [_keyring_user_for(p)]
        if p == "gemini" and _KEYRING_USER not in slots:
            # Only the gemini provider uses the legacy slot — it was
            # the original (single-provider) key store.
            slots.append(_KEYRING_USER)
        for slot in slots:
            try:
                value = kr.get_password(_KEYRING_SERVICE, slot)
                if value:
                    return value.strip()
            except Exception as exc:
                log.warning("keyring read failed: %s", exc)
                break
    # config-level fallback: only use it when the *currently active*
    # provider matches the one the fallback was last saved for
    # (config doesn't track that, so be conservative — only for gemini
    # which is the historical default).
    if cfg and cfg.api_key_fallback and p == "gemini":
        return cfg.api_key_fallback.strip()
    return None


def set_api_key(
    key: str,
    cfg: Optional[Config] = None,
    paths: Optional[AppPaths] = None,
    provider: Optional[str] = None,
) -> bool:
    """Persist the API key for *this provider*.  Returns True if stored
    securely (keyring), False if config fallback.
    """
    key = (key or "").strip()
    p = (provider or (cfg.llm_provider if cfg else "gemini")).lower()
    kr = _try_keyring()
    if kr is not None:
        try:
            kr.set_password(_KEYRING_SERVICE, _keyring_user_for(p), key)
            if cfg is not None and cfg.api_key_fallback:
                cfg.api_key_fallback = ""
                save_config(cfg, paths)
            return True
        except Exception as exc:
            log.warning("keyring write failed: %s", exc)
    cfg = cfg or load_config(paths)
    cfg.api_key_fallback = key
    save_config(cfg, paths)
    return False
