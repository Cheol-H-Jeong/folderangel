"""Shared dataclasses used throughout the pipeline.

Kept free of heavy dependencies so any layer can import them.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional


@dataclass
class FileEntry:
    path: Path
    name: str
    ext: str
    size: int
    created: datetime
    modified: datetime
    accessed: datetime
    mime: str = ""
    content_excerpt: str = ""

    def to_summary_dict(self) -> dict:
        return {
            "path": str(self.path),
            "name": self.name,
            "ext": self.ext,
            "size": self.size,
            "created": self.created.isoformat(timespec="seconds"),
            "modified": self.modified.isoformat(timespec="seconds"),
            "mime": self.mime,
            "excerpt": self.content_excerpt[:1800],
        }


@dataclass
class Category:
    id: str
    name: str
    description: str = ""
    # Time bucket the LLM picked, free-form but conventional:
    #   "2024-03" / "2024-Q1" / "2024-H1" / "2024" / "2023–2025"
    time_label: str = ""
    # Duration *type* — drives folder-name shape and downstream sorting.
    #   "burst"     short intense work (a month or so)
    #   "short"     a quarter / half-year campaign
    #   "annual"    a one-year project
    #   "multi-year" multi-year programme
    #   "mixed"     no meaningful time pattern
    duration: str = ""
    group: int = 0          # 1..9 — visual grouping prefix; 0 means ungrouped

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SecondaryAssignment:
    category_id: str
    score: float


@dataclass
class Assignment:
    file_path: Path
    primary_category_id: str
    primary_score: float = 0.0
    secondary: list[SecondaryAssignment] = field(default_factory=list)
    reason: str = ""


@dataclass
class Plan:
    categories: list[Category]
    assignments: list[Assignment]

    def category_by_id(self, cid: str) -> Optional[Category]:
        for c in self.categories:
            if c.id == cid:
                return c
        return None


@dataclass
class MovedFile:
    original_path: Path
    new_path: Path
    category_id: str
    reason: str = ""
    score: float = 0.0
    shortcuts: list[Path] = field(default_factory=list)
    # First-page text excerpt collected by the parser; surfaced into the
    # search index so the user can find files by what's *inside* them,
    # not just by name.
    content_excerpt: str = ""


@dataclass
class SkippedFile:
    path: Path
    reason: str


@dataclass
class LLMCall:
    """One LLM HTTP round-trip with timing breakdown."""
    label: str = ""
    prompt_chars: int = 0
    response_chars: int = 0
    duration_s: float = 0.0       # total wall-clock
    ttft_s: float = 0.0           # time to first token (streaming only; 0 otherwise)
    success: bool = True
    error: str = ""

    @property
    def tokens_per_second(self) -> float:
        if self.duration_s <= 0:
            return 0.0
        return (self.response_chars / 3) / self.duration_s


@dataclass
class LLMUsage:
    request_count: int = 0
    prompt_chars: int = 0
    response_chars: int = 0
    model: str = ""
    total_duration_s: float = 0.0
    calls: list[LLMCall] = field(default_factory=list)

    def avg_tokens_per_second(self) -> float:
        if self.total_duration_s <= 0:
            return 0.0
        return (self.response_chars / 3) / self.total_duration_s

    def avg_ttft_s(self) -> float:
        ttfts = [c.ttft_s for c in self.calls if c.ttft_s > 0]
        if not ttfts:
            return 0.0
        return sum(ttfts) / len(ttfts)

    @property
    def estimated_prompt_tokens(self) -> int:
        # Heuristic: ~3 characters per token for mixed Korean/English.
        return self.prompt_chars // 3

    @property
    def estimated_response_tokens(self) -> int:
        return self.response_chars // 3

    def estimate_cost_usd(self) -> float:
        """Rough USD cost estimate based on public model pricing.

        Numbers are *approximate* and meant as a back-of-envelope figure
        for the report — the actual bill depends on the live provider
        pricing tier at request time.  Prices below are USD per 1M tokens.
        Local models (Ollama / vLLM / LM Studio) report 0.
        """
        m = (self.model or "").lower()
        # (input_per_1m, output_per_1m)
        pricing = {
            "gemini-2.5-flash":      (0.30, 2.50),
            "gemini-2.5-flash-lite": (0.10, 0.40),
            "gemini-2.5-pro":        (1.25, 10.00),
            "gemini-1.5-flash":      (0.075, 0.30),
            "gemini-1.5-pro":        (1.25, 5.00),
            "gpt-4o-mini":           (0.15, 0.60),
            "gpt-4o":                (2.50, 10.00),
            "gpt-4.1-mini":          (0.40, 1.60),
            "gpt-4.1":               (2.00, 8.00),
            "claude-3-5-sonnet":     (3.00, 15.00),
            "claude-3-5-haiku":      (0.80, 4.00),
            "qwen2.5-72b-instruct":  (0.0, 0.0),
        }
        # Locally-hosted models: assume free.
        local_hint = any(
            tag in m for tag in ("qwen", "llama", "phi", "gemma", "mistral", "mixtral", "deepseek")
        )
        in_rate, out_rate = pricing.get(m, (0.0, 0.0) if local_hint else (0.30, 2.50))
        in_tokens = self.estimated_prompt_tokens
        out_tokens = self.estimated_response_tokens
        return (in_tokens / 1_000_000) * in_rate + (out_tokens / 1_000_000) * out_rate

    def estimate_cost_krw(self, usd_to_krw: float = 1380.0) -> float:
        return self.estimate_cost_usd() * usd_to_krw


@dataclass
class OperationResult:
    target_root: Path
    started_at: datetime
    finished_at: datetime
    dry_run: bool
    categories: list[Category]
    moved: list[MovedFile]
    skipped: list[SkippedFile]
    total_scanned: int
    operation_id: Optional[int] = None
    llm_usage: Optional[LLMUsage] = None
    report_path: Optional[Path] = None
    # Duplicates removed during this run: list of (deleted_path,
    # canonical_path, bytes_freed) — surfaced in the report so the
    # user knows what disappeared and why.
    dupes_removed: list = field(default_factory=list)
    bytes_freed: int = 0

    @property
    def total_moved(self) -> int:
        return len(self.moved)

    @property
    def total_shortcuts(self) -> int:
        return sum(len(m.shortcuts) for m in self.moved)

    @property
    def total_skipped(self) -> int:
        return len(self.skipped)
