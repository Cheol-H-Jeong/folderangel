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


@dataclass
class SkippedFile:
    path: Path
    reason: str


@dataclass
class LLMUsage:
    request_count: int = 0
    prompt_chars: int = 0
    response_chars: int = 0
    model: str = ""

    @property
    def estimated_prompt_tokens(self) -> int:
        # Heuristic: ~3 characters per token for mixed Korean/English.
        return self.prompt_chars // 3

    @property
    def estimated_response_tokens(self) -> int:
        return self.response_chars // 3


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

    @property
    def total_moved(self) -> int:
        return len(self.moved)

    @property
    def total_shortcuts(self) -> int:
        return sum(len(m.shortcuts) for m in self.moved)

    @property
    def total_skipped(self) -> int:
        return len(self.skipped)
