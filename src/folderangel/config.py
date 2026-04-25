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

_KEYRING_SERVICE = "folderangel"
_KEYRING_USER = "gemini_api_key"


@dataclass
class AppPaths:
    root: Path
    config: Path
    index_db: Path
    logs_dir: Path

    def ensure(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)


def default_paths() -> AppPaths:
    """Pick a platform-appropriate data dir.

    Linux/macOS: ~/.folderangel
    Windows:     %APPDATA%/FolderAngel (falls back to home if APPDATA missing)
    """
    if sys.platform.startswith("win"):
        base = Path(os.environ.get("APPDATA") or Path.home()) / "FolderAngel"
    else:
        base = Path.home() / ".folderangel"
    return AppPaths(
        root=base,
        config=base / "config.json",
        index_db=base / "index.db",
        logs_dir=base / "logs",
    )


@dataclass
class Config:
    model: str = "gemini-2.5-flash"
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
    # API key is stored separately (keyring) but mirrored here only if keyring fails
    api_key_fallback: str = ""
    ignore_patterns: list[str] = field(
        default_factory=lambda: [".*", "~$*", "Thumbs.db", ".DS_Store", "desktop.ini"]
    )

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


def get_api_key(cfg: Optional[Config] = None) -> Optional[str]:
    """Resolve API key by priority: env → keyring → config fallback.

    Env vars checked: GEMINI_API_KEY, GOOGLE_API_KEY.
    """
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if key:
        return key.strip()
    kr = _try_keyring()
    if kr is not None:
        try:
            value = kr.get_password(_KEYRING_SERVICE, _KEYRING_USER)
            if value:
                return value.strip()
        except Exception as exc:
            log.warning("keyring read failed: %s", exc)
    if cfg and cfg.api_key_fallback:
        return cfg.api_key_fallback.strip()
    return None


def set_api_key(key: str, cfg: Optional[Config] = None, paths: Optional[AppPaths] = None) -> bool:
    """Persist API key. Returns True if stored securely (keyring), False if config fallback."""
    key = (key or "").strip()
    kr = _try_keyring()
    if kr is not None:
        try:
            kr.set_password(_KEYRING_SERVICE, _KEYRING_USER, key)
            # clear fallback if we had one
            if cfg is not None and cfg.api_key_fallback:
                cfg.api_key_fallback = ""
                save_config(cfg, paths)
            return True
        except Exception as exc:
            log.warning("keyring write failed: %s", exc)
    # Fallback to config file (clearly marked)
    cfg = cfg or load_config(paths)
    cfg.api_key_fallback = key
    save_config(cfg, paths)
    return False
