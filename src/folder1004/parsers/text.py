"""Parsers for plain-text / RTF / HTML formats."""
from __future__ import annotations

import logging
import re
from html.parser import HTMLParser
from pathlib import Path

log = logging.getLogger(__name__)

_ENCODINGS = ("utf-8", "utf-8-sig", "cp949", "euc-kr", "latin-1")


def _read_text(path: Path, max_bytes: int) -> str:
    try:
        raw = path.read_bytes()[:max_bytes]
    except Exception as exc:
        log.warning("read failed %s: %s", path, exc)
        return ""
    for enc in _ENCODINGS:
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="ignore")


def parse_plain(path: Path, max_chars: int) -> str:
    text = _read_text(path, max_chars * 4)
    return text[:max_chars]


_RTF_TOKEN = re.compile(r"\\[a-zA-Z]+-?\d* ?|\\'[0-9a-fA-F]{2}|[{}]")


def parse_rtf(path: Path, max_chars: int) -> str:
    raw = _read_text(path, max_chars * 6)
    # Strip control words/groups, keep literal text.
    cleaned = _RTF_TOKEN.sub("", raw)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()[:max_chars]


class _HTMLText(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag, attrs):  # noqa: D401
        if tag in ("script", "style"):
            self._skip += 1

    def handle_endtag(self, tag):
        if tag in ("script", "style") and self._skip > 0:
            self._skip -= 1

    def handle_data(self, data):
        if self._skip == 0:
            txt = data.strip()
            if txt:
                self.parts.append(txt)


def parse_html(path: Path, max_chars: int) -> str:
    raw = _read_text(path, max_chars * 6)
    parser = _HTMLText()
    try:
        parser.feed(raw)
    except Exception as exc:
        log.warning("html parse failed %s: %s", path, exc)
    return "\n".join(parser.parts)[:max_chars]
