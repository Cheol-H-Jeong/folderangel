"""PDF excerpt extractor using pypdf."""
from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger(__name__)


def parse(path: Path, max_chars: int) -> str:
    try:
        from pypdf import PdfReader  # type: ignore
    except ImportError:
        log.warning("pypdf not installed; skipping PDF %s", path)
        return ""
    try:
        reader = PdfReader(str(path), strict=False)
    except Exception as exc:
        log.warning("pdf open failed %s: %s", path, exc)
        return ""
    chunks: list[str] = []
    total = 0
    for page in reader.pages[:5]:  # A4 ≈ 1 page, but collect up to 5 to reach 1800 chars safely
        try:
            txt = page.extract_text() or ""
        except Exception as exc:
            log.debug("pdf page extract failed: %s", exc)
            continue
        if not txt:
            continue
        chunks.append(txt)
        total += len(txt)
        if total >= max_chars:
            break
    joined = "\n".join(chunks)
    return joined[:max_chars]
