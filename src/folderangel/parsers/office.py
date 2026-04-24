"""Parsers for modern Office / ODT formats."""
from __future__ import annotations

import logging
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

log = logging.getLogger(__name__)


def _cap(chunks: list[str], max_chars: int) -> str:
    out = "\n".join(c for c in chunks if c)
    return out[:max_chars]


def parse_docx(path: Path, max_chars: int) -> str:
    try:
        from docx import Document  # type: ignore
    except ImportError:
        log.warning("python-docx not installed; skipping %s", path)
        return ""
    try:
        doc = Document(str(path))
    except Exception as exc:
        log.warning("docx open failed %s: %s", path, exc)
        return ""
    chunks: list[str] = []
    total = 0
    for p in doc.paragraphs:
        txt = (p.text or "").strip()
        if not txt:
            continue
        chunks.append(txt)
        total += len(txt)
        if total >= max_chars:
            break
    # Include the first table as a fallback if paragraphs were empty
    if total < 40:
        for tbl in doc.tables[:2]:
            for row in tbl.rows:
                row_txt = " | ".join((c.text or "").strip() for c in row.cells)
                if row_txt.strip():
                    chunks.append(row_txt)
                    total += len(row_txt)
                    if total >= max_chars:
                        break
            if total >= max_chars:
                break
    return _cap(chunks, max_chars)


def parse_pptx(path: Path, max_chars: int) -> str:
    try:
        from pptx import Presentation  # type: ignore
    except ImportError:
        log.warning("python-pptx not installed; skipping %s", path)
        return ""
    try:
        pres = Presentation(str(path))
    except Exception as exc:
        log.warning("pptx open failed %s: %s", path, exc)
        return ""
    chunks: list[str] = []
    total = 0
    for slide in pres.slides[:20]:
        for shape in slide.shapes:
            if not getattr(shape, "has_text_frame", False):
                continue
            for para in shape.text_frame.paragraphs:
                for run in para.runs:
                    txt = (run.text or "").strip()
                    if not txt:
                        continue
                    chunks.append(txt)
                    total += len(txt)
                    if total >= max_chars:
                        return _cap(chunks, max_chars)
    return _cap(chunks, max_chars)


def parse_xlsx(path: Path, max_chars: int) -> str:
    try:
        from openpyxl import load_workbook  # type: ignore
    except ImportError:
        log.warning("openpyxl not installed; skipping %s", path)
        return ""
    try:
        wb = load_workbook(str(path), read_only=True, data_only=True)
    except Exception as exc:
        log.warning("xlsx open failed %s: %s", path, exc)
        return ""
    chunks: list[str] = []
    total = 0
    for name in wb.sheetnames[:4]:
        chunks.append(f"# Sheet: {name}")
        ws = wb[name]
        for row in ws.iter_rows(max_row=15, values_only=True):
            cells = [str(c) for c in row if c is not None]
            if not cells:
                continue
            row_txt = " | ".join(cells)
            chunks.append(row_txt)
            total += len(row_txt)
            if total >= max_chars:
                return _cap(chunks, max_chars)
    return _cap(chunks, max_chars)


def parse_odt(path: Path, max_chars: int) -> str:
    try:
        with zipfile.ZipFile(path) as z:
            with z.open("content.xml") as f:
                xml = f.read()
    except (zipfile.BadZipFile, KeyError) as exc:
        log.warning("odt open failed %s: %s", path, exc)
        return ""
    try:
        root = ET.fromstring(xml)
    except ET.ParseError as exc:
        log.warning("odt xml parse failed %s: %s", path, exc)
        return ""
    texts: list[str] = []
    total = 0
    for elem in root.iter():
        if elem.text and elem.text.strip():
            texts.append(elem.text.strip())
            total += len(elem.text)
            if total >= max_chars:
                break
    return _cap(texts, max_chars)
