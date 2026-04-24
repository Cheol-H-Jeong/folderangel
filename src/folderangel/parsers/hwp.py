"""HWP / HWPX parsers.

HWPX is an XML-in-zip format; we extract section text XML and strip tags.
Binary HWP 5.x: use olefile to read BodyText streams and heuristically recover
readable Korean/ASCII characters.  This is best-effort — if the stream is
compressed (HWP_VERSION w/ compressed flag) we rely on zlib.
"""
from __future__ import annotations

import logging
import re
import struct
import zipfile
import zlib
from pathlib import Path
from xml.etree import ElementTree as ET

log = logging.getLogger(__name__)


def parse_hwpx(path: Path, max_chars: int) -> str:
    try:
        zf = zipfile.ZipFile(path)
    except zipfile.BadZipFile as exc:
        log.warning("hwpx open failed %s: %s", path, exc)
        return ""
    chunks: list[str] = []
    total = 0
    try:
        section_names = sorted(
            n for n in zf.namelist() if n.startswith("Contents/section") and n.endswith(".xml")
        )
        if not section_names:
            section_names = sorted(
                n for n in zf.namelist() if n.lower().endswith(".xml") and "section" in n.lower()
            )
        for name in section_names:
            try:
                with zf.open(name) as f:
                    xml = f.read()
            except Exception as exc:
                log.debug("hwpx read %s: %s", name, exc)
                continue
            try:
                root = ET.fromstring(xml)
            except ET.ParseError:
                continue
            for elem in root.iter():
                if elem.text and elem.text.strip():
                    text = elem.text.strip()
                    chunks.append(text)
                    total += len(text)
                    if total >= max_chars:
                        return "\n".join(chunks)[:max_chars]
    finally:
        zf.close()
    return "\n".join(chunks)[:max_chars]


# --- HWP 5.x binary -------------------------------------------------------

_PRINTABLE = re.compile(r"[\x20-\x7e가-힣ㄱ-ㆎ一-鿿\s·…\-\.,\(\)\[\]\{\}\"\'‘’“”!?:;#&@/]+")


def _decode_hwp_body(data: bytes) -> str:
    """Very loose body-text stream decoder.

    HWP 5 text records use a TLV format where text is UTF-16LE inside a record
    with tag 0x43 (PARA_TEXT).  Rather than implement the whole container, we
    sweep the buffer looking for UTF-16LE text runs, which works for cover
    pages (the first screen of a document) well enough to guide categorisation.
    """
    try:
        decoded = data.decode("utf-16-le", errors="ignore")
    except Exception:
        return ""
    # extract continuous printable runs
    runs = _PRINTABLE.findall(decoded)
    return "\n".join(r.strip() for r in runs if len(r.strip()) >= 2)


def _read_stream_safely(ole, name: str) -> bytes:
    try:
        with ole.openstream(name) as s:
            return s.read()
    except Exception as exc:  # pragma: no cover
        log.debug("hwp stream read failed %s: %s", name, exc)
        return b""


def parse_hwp(path: Path, max_chars: int) -> str:
    try:
        import olefile  # type: ignore
    except ImportError:
        log.warning("olefile not installed; skipping HWP %s", path)
        return ""
    try:
        if not olefile.isOleFile(str(path)):
            return ""
        ole = olefile.OleFileIO(str(path))
    except Exception as exc:
        log.warning("hwp open failed %s: %s", path, exc)
        return ""

    chunks: list[str] = []
    total = 0
    try:
        # Determine whether body streams are compressed.  FileHeader stream contains
        # a bit flag at byte offset 36 (bit 0 = compressed).
        compressed = True
        fh = _read_stream_safely(ole, "FileHeader")
        if len(fh) >= 40:
            try:
                flags = struct.unpack("<I", fh[36:40])[0]
                compressed = bool(flags & 0x01)
            except struct.error:
                pass

        body_streams = [
            entry
            for entry in ole.listdir(streams=True)
            if len(entry) >= 2 and entry[0] in ("BodyText", "ViewText")
        ]
        body_streams.sort()
        for entry in body_streams[:6]:
            raw = _read_stream_safely(ole, "/".join(entry))
            if not raw:
                continue
            if compressed:
                try:
                    raw = zlib.decompress(raw, -15)
                except zlib.error:
                    # Some HWPs mark compressed but particular streams differ; try raw.
                    pass
            text = _decode_hwp_body(raw)
            if text:
                chunks.append(text)
                total += len(text)
                if total >= max_chars:
                    break
    finally:
        try:
            ole.close()
        except Exception:
            pass
    joined = "\n".join(chunks)
    return joined[:max_chars]
