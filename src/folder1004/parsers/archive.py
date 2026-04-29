"""Archive-content parser.

Strategy: do NOT extract anything.  Just list the archive's member
file names + extensions and emit them as a synthetic body excerpt.
The classifier treats this list like the document body of a
"manifest" file — much better than an empty excerpt because the
member names usually carry the project / contract / report identity
("RTX_GPU_3대_구매계약_세금계산서_*.pdf", "행안부_제안서_v1.hwp",
…), and that identity is exactly what drives folder placement.

Supported containers:
    .zip        — Python stdlib :mod:`zipfile`
    .tar / .tar.gz / .tgz / .tar.bz2 / .tbz / .tar.xz / .txz
                 — Python stdlib :mod:`tarfile`
    .jar / .war — same as zip

7z and rar would need third-party libs (``py7zr``, ``rarfile``); not
included here, but adding them is just a new branch in :func:`parse`.
"""
from __future__ import annotations

import logging
import tarfile
import zipfile
from pathlib import Path

log = logging.getLogger(__name__)


_TAR_EXTS = (
    ".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tbz",
    ".tar.xz", ".txz",
)
_ZIP_EXTS = (".zip", ".jar", ".war")


def is_archive(path: Path) -> bool:
    name = path.name.lower()
    return name.endswith(_ZIP_EXTS) or name.endswith(_TAR_EXTS)


def _format_listing(archive_name: str, names: list[str], max_chars: int) -> str:
    """Render the member-name list as a body-style excerpt."""
    if not names:
        return f"[archive: {archive_name}] (비어 있음)"
    # Drop folder-only entries ("dir/"); keep the actual file names.
    files = [n for n in names if n and not n.endswith("/")]
    if not files:
        return f"[archive: {archive_name}] (디렉토리만 {len(names)}개)"
    # The most identity-bearing tokens tend to be near the top of the
    # listing (root-level files first), so we keep arrival order — no
    # alphabetical sort.  Truncate to fit max_chars.
    header = f"[archive: {archive_name} — {len(files)}개 파일]\n"
    body_parts = []
    used = len(header)
    for n in files:
        candidate = n + ", "
        if used + len(candidate) > max_chars:
            body_parts.append("…")
            break
        body_parts.append(candidate)
        used += len(candidate)
    return header + "".join(body_parts).rstrip(", ")


def parse(path: Path, max_chars: int) -> str:
    """Return up to ``max_chars`` of the archive's member-name listing.

    Never extracts file contents — pure name+extension scan.  Returns
    "" only on hard failure (corrupt archive, unsupported variant).
    """
    name = path.name
    name_lc = name.lower()
    try:
        if name_lc.endswith(_ZIP_EXTS):
            with zipfile.ZipFile(str(path), "r") as zf:
                # ZipInfo.filename is the archive-internal path.
                names = [zi.filename for zi in zf.infolist()]
            return _format_listing(name, names, max_chars)
        if name_lc.endswith(_TAR_EXTS):
            # tarfile auto-detects the compression suffix.
            with tarfile.open(str(path), "r:*") as tf:
                names = [m.name for m in tf.getmembers()]
            return _format_listing(name, names, max_chars)
    except (zipfile.BadZipFile, tarfile.TarError, OSError) as exc:
        log.warning("archive parse failed for %s: %s", path, exc)
        return ""
    except Exception as exc:  # pragma: no cover — defensive
        log.warning("archive parse unexpected error for %s: %s", path, exc)
        return ""
    return ""
