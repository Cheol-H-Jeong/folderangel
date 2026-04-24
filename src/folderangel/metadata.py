"""File metadata collection — cross-platform."""
from __future__ import annotations

import logging
import mimetypes
import sys
from datetime import datetime, timezone
from pathlib import Path

from .models import FileEntry

log = logging.getLogger(__name__)


def _created_from_stat(st) -> datetime:
    """Best-effort "created" timestamp.

    Windows: st_ctime is creation time. POSIX: st_birthtime if present, else st_ctime.
    """
    if sys.platform.startswith("win"):
        ts = st.st_ctime
    else:
        ts = getattr(st, "st_birthtime", None) or st.st_ctime
    try:
        return datetime.fromtimestamp(ts, tz=timezone.utc).astimezone()
    except (OSError, OverflowError, ValueError):
        return datetime.now().astimezone()


def _ts_to_dt(ts: float) -> datetime:
    try:
        return datetime.fromtimestamp(ts, tz=timezone.utc).astimezone()
    except (OSError, OverflowError, ValueError):
        return datetime.now().astimezone()


def collect(path: Path) -> FileEntry:
    path = Path(path)
    st = path.stat()
    ext = path.suffix.lower()
    mime, _ = mimetypes.guess_type(str(path))
    return FileEntry(
        path=path,
        name=path.name,
        ext=ext,
        size=st.st_size,
        created=_created_from_stat(st),
        modified=_ts_to_dt(st.st_mtime),
        accessed=_ts_to_dt(st.st_atime),
        mime=mime or "",
    )
