"""Cross-platform shortcut creation.

On Linux/macOS: symbolic link.
On Windows: a ``.lnk`` Windows Shell link if ``pywin32`` (or PowerShell) is
available, else a ``.url`` file as a last-resort fallback.
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

log = logging.getLogger(__name__)


def _unique(path: Path) -> Path:
    if not path.exists() and not path.is_symlink():
        return path
    stem = path.stem
    suffix = path.suffix
    parent = path.parent
    idx = 2
    while True:
        candidate = parent / f"{stem} ({idx}){suffix}"
        if not candidate.exists() and not candidate.is_symlink():
            return candidate
        idx += 1


def create_shortcut(target: Path, link_dir: Path, base_name: str | None = None) -> Path:
    """Create a shortcut inside ``link_dir`` that points to ``target``.

    Returns the absolute path of the created shortcut.  On success the caller
    should not assume a particular file type: on Linux this will be a symlink
    named after the original file, on Windows a ``.lnk`` (or ``.url`` on the
    fallback path).
    """
    target = Path(target).resolve()
    link_dir = Path(link_dir)
    link_dir.mkdir(parents=True, exist_ok=True)
    base = base_name or target.name

    if sys.platform.startswith("win"):
        lnk = _unique(link_dir / f"{base}.lnk")
        ok = _create_lnk(target, lnk)
        if ok:
            return lnk
        # Final fallback — a .url file works in Explorer, less ideal for folders
        url_file = _unique(link_dir / f"{base}.url")
        url_file.write_text(
            "[InternetShortcut]\nURL=file:///{}\n".format(str(target).replace("\\", "/")),
            encoding="utf-8",
        )
        return url_file

    # POSIX: symlink with the original filename preserved.
    link_path = _unique(link_dir / base)
    try:
        os.symlink(target, link_path)
    except OSError as exc:
        log.warning("symlink failed (%s), falling back to copy", exc)
        shutil.copy2(target, link_path)
    return link_path


# ---------------- Windows helpers ----------------

def _create_lnk(target: Path, lnk: Path) -> bool:
    """Try pywin32 first, then PowerShell.  Returns True if successful."""
    try:
        import pythoncom  # type: ignore
        from win32com.client import Dispatch  # type: ignore

        shell = Dispatch("WScript.Shell")
        shortcut = shell.CreateShortcut(str(lnk))
        shortcut.Targetpath = str(target)
        shortcut.WorkingDirectory = str(target.parent)
        shortcut.IconLocation = str(target)
        shortcut.save()
        return True
    except Exception as exc:
        log.debug("pywin32 lnk creation failed: %s", exc)

    # PowerShell fallback
    ps_cmd = (
        "$WshShell = New-Object -ComObject WScript.Shell; "
        f"$s = $WshShell.CreateShortcut('{lnk}'); "
        f"$s.TargetPath = '{target}'; "
        f"$s.WorkingDirectory = '{target.parent}'; "
        "$s.Save()"
    )
    try:
        subprocess.run(
            ["powershell", "-NoProfile", "-Command", ps_cmd],
            check=True,
            timeout=15,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return True
    except Exception as exc:  # pragma: no cover
        log.warning("powershell lnk fallback failed: %s", exc)
        return False
