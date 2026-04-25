"""Cross-platform shortcut creation.

We want a shortcut that, when **double-clicked**, opens the original file
in its default application — never navigates the user into a folder, never
silently fails.  Implementation per OS:

- **Windows**: a ``.lnk`` Windows Shell link with ``Targetpath`` pointing
  at the file (pywin32 → PowerShell fallback → ``.url``).  Double-click
  in Explorer activates the file's default handler.
- **macOS**: a symbolic link to the file (Finder follows symlinks to
  files cleanly with the original opener).
- **Linux**: a ``.desktop`` launcher whose ``Exec=`` calls ``xdg-open``
  on the absolute target path.  ``Type=Application`` with the executable
  bit set means double-clicking it in GNOME Files / Nautilus / Dolphin
  fires the file's default handler instead of navigating into it.  We
  also fall back to a symlink only when desktop launchers cannot be
  written for some reason.
"""
from __future__ import annotations

import logging
import os
import shlex
import shutil
import stat
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

    The shortcut, when double-clicked in the OS file manager, opens the
    *original file* in its default application (it does not navigate to a
    folder).  Returns the absolute path of the created shortcut.
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
        # Final fallback — .url file points at the file URI.
        url_file = _unique(link_dir / f"{base}.url")
        url_file.write_text(
            "[InternetShortcut]\nURL=file:///{}\n".format(str(target).replace("\\", "/")),
            encoding="utf-8",
        )
        return url_file

    if sys.platform == "darwin":
        # Finder happily opens symlinks to files using the file's default app.
        link_path = _unique(link_dir / base)
        try:
            os.symlink(target, link_path)
            return link_path
        except OSError as exc:
            log.warning("macOS symlink failed (%s), falling back to copy", exc)
            shutil.copy2(target, link_path)
            return link_path

    # ---- Linux: prefer a .desktop launcher so double-click *opens* the file
    # via xdg-open instead of navigating into it (which is what some file
    # managers do with symlinks-to-files in their list view).
    desktop_path = _unique(link_dir / f"{base}.desktop")
    try:
        _write_desktop_file(desktop_path, target)
        return desktop_path
    except OSError as exc:
        log.warning(".desktop launcher failed (%s); falling back to symlink", exc)

    link_path = _unique(link_dir / base)
    try:
        os.symlink(target, link_path)
        return link_path
    except OSError as exc:
        log.warning("symlink failed (%s), falling back to copy", exc)
        shutil.copy2(target, link_path)
    return link_path


def _write_desktop_file(path: Path, target: Path) -> None:
    """Create a ``Type=Application`` .desktop launcher that opens ``target``.

    Linux file managers expect ``Exec=`` to be a literal command line.  We
    quote the absolute target path and pass it to ``xdg-open``, so the
    user's MIME default handler (Evince, LibreOffice, image viewer, etc.)
    handles the actual open.
    """
    quoted = shlex.quote(str(target))
    contents = (
        "[Desktop Entry]\n"
        "Type=Application\n"
        f"Name={target.name}\n"
        f"Comment=FolderAngel link to {target}\n"
        f"Exec=xdg-open {quoted}\n"
        f"Icon=text-x-generic\n"
        "Terminal=false\n"
        "NoDisplay=false\n"
        "Categories=Utility;\n"
    )
    path.write_text(contents, encoding="utf-8")
    # Mark executable so file managers treat it as a launcher and trust it
    # without the "Untrusted application launcher" prompt on GNOME.
    mode = path.stat().st_mode
    path.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    # Some file managers (Nautilus ≥ 43) additionally require the launcher
    # to be marked trusted via gio metadata.  Best-effort, ignore failures.
    try:
        subprocess.run(
            ["gio", "set", str(path), "metadata::trusted", "true"],
            check=False,
            timeout=3,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass


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
