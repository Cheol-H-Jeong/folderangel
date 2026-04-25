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

    # ---- Linux: try a Type=Link .desktop entry first (file managers open
    # this with the user's MIME default handler exactly as if the file
    # itself was double-clicked), then fall back to a Type=Application
    # launcher with an Exec= line, then to a plain symlink.  We pick the
    # path that the runtime file manager actually accepts.
    desktop_path = _unique(link_dir / f"{base}.desktop")
    try:
        _write_desktop_link(desktop_path, target)
        return desktop_path
    except OSError as exc:
        log.warning("Type=Link .desktop failed (%s); trying Type=Application", exc)

    try:
        _write_desktop_application(desktop_path, target)
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


# ---------------- Linux helpers ----------------


def _mark_executable_and_trusted(path: Path) -> None:
    mode = path.stat().st_mode
    path.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    # Nautilus / GNOME Files 43+ require explicit metadata trust before they
    # will execute a launcher on double-click.  Both of these are best-effort.
    for cmd in (
        ["gio", "set", str(path), "metadata::trusted", "true"],
        ["gio", "set", str(path), "metadata::xfce-exe-checksum",
         _file_checksum_hex(path)],
    ):
        try:
            subprocess.run(
                cmd,
                check=False,
                timeout=3,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass


def _file_checksum_hex(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    try:
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return ""


def _write_desktop_link(path: Path, target: Path) -> None:
    """Create a ``Type=Link`` ``.desktop`` entry pointing at ``target``.

    File managers treat this as a link to a URI; double-click opens the
    underlying file with its default handler — same as opening the
    original file from its real location.
    """
    target_uri = "file://" + str(target).replace("'", "\\'")
    contents = (
        "[Desktop Entry]\n"
        "Version=1.0\n"
        "Type=Link\n"
        f"Name={target.name}\n"
        f"Comment=FolderAngel link to {target}\n"
        f"URL={target_uri}\n"
        f"Icon=text-x-generic\n"
    )
    path.write_text(contents, encoding="utf-8")
    _mark_executable_and_trusted(path)


def _write_desktop_application(path: Path, target: Path) -> None:
    """Create a ``Type=Application`` launcher whose Exec= opens ``target``.

    We pick a launcher command that is more robust than plain ``xdg-open``:
    ``gio open`` on GNOME, ``kioclient5/6 exec`` on KDE, ``xdg-open`` as
    final fallback.  The shell wrapper tries them in order so the first
    one available actually fires.
    """
    quoted = shlex.quote(str(target))
    # ``sh -c`` lets us try multiple openers without depending on the
    # specific desktop environment.
    exec_cmd = (
        "sh -c "
        + shlex.quote(
            f"command -v gio >/dev/null 2>&1 && exec gio open {quoted}; "
            f"command -v kioclient5 >/dev/null 2>&1 && exec kioclient5 exec {quoted}; "
            f"command -v kioclient6 >/dev/null 2>&1 && exec kioclient6 exec {quoted}; "
            f"exec xdg-open {quoted}"
        )
    )
    contents = (
        "[Desktop Entry]\n"
        "Version=1.0\n"
        "Type=Application\n"
        f"Name={target.name}\n"
        f"Comment=FolderAngel link to {target}\n"
        f"Exec={exec_cmd}\n"
        f"TryExec=xdg-open\n"
        f"Icon=text-x-generic\n"
        "Terminal=false\n"
        "NoDisplay=false\n"
        "Categories=Utility;\n"
    )
    path.write_text(contents, encoding="utf-8")
    _mark_executable_and_trusted(path)




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
