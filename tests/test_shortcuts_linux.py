"""Linux-specific shortcut behaviour."""
import os
import stat
import sys
from pathlib import Path

import pytest

if not sys.platform.startswith("linux"):
    pytest.skip("linux-only", allow_module_level=True)

from folderangel.shortcuts import create_shortcut


def test_desktop_launcher_for_file(tmp_path):
    target = tmp_path / "report.pdf"
    target.write_text("dummy")
    link_dir = tmp_path / "secondary"
    link_dir.mkdir()

    sp = create_shortcut(target, link_dir)
    assert sp.exists()
    assert sp.suffix == ".desktop"
    text = sp.read_text(encoding="utf-8")

    # Either of the two valid layouts is acceptable.  We require the file
    # manager to be able to find the original target either via URL= or via
    # an Exec= line that includes the absolute target path.
    assert "Type=Link" in text or "Type=Application" in text
    assert ("URL=file://" in text) or ("Exec=" in text)
    assert str(target) in text
    assert "Name=report.pdf" in text

    # Must be executable so file managers treat it as a launcher
    mode = sp.stat().st_mode
    assert mode & stat.S_IXUSR


def test_unique_naming_avoids_overwrite(tmp_path):
    target = tmp_path / "doc.txt"
    target.write_text("x")
    link_dir = tmp_path / "sec"
    link_dir.mkdir()

    a = create_shortcut(target, link_dir)
    b = create_shortcut(target, link_dir)
    assert a.exists() and b.exists()
    assert a != b
