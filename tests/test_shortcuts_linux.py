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

    # Required keys for a launcher that opens the file rather than navigating
    assert "Type=Application" in text
    assert "Exec=xdg-open" in text
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
