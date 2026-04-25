"""Linux-specific shortcut behaviour."""
import os
import stat
import sys
from pathlib import Path

import pytest

if not sys.platform.startswith("linux"):
    pytest.skip("linux-only", allow_module_level=True)

from folderangel.shortcuts import create_shortcut


def test_shortcut_is_indistinguishable_from_real_file(tmp_path):
    """Linux strategy is hardlink → symlink → .desktop fallback.

    For the common case (same filesystem) we must end up with a file
    that file managers double-click without any trust gate, i.e. a
    hardlink (preferred) or symlink.  Both share the same inode-data
    contract: reading the shortcut returns the original bytes.
    """
    target = tmp_path / "report.pdf"
    target.write_bytes(b"hello-fa")
    link_dir = tmp_path / "secondary"
    link_dir.mkdir()

    sp = create_shortcut(target, link_dir)
    assert sp.exists()

    # On any healthy single-fs Linux box this should be a hardlink (or
    # at worst a symlink), NOT a .desktop file — those need a per-file
    # "Allow Launching" toggle in modern GNOME and break double-click.
    assert sp.suffix != ".desktop", (
        f"on this environment we landed on a .desktop fallback ({sp}); "
        "GNOME Files refuses to launch those without explicit user trust, "
        "so the shortcut would not work on double-click"
    )
    # Bytes must match the original (works for hardlink and symlink).
    assert sp.read_bytes() == b"hello-fa"
    # And modifying the original is reflected — proves the link contract.
    target.write_bytes(b"hello-fa-v2")
    assert sp.read_bytes() == b"hello-fa-v2"


def test_unique_naming_avoids_overwrite(tmp_path):
    target = tmp_path / "doc.txt"
    target.write_text("x")
    link_dir = tmp_path / "sec"
    link_dir.mkdir()

    a = create_shortcut(target, link_dir)
    b = create_shortcut(target, link_dir)
    assert a.exists() and b.exists()
    assert a != b
