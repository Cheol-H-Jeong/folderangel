"""UI instantiation smoke test under the offscreen Qt platform."""
import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")
from PySide6 import QtWidgets  # noqa: E402

from folderangel.config import default_paths, load_config  # noqa: E402
from folderangel.ui.main import MainWindow  # noqa: E402


def test_mainwindow_builds(tmp_path, monkeypatch):
    # Point XDG/HOME-like paths at tmp
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("APPDATA", str(tmp_path))
    paths = default_paths()
    paths.ensure()
    cfg = load_config(paths)

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    w = MainWindow(cfg, paths)
    w.show()
    app.processEvents()
    # Switch through each tab
    for idx in range(4):
        w._goto(idx)
        app.processEvents()
    w.close()
    w.index_db.close()
