"""UI instantiation smoke test under the offscreen Qt platform."""
import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")
from PySide6 import QtWidgets  # noqa: E402

from folderangel.config import default_paths, load_config  # noqa: E402
from folderangel.ui.main import MainWindow  # noqa: E402


def test_live_status_collapses_heartbeat_and_token_stream(tmp_path, monkeypatch):
    """Heartbeat ("…N s 경과") and token-stream ("토큰 수신") lines for
    the same planning stage must overwrite each other on the same row,
    not pile up one new row per second.  Stage transitions still
    append a fresh row.
    """
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("APPDATA", str(tmp_path))
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    from PySide6 import QtWidgets
    from folderangel.config import default_paths, load_config
    from folderangel.ui.views import OrganizeView

    paths = default_paths(); paths.ensure()
    cfg = load_config(paths)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    v = OrganizeView(cfg); v.set_running(True)

    seq = [
        "plan: LLM 호출 중 (5 파일)…",
        "plan: LLM 응답 대기 중 (5 파일) … 0s 경과",
        "plan: LLM 응답 대기 중 (5 파일) … 1s 경과",
        "plan 토큰 수신 (5 파일): 12자 수신 중 — …",
        "plan: LLM 응답 대기 중 (5 파일) … 2s 경과",
        "plan 토큰 수신 (5 파일): 48자 수신 중 — …",
        "plan 토큰 수신 (5 파일): 96자 수신 중 — …",
        "plan: 응답 수신 — 카테고리 1",
        "organize: 파일 이동 시작",
    ]
    for line in seq:
        v.on_status(line)

    text = v.log_view.toPlainText()
    rows = [l for l in text.splitlines() if l.strip()]
    # 9 status events arrived but 6 of them are heartbeat / token-stream
    # for the same plan stage and must collapse onto a single in-place
    # row.  The other 3 rows are real stage transitions
    # (호출 중, 응답 수신, organize 시작).  So we expect ≤ 4 rows total.
    assert len(rows) <= 4, f"too many rows ({len(rows)}); rows={rows}"
    # The collapsed plan-stream row must show the latest progress.
    assert any("96자" in l for l in rows)
    # And no row counts the heartbeat-second appearing twice.
    assert sum(1 for l in rows if "1s 경과" in l) == 0
    assert sum(1 for l in rows if "2s 경과" in l) == 0


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
