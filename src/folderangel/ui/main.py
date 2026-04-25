"""Main application window — Apple-inspired side-nav + pages layout."""
from __future__ import annotations

import logging
import sys
from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets

from ..config import Config, default_paths, load_config
from ..index import IndexDB
from ..worker import OrganizeWorker
from .styles import resolve_qss
from .views import HistoryView, OrganizeView, SearchView, SettingsView
from .widgets import NavButton

log = logging.getLogger(__name__)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, config: Config, paths):
        super().__init__()
        self.config = config
        self.paths = paths
        self.index_db = IndexDB(paths.index_db)
        self.setWindowTitle("FolderAngel")
        self.resize(1180, 760)

        self._thread: QtCore.QThread | None = None
        self._worker: OrganizeWorker | None = None

        self._build()
        self._apply_style()

    # ------------------------------------------------------------------
    def _build(self):
        root = QtWidgets.QWidget()
        root.setObjectName("MainRoot")
        self.setCentralWidget(root)
        row = QtWidgets.QHBoxLayout(root)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(0)

        # Sidebar ------------------------------------------------------
        sidebar = QtWidgets.QFrame()
        sidebar.setObjectName("Sidebar")
        sidebar.setFixedWidth(220)
        sb = QtWidgets.QVBoxLayout(sidebar)
        sb.setContentsMargins(16, 18, 16, 16)
        sb.setSpacing(6)

        logo = QtWidgets.QLabel("FolderAngel")
        logo.setStyleSheet("font-size:19px;font-weight:700;padding:6px 10px 18px 10px;")
        sb.addWidget(logo)

        self.nav_buttons: list[NavButton] = []
        for idx, (key, name, icon) in enumerate(
            [
                ("organize", "정리", "✨"),
                ("search", "검색", "🔎"),
                ("history", "히스토리", "🕒"),
                ("settings", "설정", "⚙"),
            ]
        ):
            btn = NavButton(name, icon)
            btn.clicked.connect(lambda _=False, i=idx: self._goto(i))
            sb.addWidget(btn)
            self.nav_buttons.append(btn)

        sb.addStretch(1)
        self.status_badge = QtWidgets.QLabel()
        self.status_badge.setWordWrap(True)
        self.status_badge.setStyleSheet("color:#6e6e73;font-size:12px;padding:4px 10px;")
        sb.addWidget(self.status_badge)

        row.addWidget(sidebar)

        # Pages --------------------------------------------------------
        self.stack = QtWidgets.QStackedWidget()
        self.organize_view = OrganizeView(self.config)
        self.search_view = SearchView(self.index_db)
        self.history_view = HistoryView(self.index_db)
        self.settings_view = SettingsView(self.config)

        self.stack.addWidget(self.organize_view)
        self.stack.addWidget(self.search_view)
        self.stack.addWidget(self.history_view)
        self.stack.addWidget(self.settings_view)
        row.addWidget(self.stack, 1)

        self.organize_view.start_requested.connect(self._start)
        self.organize_view.cancel_requested.connect(self._cancel)
        self.organize_view.rollback_requested.connect(self._rollback)
        self.history_view.rollback_requested.connect(self._rollback)
        self.settings_view.config_changed.connect(self._on_config_changed)

        # Menu / shortcuts --------------------------------------------
        QtGui.QShortcut(QtGui.QKeySequence.Find, self, self._focus_search)
        QtGui.QShortcut(QtGui.QKeySequence("Ctrl+,"), self, lambda: self._goto(3))
        QtGui.QShortcut(QtGui.QKeySequence("Ctrl+1"), self, lambda: self._goto(0))
        QtGui.QShortcut(QtGui.QKeySequence("Ctrl+2"), self, lambda: self._goto(1))
        QtGui.QShortcut(QtGui.QKeySequence("Ctrl+3"), self, lambda: self._goto(2))

        self._goto(0)
        self._update_status_badge()

    def _apply_style(self):
        self.setStyleSheet(resolve_qss(self.config.appearance))

    def _goto(self, idx: int):
        self.stack.setCurrentIndex(idx)
        for i, btn in enumerate(self.nav_buttons):
            btn.setChecked(i == idx)
        if idx == 1:
            self.search_view.focus_search()
        elif idx == 2:
            self.history_view.refresh()

    def _focus_search(self):
        self._goto(1)

    def _update_status_badge(self):
        from ..config import get_api_key, provider_label

        key = get_api_key(self.config)
        if key:
            self.status_badge.setText(
                f"{provider_label(self.config)}: 연결됨\n모델: {self.config.model}"
            )
        else:
            self.status_badge.setText("Mock 모드\n설정에서 API 키 등록")

    def _on_config_changed(self):
        self._apply_style()
        self.organize_view.refresh_api_badge()
        self._update_status_badge()

    # ------------------------------------------------------------------
    def _start(self, path: str, recursive: bool, dry_run: bool):
        if self._thread is not None:
            return
        # Open a fresh per-run log file so every Organize run is captured
        # with full INFO/DEBUG and tracebacks.
        from ..runlog import start_session

        try:
            start_session("organize")
        except Exception:
            pass
        self._thread = QtCore.QThread()
        self._worker = OrganizeWorker(
            Path(path), self.config, recursive, dry_run, self.index_db
        )
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.stage_changed.connect(self.organize_view.on_stage)
        self._worker.status.connect(self.organize_view.on_status)
        self._worker.finished.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._thread.start()

    def _cancel(self):
        if self._worker:
            self._worker.cancel()
        # Update the UI immediately so the user gets an instant
        # acknowledgement, regardless of how quickly the worker thread
        # actually unwinds.
        if hasattr(self, "organize_view"):
            self.organize_view.show_canceling()
        # Give the worker a brief grace window to stop on its own
        # (next safe checkpoint), then forcibly tear it down so the UI
        # never appears stuck.
        QtCore.QTimer.singleShot(800, self._force_teardown_after_cancel)

    def _force_teardown_after_cancel(self):
        if self._thread is None:
            return
        if self._thread.isRunning():
            # Last-resort: ask the event loop to quit and, if that does
            # not return, terminate the OS thread.  The worker holds no
            # locks on the main GUI state, so this is safe.
            self._thread.quit()
            if not self._thread.wait(400):
                try:
                    self._thread.terminate()
                    self._thread.wait(400)
                except Exception:
                    pass
        self._teardown_worker()
        self.organize_view.show_canceled()

    def _on_finished(self, op):
        self.organize_view.on_finished(op)
        self._teardown_worker()
        # refresh history list since we added a record
        if self.stack.currentWidget() is self.history_view:
            self.history_view.refresh()

    def _on_failed(self, msg: str):
        self.organize_view.on_failed(msg)
        self._teardown_worker()

    def _teardown_worker(self):
        if self._thread:
            self._thread.quit()
            self._thread.wait(2000)
        self._thread = None
        self._worker = None

    def _rollback(self, op_id: int):
        # Negative op_id is the UI's convention for "force this older op".
        force = op_id < 0
        real_id = -op_id if force else op_id
        res = self.index_db.rollback(real_id, force=force)
        msg = f"복원: {res.restored}개"
        if res.failed:
            msg += f"\n실패/건너뜀: {len(res.failed)}개\n" + "\n".join(res.failed[:5])
        QtWidgets.QMessageBox.information(self, "롤백 완료", msg)
        self.history_view.refresh()

    def closeEvent(self, event: QtGui.QCloseEvent):
        try:
            self._teardown_worker()
        finally:
            self.index_db.close()
        super().closeEvent(event)


def launch(argv: list[str] | None = None) -> int:
    argv = argv or sys.argv
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    from ..runlog import start_session

    try:
        start_session("gui")
    except Exception:
        pass

    QtWidgets.QApplication.setHighDpiScaleFactorRoundingPolicy(
        QtCore.Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    app = QtWidgets.QApplication(argv)
    app.setApplicationName("FolderAngel")
    app.setOrganizationName("FolderAngel")

    paths = default_paths()
    config = load_config(paths)
    window = MainWindow(config, paths)
    window.show()
    return app.exec()
