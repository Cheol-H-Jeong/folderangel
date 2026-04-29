"""Reusable UI widgets."""
from __future__ import annotations

from PySide6 import QtCore, QtGui, QtWidgets


class Card(QtWidgets.QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setProperty("class", "Card")
        # Explicit object class workaround so QSS `.Card` selector works.
        self.setObjectName("Card")
        self.setStyleSheet("")


class NavButton(QtWidgets.QPushButton):
    def __init__(self, text: str, icon_char: str = "•", parent=None):
        super().__init__(f"{icon_char}   {text}", parent)
        self.setObjectName("NavItem")
        self.setCheckable(True)
        self.setCursor(QtCore.Qt.PointingHandCursor)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed
        )


class StageIndicator(QtWidgets.QWidget):
    """Horizontal ``Scan → Parse → Plan → Organize`` stage strip."""

    STAGES = [("scan", "스캔"), ("parse", "파싱"), ("plan", "분류 계획"), ("organize", "정리")]

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        self._labels: dict[str, QtWidgets.QLabel] = {}
        for idx, (key, name) in enumerate(self.STAGES):
            pill = QtWidgets.QLabel(name)
            pill.setAlignment(QtCore.Qt.AlignCenter)
            pill.setMinimumHeight(28)
            pill.setMinimumWidth(90)
            pill.setStyleSheet(
                "background:#e5e5ea;color:#6e6e73;border-radius:14px;padding:4px 10px;font-weight:600;"
            )
            layout.addWidget(pill)
            self._labels[key] = pill
            if idx < len(self.STAGES) - 1:
                arrow = QtWidgets.QLabel("›")
                arrow.setStyleSheet("color:#bcbcc2;font-size:18px;")
                layout.addWidget(arrow)
        layout.addStretch(1)

    def set_active(self, stage_key: str):
        for key, label in self._labels.items():
            if key == stage_key:
                label.setStyleSheet(
                    "background:#0071e3;color:#ffffff;border-radius:14px;padding:4px 10px;font-weight:600;"
                )
            else:
                label.setStyleSheet(
                    "background:#e5e5ea;color:#6e6e73;border-radius:14px;padding:4px 10px;font-weight:600;"
                )

    def reset(self):
        for label in self._labels.values():
            label.setStyleSheet(
                "background:#e5e5ea;color:#6e6e73;border-radius:14px;padding:4px 10px;font-weight:600;"
            )


class PathDropBar(QtWidgets.QFrame):
    """Path bar with a Browse button and drag-and-drop support for folders."""

    path_changed = QtCore.Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("Card")
        self.setAcceptDrops(True)
        row = QtWidgets.QHBoxLayout(self)
        row.setContentsMargins(14, 12, 14, 12)
        row.setSpacing(10)

        self._label = QtWidgets.QLabel("폴더를 선택하거나 여기에 드롭하세요")
        self._label.setStyleSheet("color:#6e6e73;")

        self._btn_browse = QtWidgets.QPushButton("폴더 선택…")
        self._btn_browse.clicked.connect(self._browse)

        row.addWidget(self._label, 1)
        row.addWidget(self._btn_browse)

        self._path: str = ""

    def path(self) -> str:
        return self._path

    def set_path(self, p: str):
        self._path = p
        self._label.setText(p or "폴더를 선택하거나 여기에 드롭하세요")
        self._label.setStyleSheet("color:#1d1d1f;" if p else "color:#6e6e73;")
        self.path_changed.emit(p)

    def _browse(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "폴더 선택", self._path or "")
        if d:
            self.set_path(d)

    # Drag and drop ---------------------------------------------------
    def dragEnterEvent(self, event: QtGui.QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QtGui.QDropEvent):
        for url in event.mimeData().urls():
            local = url.toLocalFile()
            if local:
                from pathlib import Path

                if Path(local).is_dir():
                    self.set_path(local)
                    break


class StatsRow(QtWidgets.QWidget):
    """Three-badge row used in the report card."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._labels: list[QtWidgets.QLabel] = []
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

    def update_items(self, items: list[tuple[str, str]]):
        layout: QtWidgets.QHBoxLayout = self.layout()
        while layout.count():
            child = layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        for title, value in items:
            w = QtWidgets.QFrame()
            w.setObjectName("Card")
            v = QtWidgets.QVBoxLayout(w)
            v.setContentsMargins(16, 12, 16, 12)
            v.setSpacing(2)
            big = QtWidgets.QLabel(value)
            big.setStyleSheet("font-size:22px;font-weight:700;")
            sub = QtWidgets.QLabel(title)
            sub.setStyleSheet("color:#6e6e73;")
            v.addWidget(big)
            v.addWidget(sub)
            layout.addWidget(w, 1)
