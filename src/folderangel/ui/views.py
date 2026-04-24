"""Organize / Search / History / Settings views."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets

from ..config import Config, get_api_key, save_config, set_api_key
from ..index import IndexDB
from ..models import OperationResult
from .widgets import Card, PathDropBar, StageIndicator, StatsRow


def _open_in_explorer(path: Path):
    try:
        if sys.platform.startswith("win"):
            os.startfile(str(path))  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(path)])
        else:
            subprocess.Popen(["xdg-open", str(path)])
    except Exception:
        pass


class OrganizeView(QtWidgets.QWidget):
    start_requested = QtCore.Signal(str, bool, bool)  # path, recursive, dry_run
    cancel_requested = QtCore.Signal()
    rollback_requested = QtCore.Signal(int)

    def __init__(self, config: Config, parent=None):
        super().__init__(parent)
        self.config = config
        self._last_op: OperationResult | None = None
        self._build()

    def _build(self):
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(28, 28, 28, 20)
        outer.setSpacing(18)

        title = QtWidgets.QLabel("폴더 정리")
        title.setObjectName("Title")
        subtitle = QtWidgets.QLabel("폴더를 고르면 파일을 읽고 의미에 맞게 자동 분류합니다.")
        subtitle.setObjectName("Subtitle")
        outer.addWidget(title)
        outer.addWidget(subtitle)

        self.path_bar = PathDropBar()
        outer.addWidget(self.path_bar)

        options = Card()
        opt_row = QtWidgets.QHBoxLayout(options)
        opt_row.setContentsMargins(18, 14, 18, 14)
        opt_row.setSpacing(20)

        self.chk_recursive = QtWidgets.QCheckBox("하위 폴더 포함")
        self.chk_recursive.setChecked(self.config.recursive_default)
        self.chk_dry = QtWidgets.QCheckBox("Dry-Run (미리보기만)")
        opt_row.addWidget(self.chk_recursive)
        opt_row.addWidget(self.chk_dry)
        opt_row.addStretch(1)

        self.badge_api = QtWidgets.QLabel("API 키 확인 중…")
        self.badge_api.setObjectName("Badge")
        opt_row.addWidget(self.badge_api)

        outer.addWidget(options)

        # Progress card
        self.progress_card = Card()
        pc = QtWidgets.QVBoxLayout(self.progress_card)
        pc.setContentsMargins(18, 16, 18, 16)
        pc.setSpacing(12)
        self.stage_ind = StageIndicator()
        pc.addWidget(self.stage_ind)
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        pc.addWidget(self.progress_bar)
        self.progress_label = QtWidgets.QLabel("대기 중")
        self.progress_label.setStyleSheet("color:#6e6e73;")
        pc.addWidget(self.progress_label)
        outer.addWidget(self.progress_card)

        # Action row
        actions = QtWidgets.QHBoxLayout()
        actions.setSpacing(10)
        self.btn_primary = QtWidgets.QPushButton("정리 시작")
        self.btn_primary.setObjectName("Primary")
        self.btn_primary.clicked.connect(self._on_start)
        self.btn_cancel = QtWidgets.QPushButton("취소")
        self.btn_cancel.setObjectName("Ghost")
        self.btn_cancel.setVisible(False)
        self.btn_cancel.clicked.connect(self.cancel_requested)
        actions.addStretch(1)
        actions.addWidget(self.btn_cancel)
        actions.addWidget(self.btn_primary)
        outer.addLayout(actions)

        # Report card
        self.report_card = Card()
        self.report_card.setVisible(False)
        rc = QtWidgets.QVBoxLayout(self.report_card)
        rc.setContentsMargins(18, 16, 18, 16)
        rc.setSpacing(14)

        top = QtWidgets.QHBoxLayout()
        rt = QtWidgets.QLabel("최근 정리 결과")
        rt.setStyleSheet("font-size:18px;font-weight:600;")
        top.addWidget(rt)
        top.addStretch(1)
        self.btn_open_folder = QtWidgets.QPushButton("폴더 열기")
        self.btn_open_folder.setObjectName("Ghost")
        self.btn_open_folder.clicked.connect(self._open_target)
        self.btn_open_report = QtWidgets.QPushButton("리포트 열기")
        self.btn_open_report.setObjectName("Ghost")
        self.btn_open_report.clicked.connect(self._open_report)
        self.btn_rollback = QtWidgets.QPushButton("되돌리기")
        self.btn_rollback.setObjectName("Warning")
        self.btn_rollback.clicked.connect(self._emit_rollback)
        top.addWidget(self.btn_open_folder)
        top.addWidget(self.btn_open_report)
        top.addWidget(self.btn_rollback)
        rc.addLayout(top)

        self.stats_row = StatsRow()
        rc.addWidget(self.stats_row)

        self.cat_table = QtWidgets.QTableWidget(0, 3)
        self.cat_table.setHorizontalHeaderLabels(["카테고리", "폴더명", "파일 수"])
        self.cat_table.horizontalHeader().setStretchLastSection(False)
        self.cat_table.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        self.cat_table.horizontalHeader().setSectionResizeMode(1, QtWidgets.QHeaderView.Stretch)
        self.cat_table.horizontalHeader().setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeToContents)
        self.cat_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.cat_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.cat_table.setAlternatingRowColors(True)
        rc.addWidget(self.cat_table, 1)

        outer.addWidget(self.report_card, 1)
        outer.addStretch(1)
        self.refresh_api_badge()

    # ------------------------------------------------------------------
    def refresh_api_badge(self):
        key = get_api_key(self.config)
        if key:
            self.badge_api.setText("Gemini 연결됨")
            self.badge_api.setObjectName("Badge")
        else:
            self.badge_api.setText("Mock 모드 (API 키 없음)")
            self.badge_api.setObjectName("BadgeWarn")
        self.badge_api.setStyle(self.badge_api.style())

    # ------------------------------------------------------------------
    def _on_start(self):
        path = self.path_bar.path()
        if not path:
            QtWidgets.QMessageBox.warning(self, "경로 필요", "먼저 정리할 폴더를 선택하세요.")
            return
        if not Path(path).is_dir():
            QtWidgets.QMessageBox.warning(self, "폴더 아님", "선택한 경로가 폴더가 아닙니다.")
            return
        self.set_running(True)
        self.start_requested.emit(path, self.chk_recursive.isChecked(), self.chk_dry.isChecked())

    def set_running(self, running: bool):
        self.btn_primary.setDisabled(running)
        self.btn_cancel.setVisible(running)
        if running:
            self.report_card.setVisible(False)
            self.progress_bar.setValue(0)
            self.stage_ind.reset()
            self.progress_label.setText("시작 중…")

    def on_stage(self, stage: str, pct: float):
        self.stage_ind.set_active(stage)
        self.progress_bar.setValue(max(0, min(100, int(pct * 100))))

    def on_status(self, text: str):
        self.progress_label.setText(text)

    def on_finished(self, op: OperationResult):
        self._last_op = op
        self.set_running(False)
        self.progress_bar.setValue(100)
        self.progress_label.setText(
            f"완료 — 이동 {op.total_moved}, 바로가기 {op.total_shortcuts}, 스킵 {op.total_skipped}"
        )
        self.stats_row.update_items(
            [
                ("스캔 파일", str(op.total_scanned)),
                ("이동", str(op.total_moved)),
                ("바로가기", str(op.total_shortcuts)),
                ("스킵", str(op.total_skipped)),
            ]
        )
        from collections import Counter

        counter = Counter(m.category_id for m in op.moved)
        self.cat_table.setRowCount(len(op.categories))
        for row, cat in enumerate(op.categories):
            self.cat_table.setItem(row, 0, QtWidgets.QTableWidgetItem(cat.id))
            self.cat_table.setItem(row, 1, QtWidgets.QTableWidgetItem(cat.name))
            self.cat_table.setItem(row, 2, QtWidgets.QTableWidgetItem(str(counter.get(cat.id, 0))))
        self.report_card.setVisible(True)
        self.btn_rollback.setEnabled(bool(op.operation_id) and not op.dry_run)

    def on_failed(self, msg: str):
        self.set_running(False)
        QtWidgets.QMessageBox.critical(self, "오류", msg)

    def _emit_rollback(self):
        if self._last_op and self._last_op.operation_id:
            self.rollback_requested.emit(self._last_op.operation_id)

    def _open_target(self):
        if self._last_op:
            _open_in_explorer(self._last_op.target_root)

    def _open_report(self):
        if self._last_op:
            # The reporter writes to target_root/FolderAngel_Report_*.md
            stamp = self._last_op.finished_at.strftime("%Y%m%d_%H%M%S")
            p = self._last_op.target_root / f"FolderAngel_Report_{stamp}.md"
            if p.exists():
                _open_in_explorer(p)


class SearchView(QtWidgets.QWidget):
    def __init__(self, index_db: IndexDB, parent=None):
        super().__init__(parent)
        self.index_db = index_db
        v = QtWidgets.QVBoxLayout(self)
        v.setContentsMargins(28, 28, 28, 20)
        v.setSpacing(14)
        t = QtWidgets.QLabel("검색")
        t.setObjectName("Title")
        sub = QtWidgets.QLabel("정리된 파일을 이름, 카테고리, 원본 경로로 찾습니다.")
        sub.setObjectName("Subtitle")
        v.addWidget(t)
        v.addWidget(sub)

        row = QtWidgets.QHBoxLayout()
        self.search = QtWidgets.QLineEdit()
        self.search.setPlaceholderText("예: 2025 계약, 보고서, receipt…")
        self.search.returnPressed.connect(self._do_search)
        btn = QtWidgets.QPushButton("검색")
        btn.clicked.connect(self._do_search)
        row.addWidget(self.search, 1)
        row.addWidget(btn)
        v.addLayout(row)

        self.table = QtWidgets.QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["파일", "카테고리", "현재 위치", "원본", "정리 시각"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Interactive)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.doubleClicked.connect(self._open_selected)
        self.table.setAlternatingRowColors(True)
        v.addWidget(self.table, 1)

    def focus_search(self):
        self.search.setFocus(QtCore.Qt.ShortcutFocusReason)
        self.search.selectAll()

    def _do_search(self):
        q = self.search.text().strip()
        hits = self.index_db.search(q, limit=200)
        self.table.setRowCount(len(hits))
        for row, h in enumerate(hits):
            self.table.setItem(row, 0, QtWidgets.QTableWidgetItem(Path(h.new_path).name))
            self.table.setItem(row, 1, QtWidgets.QTableWidgetItem(h.category))
            self.table.setItem(row, 2, QtWidgets.QTableWidgetItem(h.new_path))
            self.table.setItem(row, 3, QtWidgets.QTableWidgetItem(h.original_path))
            self.table.setItem(row, 4, QtWidgets.QTableWidgetItem(h.created_at))

    def _open_selected(self, idx: QtCore.QModelIndex):
        row = idx.row()
        item = self.table.item(row, 2)
        if item:
            p = Path(item.text())
            _open_in_explorer(p.parent if p.exists() else p)


class HistoryView(QtWidgets.QWidget):
    rollback_requested = QtCore.Signal(int)

    def __init__(self, index_db: IndexDB, parent=None):
        super().__init__(parent)
        self.index_db = index_db
        v = QtWidgets.QVBoxLayout(self)
        v.setContentsMargins(28, 28, 28, 20)
        v.setSpacing(14)
        t = QtWidgets.QLabel("히스토리")
        t.setObjectName("Title")
        sub = QtWidgets.QLabel("최근 정리 작업을 확인하고 원하면 되돌릴 수 있습니다.")
        sub.setObjectName("Subtitle")
        v.addWidget(t)
        v.addWidget(sub)

        self.table = QtWidgets.QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["ID", "대상 폴더", "시작", "파일 수", "모드"])
        self.table.horizontalHeader().setSectionResizeMode(1, QtWidgets.QHeaderView.Stretch)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        v.addWidget(self.table, 1)

        row = QtWidgets.QHBoxLayout()
        row.addStretch(1)
        btn_refresh = QtWidgets.QPushButton("새로고침")
        btn_refresh.setObjectName("Ghost")
        btn_refresh.clicked.connect(self.refresh)
        btn_rb = QtWidgets.QPushButton("선택 롤백")
        btn_rb.setObjectName("Warning")
        btn_rb.clicked.connect(self._rollback_selected)
        row.addWidget(btn_refresh)
        row.addWidget(btn_rb)
        v.addLayout(row)

    def refresh(self):
        ops = self.index_db.list_operations(limit=100)
        self.table.setRowCount(len(ops))
        for row, op in enumerate(ops):
            self.table.setItem(row, 0, QtWidgets.QTableWidgetItem(str(op.op_id)))
            self.table.setItem(row, 1, QtWidgets.QTableWidgetItem(op.target_root))
            self.table.setItem(row, 2, QtWidgets.QTableWidgetItem(op.started_at))
            self.table.setItem(row, 3, QtWidgets.QTableWidgetItem(str(op.moved_count)))
            self.table.setItem(row, 4, QtWidgets.QTableWidgetItem("Dry" if op.dry_run else "실행"))

    def _rollback_selected(self):
        idxs = self.table.selectionModel().selectedRows()
        if not idxs:
            return
        row = idxs[0].row()
        op_id = int(self.table.item(row, 0).text())
        resp = QtWidgets.QMessageBox.question(
            self,
            "롤백",
            f"오퍼레이션 #{op_id}를 되돌리시겠습니까?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
        )
        if resp == QtWidgets.QMessageBox.Yes:
            self.rollback_requested.emit(op_id)


class SettingsView(QtWidgets.QWidget):
    config_changed = QtCore.Signal()

    def __init__(self, config: Config, parent=None):
        super().__init__(parent)
        self.config = config
        v = QtWidgets.QVBoxLayout(self)
        v.setContentsMargins(28, 28, 28, 20)
        v.setSpacing(14)

        t = QtWidgets.QLabel("설정")
        t.setObjectName("Title")
        sub = QtWidgets.QLabel("API 키와 분류 동작을 조정합니다.")
        sub.setObjectName("Subtitle")
        v.addWidget(t)
        v.addWidget(sub)

        # API key card
        api_card = Card()
        f = QtWidgets.QFormLayout(api_card)
        f.setContentsMargins(18, 14, 18, 18)
        f.setSpacing(10)
        self.edit_key = QtWidgets.QLineEdit()
        self.edit_key.setEchoMode(QtWidgets.QLineEdit.Password)
        self.edit_key.setPlaceholderText("sk-… or AIzaSy… (비워두면 Mock 모드)")
        if get_api_key(self.config):
            self.edit_key.setPlaceholderText("저장된 키 사용 중 — 덮어쓰려면 입력")
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.edit_key, 1)
        btn_save_key = QtWidgets.QPushButton("저장")
        btn_save_key.setObjectName("Primary")
        btn_save_key.clicked.connect(self._save_key)
        btn_clear_key = QtWidgets.QPushButton("삭제")
        btn_clear_key.setObjectName("Ghost")
        btn_clear_key.clicked.connect(self._clear_key)
        row.addWidget(btn_save_key)
        row.addWidget(btn_clear_key)
        wrap = QtWidgets.QWidget()
        wrap.setLayout(row)
        f.addRow("Gemini API 키", wrap)

        self.cmb_model = QtWidgets.QComboBox()
        self.cmb_model.addItems(["gemini-2.5-flash", "gemini-2.5-pro", "gemini-1.5-flash"])
        idx = self.cmb_model.findText(self.config.model)
        if idx >= 0:
            self.cmb_model.setCurrentIndex(idx)
        f.addRow("모델", self.cmb_model)

        self.spin_batch = QtWidgets.QSpinBox()
        self.spin_batch.setRange(5, 100)
        self.spin_batch.setValue(self.config.batch_size)
        f.addRow("배치 크기", self.spin_batch)

        self.spin_amb = QtWidgets.QDoubleSpinBox()
        self.spin_amb.setRange(0.0, 0.9)
        self.spin_amb.setSingleStep(0.05)
        self.spin_amb.setValue(self.config.ambiguity_threshold)
        f.addRow("모호 임계값", self.spin_amb)

        self.spin_excerpt = QtWidgets.QSpinBox()
        self.spin_excerpt.setRange(400, 6000)
        self.spin_excerpt.setSingleStep(100)
        self.spin_excerpt.setValue(self.config.max_excerpt_chars)
        f.addRow("본문 최대 글자", self.spin_excerpt)

        self.cmb_appearance = QtWidgets.QComboBox()
        self.cmb_appearance.addItems(["auto", "light", "dark"])
        idx = self.cmb_appearance.findText(self.config.appearance)
        if idx >= 0:
            self.cmb_appearance.setCurrentIndex(idx)
        f.addRow("테마", self.cmb_appearance)

        v.addWidget(api_card)

        btn_row = QtWidgets.QHBoxLayout()
        btn_row.addStretch(1)
        btn_save = QtWidgets.QPushButton("설정 저장")
        btn_save.setObjectName("Primary")
        btn_save.clicked.connect(self._save)
        btn_row.addWidget(btn_save)
        v.addLayout(btn_row)
        v.addStretch(1)

    def _save_key(self):
        key = self.edit_key.text().strip()
        if not key:
            return
        secure = set_api_key(key, self.config)
        self.edit_key.clear()
        self.edit_key.setPlaceholderText(
            "저장된 키 사용 중 — 덮어쓰려면 입력" + ("" if secure else " (keyring 없음 → config.json 평문)")
        )
        self.config_changed.emit()

    def _clear_key(self):
        # Deliberately idempotent: overwrites with empty string in both keyring and config
        set_api_key("", self.config)
        self.config.api_key_fallback = ""
        save_config(self.config)
        self.edit_key.setPlaceholderText("sk-… or AIzaSy… (비워두면 Mock 모드)")
        self.config_changed.emit()

    def _save(self):
        self.config.model = self.cmb_model.currentText()
        self.config.batch_size = self.spin_batch.value()
        self.config.ambiguity_threshold = self.spin_amb.value()
        self.config.max_excerpt_chars = self.spin_excerpt.value()
        self.config.appearance = self.cmb_appearance.currentText()
        save_config(self.config)
        self.config_changed.emit()
        QtWidgets.QMessageBox.information(self, "저장됨", "설정이 저장되었습니다.")
