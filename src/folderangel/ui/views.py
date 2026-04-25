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

        # Inline toast banner (취소 / 안내 / 오류).
        self._toast = QtWidgets.QFrame()
        self._toast.setObjectName("Toast")
        self._toast.setVisible(False)
        toast_layout = QtWidgets.QHBoxLayout(self._toast)
        toast_layout.setContentsMargins(14, 12, 14, 12)
        toast_layout.setSpacing(12)
        self._toast_title = QtWidgets.QLabel()
        self._toast_body = QtWidgets.QLabel()
        toast_text = QtWidgets.QVBoxLayout()
        toast_text.setSpacing(2)
        toast_text.addWidget(self._toast_title)
        toast_text.addWidget(self._toast_body)
        toast_layout.addLayout(toast_text, 1)
        outer.addWidget(self._toast)

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

        self.log_view = QtWidgets.QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(2000)
        self.log_view.setMinimumHeight(140)
        self.log_view.setStyleSheet(
            "QPlainTextEdit { background:#0f1115; color:#d6d6dc; border-radius:10px;"
            " padding:10px; font-family:'JetBrains Mono','SF Mono',Menlo,Consolas,monospace; font-size:12px; }"
        )
        self.log_view.setPlaceholderText("진행 로그가 여기에 한 줄씩 표시됩니다.")
        pc.addWidget(self.log_view, 1)
        outer.addWidget(self.progress_card, 1)

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
        self.btn_open_log = QtWidgets.QPushButton("로그 폴더 열기")
        self.btn_open_log.setObjectName("Ghost")
        self.btn_open_log.clicked.connect(self._open_log_dir)
        actions.addWidget(self.btn_open_log)
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
        self.refresh_api_badge()

    # ------------------------------------------------------------------
    def refresh_api_badge(self):
        from ..config import provider_label

        key = get_api_key(self.config)
        if key:
            self.badge_api.setText(f"{provider_label(self.config)} 연결됨")
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
            self.log_view.clear()
            self._set_toast(None)

    def show_canceling(self):
        self._set_toast(("warn", "취소 요청됨", "현재 단계가 안전하게 멈출 때까지 잠시만요…"))

    def show_canceled(self):
        self.set_running(False)
        self.progress_bar.setValue(0)
        self.progress_label.setText("취소되었습니다.")
        self._set_toast(("info", "정리를 취소했습니다", "이미 옮긴 파일은 그대로 유지됩니다. 다시 시작할 수 있습니다."))

    def _set_toast(self, payload):
        """Inline toast banner.  payload = (kind, title, body) or None."""
        if not hasattr(self, "_toast"):
            return
        if payload is None:
            self._toast.setVisible(False)
            return
        kind, title, body = payload
        palette = {
            "warn":  ("#FFF4E5", "#C37200", "#FFE2B2"),
            "info":  ("#EAF4FF", "#0B66C2", "#C7DEF8"),
            "error": ("#FFECEC", "#B3261E", "#F4C7C5"),
        }
        bg, fg, border = palette.get(kind, palette["info"])
        self._toast.setStyleSheet(
            f"QFrame#Toast {{ background:{bg}; border:1px solid {border}; "
            f"border-radius:12px; }}"
        )
        self._toast_title.setText(title)
        self._toast_title.setStyleSheet(f"color:{fg};font-weight:700;font-size:14px;")
        self._toast_body.setText(body)
        self._toast_body.setStyleSheet(f"color:{fg};font-size:12px;")
        self._toast.setVisible(True)

    def on_stage(self, stage: str, pct: float):
        self.stage_ind.set_active(stage)
        if pct < 0:
            # Indeterminate — show a busy/marquee bar so the user can see
            # the app is still alive during long LLM calls.
            self.progress_bar.setRange(0, 0)
        else:
            if self.progress_bar.maximum() == 0:
                self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(max(0, min(100, int(pct * 100))))

    def on_status(self, text: str):
        # Single short label up top + line-by-line tail in the log view.
        head = text if len(text) <= 90 else text[:87] + "…"
        self.progress_label.setText(head)
        from datetime import datetime as _dt

        ts = _dt.now().strftime("%H:%M:%S")
        self.log_view.appendPlainText(f"[{ts}] {text}")
        # auto-scroll to bottom
        sb = self.log_view.verticalScrollBar()
        sb.setValue(sb.maximum())

    def on_finished(self, op: OperationResult):
        self._last_op = op
        self.set_running(False)
        self.progress_bar.setValue(100)
        self.progress_label.setText(
            f"완료 — 이동 {op.total_moved}, 바로가기 {op.total_shortcuts}, 스킵 {op.total_skipped}"
        )
        usage = op.llm_usage
        if usage is None or usage.model == "mock" or usage.request_count == 0:
            llm_label = "0 (Mock)"
            cost_label = "₩0"
            speed_label = "—"
        else:
            llm_label = f"{usage.request_count}회"
            krw = usage.estimate_cost_krw()
            usd = usage.estimate_cost_usd()
            if krw < 1.0:
                cost_label = f"≈ ₩{krw:.2f}\n(${usd:.5f})"
            else:
                cost_label = f"≈ ₩{krw:,.1f}\n(${usd:.4f})"
            tps = usage.avg_tokens_per_second()
            speed_label = f"{tps:.1f} tok/s\n총 {usage.total_duration_s:.1f}s"
        self.stats_row.update_items(
            [
                ("스캔 파일", str(op.total_scanned)),
                ("이동", str(op.total_moved)),
                ("바로가기", str(op.total_shortcuts)),
                ("스킵", str(op.total_skipped)),
                ("LLM 호출", llm_label),
                ("예상 비용", cost_label),
                ("LLM 속도", speed_label),
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
        # Detect the user-cancel path and surface it gently rather than as
        # a scary "Critical Error" modal.
        low = (msg or "").lower()
        if "cancel" in low or "취소" in (msg or ""):
            self.show_canceled()
            return
        # Friendlier copy for common transport failures.
        friendly = msg or ""
        if "read timeout" in low or "timed out" in low:
            friendly = "LLM 응답이 시간 안에 도착하지 못했어요. 잠시 후 다시 시도해 주세요."
        elif "connectionerror" in low or "connection refused" in low:
            friendly = "LLM 서버에 연결하지 못했어요. 엔드포인트 URL과 서버 상태를 확인해 주세요."
        elif "invalid api key" in low or "unauthorized" in low or "401" in low:
            friendly = "API 키가 인증되지 않았어요. 설정에서 키를 다시 확인해 주세요."
        else:
            friendly = f"문제가 발생했어요: {msg}"
        try:
            from ..runlog import current_log_path

            lp = current_log_path()
            if lp is not None:
                friendly += f"  ·  자세한 내용은 로그를 확인하세요: {lp}"
        except Exception:
            pass
        self._set_toast(("error", "정리를 끝내지 못했어요", friendly))

    def _open_log_dir(self):
        from ..config import default_paths

        d = default_paths().logs_dir
        d.mkdir(parents=True, exist_ok=True)
        _open_in_explorer(d)

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
        self.edit_key.setPlaceholderText("API 키를 입력 (비워두면 Mock 모드 — 키 없이 실행)")
        if get_api_key(self.config):
            self.edit_key.setPlaceholderText("저장된 키 사용 중 — 덮어쓰려면 새 키 입력")
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
        # Provider-aware label kept in sync via _refresh_api_label().
        self.lbl_api_key = QtWidgets.QLabel("API 키")
        f.addRow(self.lbl_api_key, wrap)

        # Provider drop-down
        self.cmb_provider = QtWidgets.QComboBox()
        self.cmb_provider.addItem("Gemini (Google AI Studio)", "gemini")
        self.cmb_provider.addItem(
            "OpenAI 호환 (OpenAI / Qwen / Ollama / vLLM / OpenRouter / …)",
            "openai_compat",
        )
        for i in range(self.cmb_provider.count()):
            if self.cmb_provider.itemData(i) == self.config.llm_provider:
                self.cmb_provider.setCurrentIndex(i)
                break
        self.cmb_provider.currentIndexChanged.connect(self._on_provider_changed)
        f.addRow("LLM 제공자", self.cmb_provider)

        # Base URL (free text — works for any OpenAI-compatible endpoint).
        self.edit_base_url = QtWidgets.QLineEdit(self.config.llm_base_url)
        self.edit_base_url.setPlaceholderText(
            "예: https://api.openai.com/v1, http://localhost:11434/v1, "
            "https://generativelanguage.googleapis.com/v1beta/openai"
        )
        f.addRow("엔드포인트 URL", self.edit_base_url)

        # Free-text model so users on alternative providers can type any name.
        self.cmb_model = QtWidgets.QComboBox()
        self.cmb_model.setEditable(True)
        self.cmb_model.addItems(
            [
                "gemini-2.5-flash",
                "gemini-2.5-pro",
                "gemini-2.5-flash-lite",
                "gpt-4o-mini",
                "gpt-4o",
                "claude-3-5-sonnet",
                "llama-3.1-70b-instruct",
                "qwen2.5-72b-instruct",
            ]
        )
        if self.config.model:
            self.cmb_model.setCurrentText(self.config.model)
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

        # Reasoning ("thinking") mode toggle for Qwen3 / DeepSeek-R1 / etc.
        self.cmb_reasoning = QtWidgets.QComboBox()
        self.cmb_reasoning.addItem("끄기 (속도 우선, 권장)", "off")
        self.cmb_reasoning.addItem("켜기 (사고 과정 사용 — 5~10배 느려짐)", "on")
        self.cmb_reasoning.addItem("자동 (현재는 끄기와 동일)", "auto")
        for i in range(self.cmb_reasoning.count()):
            if self.cmb_reasoning.itemData(i) == self.config.reasoning_mode:
                self.cmb_reasoning.setCurrentIndex(i)
                break
        self.cmb_reasoning.setToolTip(
            "Qwen3/DeepSeek-R1 같은 reasoning 모델의 <think> 단계.\n"
            "폴더 분류는 단순 JSON 출력이라 보통 끄는 것이 5~10배 빠릅니다."
        )
        f.addRow("Reasoning 모드", self.cmb_reasoning)

        self.chk_economy = QtWidgets.QCheckBox("LLM 호출 최소화 (Economy 모드)")
        self.chk_economy.setChecked(self.config.economy_mode)
        self.chk_economy.setToolTip(
            "한 번의 호출로 폴더 설계와 분류를 동시에 수행합니다.\n"
            "프로젝트명 인식이 좋아지고 토큰 사용량이 크게 줄어듭니다."
        )
        f.addRow("LLM 호출 절약", self.chk_economy)

        self.spin_econ_max = QtWidgets.QSpinBox()
        self.spin_econ_max.setRange(20, 500)
        self.spin_econ_max.setValue(self.config.economy_max_files)
        self.spin_econ_max.setToolTip("Economy 모드에서 한 호출당 보내는 최대 파일 수")
        f.addRow("호출당 최대 파일", self.spin_econ_max)

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
        # Synchronise the API-key label/placeholder with the current provider.
        self._refresh_api_label()
        self.edit_base_url.textChanged.connect(lambda _t: self._refresh_api_label())

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

    def _on_provider_changed(self, _idx: int):
        provider = self.cmb_provider.currentData()
        # Only suggest a base URL when the user has not typed one yet.
        if not self.edit_base_url.text().strip():
            if provider == "openai_compat":
                self.edit_base_url.setText("https://api.openai.com/v1")
            elif provider == "gemini":
                self.edit_base_url.setText("")  # use built-in default
        self._refresh_api_label(provider=provider)

    def _refresh_api_label(self, provider: str | None = None) -> None:
        if provider is None:
            provider = self.cmb_provider.currentData() or "gemini"
        # Build a transient Config with the right provider so provider_label
        # picks the human-friendly name (Qwen / OpenAI / Ollama / …).
        from ..config import provider_label

        proxy = type(self.config)()
        proxy.llm_provider = provider
        proxy.llm_base_url = self.edit_base_url.text().strip()
        name = provider_label(proxy)
        self.lbl_api_key.setText(f"{name} API 키")

    def _save(self):
        provider = self.cmb_provider.currentData() or "gemini"
        self.config.llm_provider = provider
        self.config.llm_base_url = self.edit_base_url.text().strip()
        self.config.model = self.cmb_model.currentText().strip()
        self.config.batch_size = self.spin_batch.value()
        self.config.ambiguity_threshold = self.spin_amb.value()
        self.config.max_excerpt_chars = self.spin_excerpt.value()
        self.config.appearance = self.cmb_appearance.currentText()
        self.config.economy_mode = self.chk_economy.isChecked()
        self.config.economy_max_files = self.spin_econ_max.value()
        self.config.reasoning_mode = self.cmb_reasoning.currentData() or "off"
        save_config(self.config)
        self.config_changed.emit()
        QtWidgets.QMessageBox.information(self, "저장됨", "설정이 저장되었습니다.")
