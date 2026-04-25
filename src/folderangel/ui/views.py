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


def provider_label_for_ui(provider: str, base_url: str) -> str:
    """Convenience wrapper used by the Settings card; isolates the UI
    layer from the config module's dataclass requirements."""
    from ..config import provider_label, Config

    proxy = Config()
    proxy.llm_provider = provider
    proxy.llm_base_url = base_url or ""
    return provider_label(proxy)


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
        # 'Undo' is only meaningful for the operation we just produced —
        # which IS the latest one at this moment.  It will turn into "an
        # older op" the moment the user runs another organise; the
        # History tab governs that case explicitly.
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

        self.table = QtWidgets.QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(
            ["파일", "매치", "카테고리", "현재 위치", "스니펫", "정리 시각"]
        )
        hdr = self.table.horizontalHeader()
        hdr.setSectionResizeMode(QtWidgets.QHeaderView.Interactive)
        hdr.setStretchLastSection(False)
        # Sensible default column widths so the filename never starts as
        # an ellipsis; user can still drag any column wider.
        self.table.setColumnWidth(0, 320)   # filename
        self.table.setColumnWidth(1, 70)    # match field
        self.table.setColumnWidth(2, 200)   # category
        self.table.setColumnWidth(3, 320)   # current location
        self.table.setColumnWidth(4, 380)   # snippet
        self.table.setColumnWidth(5, 130)   # timestamp
        hdr.setSectionResizeMode(4, QtWidgets.QHeaderView.Stretch)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.table.doubleClicked.connect(self._open_selected)
        self.table.setAlternatingRowColors(True)
        self.table.setWordWrap(True)
        self.table.verticalHeader().setVisible(False)
        self.table.verticalHeader().setDefaultSectionSize(30)
        v.addWidget(self.table, 1)

    def focus_search(self):
        self.search.setFocus(QtCore.Qt.ShortcutFocusReason)
        self.search.selectAll()

    def _do_search(self):
        q = self.search.text().strip()
        hits = self.index_db.search(q, limit=300)
        self.table.setRowCount(len(hits))
        for row, h in enumerate(hits):
            filename = Path(h.new_path).name
            cells = [
                (filename,           f"{filename}\n원본: {h.original_path}"),
                (h.matched_in or "", f"매치 위치: {h.matched_in or '미상'}"),
                (h.category,         h.category),
                (h.new_path,         h.new_path),
                (h.snippet or "",    h.snippet or "(미리보기 없음)"),
                (h.created_at,       h.created_at),
            ]
            for col, (text, tip) in enumerate(cells):
                item = QtWidgets.QTableWidgetItem(text)
                item.setToolTip(tip)
                self.table.setItem(row, col, item)
        self.table.resizeRowsToContents()

    def _open_selected(self, idx: QtCore.QModelIndex):
        row = idx.row()
        # "Current location" is col 3 in the new layout; fall back to
        # the filename cell if a row is sparse.
        item = self.table.item(row, 3) or self.table.item(row, 0)
        if item:
            p = Path(item.text())
            _open_in_explorer(p if p.exists() else p.parent)


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
        latest = self.index_db.latest_operation_id()
        is_latest = (latest is not None and op_id == latest)

        if is_latest:
            resp = QtWidgets.QMessageBox.question(
                self,
                "롤백",
                f"가장 최근 정리(#{op_id})를 되돌립니다. 진행할까요?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            )
            if resp == QtWidgets.QMessageBox.Yes:
                self.rollback_requested.emit(op_id)
            return

        # Older operation: gate behind a much louder, explicit warning.
        warn = QtWidgets.QMessageBox(self)
        warn.setIcon(QtWidgets.QMessageBox.Warning)
        warn.setWindowTitle("이전 정리 롤백 — 위험")
        warn.setText(
            f"오퍼레이션 #{op_id} 은(는) 가장 최근 정리가 아닙니다.\n"
            "그 이후에 사용자가 폴더를 더 정리했거나 파일을 옮겼다면,\n"
            "이 작업은 새로 만든 결과를 덮어쓰거나 깨뜨릴 수 있습니다."
        )
        warn.setInformativeText(
            "안전하게 되돌리려면 가장 최근 정리부터 차례로 롤백하세요.\n"
            "그래도 진행하시겠다면 '강제 롤백'을 선택하세요. "
            "기록과 다르게 이미 옮긴 파일은 자동으로 건너뜁니다."
        )
        cancel = warn.addButton("취소", QtWidgets.QMessageBox.RejectRole)
        force = warn.addButton("강제 롤백", QtWidgets.QMessageBox.DestructiveRole)
        warn.setDefaultButton(cancel)
        warn.exec()
        if warn.clickedButton() is force:
            # Emit with the force-flag convention: negative op_id means force.
            # (Keeps the existing Signal signature backwards compatible.)
            self.rollback_requested.emit(-op_id)


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
        sub = QtWidgets.QLabel("LLM 연결과 분류 동작을 조정합니다. 변경 후 '설정 저장'을 누르면 적용됩니다.")
        sub.setObjectName("Subtitle")
        v.addWidget(t)
        v.addWidget(sub)

        # ────────────────────────────────────────────────────────────
        # Card 1 — LLM 연결 (top of the page; provider drives everything
        # below it).  Order matches the user's mental flow:
        #   "어디에 연결하지? → 어떤 모델? → 어떤 키?"
        # ────────────────────────────────────────────────────────────
        conn_card = Card()
        c1 = QtWidgets.QVBoxLayout(conn_card)
        c1.setContentsMargins(18, 16, 18, 16)
        c1.setSpacing(12)
        c1_title = QtWidgets.QLabel("LLM 연결")
        c1_title.setStyleSheet("font-size:16px;font-weight:600;")
        c1.addWidget(c1_title)
        c1_sub = QtWidgets.QLabel("제공자를 먼저 고르면 그 아래 항목이 그 제공자에 맞게 채워집니다.")
        c1_sub.setStyleSheet("color:#6e6e73;font-size:12px;")
        c1.addWidget(c1_sub)

        f1 = QtWidgets.QFormLayout()
        f1.setSpacing(10)

        # 1) Provider — top of the form, this is the master selector.
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
        f1.addRow("LLM 제공자", self.cmb_provider)

        # 2) Endpoint URL — gemini hides this row entirely, openai_compat
        #    shows it with a sensible placeholder.
        self.lbl_base_url = QtWidgets.QLabel("엔드포인트 URL")
        self.edit_base_url = QtWidgets.QLineEdit()
        self.edit_base_url.setPlaceholderText(
            "예: https://api.openai.com/v1, http://localhost:11434/v1"
        )
        f1.addRow(self.lbl_base_url, self.edit_base_url)

        # 3) Model — combobox is editable so any backend's model id works.
        self.lbl_model = QtWidgets.QLabel("모델")
        self.cmb_model = QtWidgets.QComboBox()
        self.cmb_model.setEditable(True)
        f1.addRow(self.lbl_model, self.cmb_model)

        # 4) API key — last in the connection card, not first.  Echo
        #    masked, dedicated save / delete buttons inline.
        self.lbl_api_key = QtWidgets.QLabel("API 키")
        self.edit_key = QtWidgets.QLineEdit()
        self.edit_key.setEchoMode(QtWidgets.QLineEdit.Password)
        self.edit_key.setMinimumWidth(280)
        key_row = QtWidgets.QHBoxLayout()
        key_row.setContentsMargins(0, 0, 0, 0)
        key_row.setSpacing(8)
        key_row.addWidget(self.edit_key, 1)
        btn_save_key = QtWidgets.QPushButton("저장")
        btn_save_key.setObjectName("Primary")
        btn_save_key.clicked.connect(self._save_key)
        btn_clear_key = QtWidgets.QPushButton("삭제")
        btn_clear_key.setObjectName("Ghost")
        btn_clear_key.clicked.connect(self._clear_key)
        key_row.addWidget(btn_save_key)
        key_row.addWidget(btn_clear_key)
        wrap_key = QtWidgets.QWidget()
        wrap_key.setLayout(key_row)
        f1.addRow(self.lbl_api_key, wrap_key)

        c1.addLayout(f1)

        # 5) Connection status (alive / mock / inferred) — single line.
        self.lbl_status = QtWidgets.QLabel("")
        self.lbl_status.setStyleSheet("color:#6e6e73;font-size:12px;")
        c1.addWidget(self.lbl_status)

        v.addWidget(conn_card)

        # ────────────────────────────────────────────────────────────
        # Card 2 — 고급 옵션 (대부분 자동, 사용자가 만지지 않아도 됨)
        # ────────────────────────────────────────────────────────────
        adv_card = Card()
        c2 = QtWidgets.QVBoxLayout(adv_card)
        c2.setContentsMargins(18, 16, 18, 16)
        c2.setSpacing(12)
        c2_title = QtWidgets.QLabel("고급 (선택사항)")
        c2_title.setStyleSheet("font-size:16px;font-weight:600;")
        c2.addWidget(c2_title)
        c2_sub = QtWidgets.QLabel(
            "배치 크기 · 모호 임계값 · 컨텍스트 분할 같은 항목은 모델 컨텍스트 한도와 "
            "파일 수에 맞춰 자동으로 결정됩니다. 아래는 정말 필요한 사람만 바꾸세요."
        )
        c2_sub.setWordWrap(True)
        c2_sub.setStyleSheet("color:#6e6e73;font-size:12px;")
        c2.addWidget(c2_sub)

        f3 = QtWidgets.QFormLayout()
        f3.setSpacing(10)

        # Reasoning toggle — only meaningful for OpenAI-compat reasoning
        # models (Qwen3 / DeepSeek-R1 / Magistral / Phi-4-mini-reasoning).
        self.lbl_reasoning = QtWidgets.QLabel("Reasoning 모드")
        self.cmb_reasoning = QtWidgets.QComboBox()
        self.cmb_reasoning.addItem("끄기 (속도 우선, 권장)", "off")
        self.cmb_reasoning.addItem("켜기 (사고 과정 사용 — 5~10배 느려짐)", "on")
        self.cmb_reasoning.addItem("자동 (현재는 끄기와 동일)", "auto")
        for i in range(self.cmb_reasoning.count()):
            if self.cmb_reasoning.itemData(i) == self.config.reasoning_mode:
                self.cmb_reasoning.setCurrentIndex(i)
                break
        self.cmb_reasoning.setToolTip(
            "Qwen3 / DeepSeek-R1 같은 reasoning 모델의 <think> 단계.\n"
            "폴더 분류는 단순 JSON 출력이라 보통 끄는 것이 5~10배 빠릅니다."
        )
        f3.addRow(self.lbl_reasoning, self.cmb_reasoning)

        c2.addLayout(f3)
        v.addWidget(adv_card)

        # ────────────────────────────────────────────────────────────
        # Card 4 — 외관
        # ────────────────────────────────────────────────────────────
        look_card = Card()
        c4 = QtWidgets.QVBoxLayout(look_card)
        c4.setContentsMargins(18, 16, 18, 16)
        c4.setSpacing(12)
        c4_title = QtWidgets.QLabel("외관")
        c4_title.setStyleSheet("font-size:16px;font-weight:600;")
        c4.addWidget(c4_title)
        f4 = QtWidgets.QFormLayout()
        f4.setSpacing(10)
        self.cmb_appearance = QtWidgets.QComboBox()
        self.cmb_appearance.addItems(["auto", "light", "dark"])
        idx = self.cmb_appearance.findText(self.config.appearance)
        if idx >= 0:
            self.cmb_appearance.setCurrentIndex(idx)
        f4.addRow("테마", self.cmb_appearance)
        c4.addLayout(f4)
        v.addWidget(look_card)

        # Save button
        btn_row = QtWidgets.QHBoxLayout()
        btn_row.addStretch(1)
        btn_save = QtWidgets.QPushButton("설정 저장")
        btn_save.setObjectName("Primary")
        btn_save.clicked.connect(self._save)
        btn_row.addWidget(btn_save)
        v.addLayout(btn_row)
        v.addStretch(1)

        # Initial sync — populate everything based on the current provider.
        self._provider_pref_cache = dict(
            getattr(self.config, "llm_settings_by_provider", {}) or {}
        )
        self._reapply_provider_view(self.cmb_provider.currentData() or "gemini")

    # ------------------------------------------------------------------
    # Provider-aware view machinery.
    # ------------------------------------------------------------------
    _GEMINI_MODELS = [
        "gemini-2.5-flash",
        "gemini-2.5-pro",
        "gemini-2.5-flash-lite",
        "gemini-1.5-flash",
        "gemini-1.5-pro",
    ]
    _OPENAI_COMPAT_MODELS = [
        "gpt-4o-mini",
        "gpt-4o",
        "gpt-4.1-mini",
        "claude-3-5-sonnet",
        "claude-3-5-haiku",
        "qwen2.5-72b-instruct",
        "qwen3.6-35b-a3b",
        "llama-3.1-70b-instruct",
    ]
    _DEFAULT_BASE_URL = {
        "gemini": "",  # built-in Google host
        "openai_compat": "https://api.openai.com/v1",
    }

    def _on_provider_changed(self, _idx: int):
        # Stash whatever the user had typed for the *previous* provider
        # before we repaint, so flipping back returns to it instead of a
        # blank line.
        self._stash_current_provider_prefs()
        provider = self.cmb_provider.currentData() or "gemini"
        self._reapply_provider_view(provider)

    def _stash_current_provider_prefs(self) -> None:
        # Read the previous provider out of the cache key (the one whose
        # values are currently shown).
        prev = getattr(self, "_active_provider", None)
        if prev is None:
            return
        self._provider_pref_cache[prev] = {
            "base_url": self.edit_base_url.text().strip(),
            "model": self.cmb_model.currentText().strip(),
        }

    def _reapply_provider_view(self, provider: str) -> None:
        """Repaint the connection card to match ``provider``.

        Behaviour:
          * Endpoint URL row hidden for Gemini (it always uses the
            built-in Google host).  Shown + populated with the user's
            previous value for OpenAI-compat.
          * Model dropdown's preset list switches per provider; the
            stored model for *that* provider is restored.
          * API-key field placeholder, label, and on-disk slot all
            switch to the provider-specific keyring entry.
          * Reasoning row is hidden for Gemini (irrelevant) and shown
            for OpenAI-compat.
          * Connection status line summarises what we'll talk to.
        """
        from ..config import provider_label

        self._active_provider = provider
        cache = self._provider_pref_cache.get(provider) or {}

        # Endpoint URL — hide for Gemini, show for OpenAI-compat.
        is_compat = (provider == "openai_compat")
        self.lbl_base_url.setVisible(is_compat)
        self.edit_base_url.setVisible(is_compat)
        self.edit_base_url.setText(
            cache.get("base_url") or self._DEFAULT_BASE_URL.get(provider, "")
        )

        # Model — swap preset list, restore per-provider value.
        presets = (
            self._OPENAI_COMPAT_MODELS if is_compat else self._GEMINI_MODELS
        )
        self.cmb_model.blockSignals(True)
        self.cmb_model.clear()
        self.cmb_model.addItems(presets)
        remembered_model = cache.get("model") or presets[0]
        self.cmb_model.setCurrentText(remembered_model)
        self.cmb_model.blockSignals(False)

        # API-key — load whatever's in keyring for *this* provider only.
        self.edit_key.clear()
        from ..config import get_api_key

        existing = get_api_key(self.config, provider=provider)
        proxy = type(self.config)()
        proxy.llm_provider = provider
        proxy.llm_base_url = self.edit_base_url.text().strip()
        pname = provider_label(proxy)
        self.lbl_api_key.setText(f"{pname} API 키")
        if existing:
            self.edit_key.setPlaceholderText(
                f"{pname} 키 저장됨 — 덮어쓰려면 새 키 입력"
            )
        else:
            if provider == "gemini":
                hint = "예: AIzaSy… (Google AI Studio)"
            else:
                hint = "예: sk-… (또는 로컬 서버 키)"
            self.edit_key.setPlaceholderText(f"비워두면 Mock 모드. {hint}")

        # Reasoning row — only for OpenAI-compat.
        self.lbl_reasoning.setVisible(is_compat)
        self.cmb_reasoning.setVisible(is_compat)

        # Status line at the bottom of the connection card.
        if existing:
            target = self.edit_base_url.text().strip() if is_compat else "Google AI Studio"
            self.lbl_status.setText(f"● 연결 준비 — {pname} · {target or '기본'} · 모델 {remembered_model}")
            self.lbl_status.setStyleSheet("color:#0a8a3a;font-size:12px;")
        else:
            self.lbl_status.setText("○ Mock 모드 — API 키가 없으면 휴리스틱 분류로 동작합니다.")
            self.lbl_status.setStyleSheet("color:#a07000;font-size:12px;")

    # ------------------------------------------------------------------
    # Save / clear API key (provider-aware).
    # ------------------------------------------------------------------
    def _save_key(self):
        key = self.edit_key.text().strip()
        if not key:
            return
        provider = self.cmb_provider.currentData() or "gemini"
        secure = set_api_key(key, self.config, provider=provider)
        self.edit_key.clear()
        suffix = "" if secure else " (keyring 없음 → config.json 평문)"
        self.edit_key.setPlaceholderText(
            f"{provider_label_for_ui(provider, self.edit_base_url.text())} 키 저장됨 — 덮어쓰려면 입력{suffix}"
        )
        self._reapply_provider_view(provider)
        self.config_changed.emit()

    def _clear_key(self):
        provider = self.cmb_provider.currentData() or "gemini"
        set_api_key("", self.config, provider=provider)
        self.config.api_key_fallback = ""
        save_config(self.config)
        self._reapply_provider_view(provider)
        self.config_changed.emit()

    # ------------------------------------------------------------------
    def _save(self):
        provider = self.cmb_provider.currentData() or "gemini"
        # Persist whatever the user had on screen, plus the cached
        # values for the *other* provider so they survive a switch.
        self._stash_current_provider_prefs()
        self.config.llm_provider = provider
        self.config.llm_base_url = self.edit_base_url.text().strip()
        self.config.model = self.cmb_model.currentText().strip()
        self.config.llm_settings_by_provider = {
            "gemini": self._provider_pref_cache.get("gemini") or
                      {"base_url": "", "model": "gemini-2.5-flash"},
            "openai_compat": self._provider_pref_cache.get("openai_compat") or
                             {"base_url": "https://api.openai.com/v1",
                              "model": "gpt-4o-mini"},
        }
        # Make sure the *current* provider's slot reflects what we just typed.
        self.config.llm_settings_by_provider[provider] = {
            "base_url": self.config.llm_base_url,
            "model": self.config.model,
        }
        # Auto-tuned values: never user-editable.  Always force the
        # behaviour to "single call when it fits, micro-batch otherwise"
        # and a sensible ambiguity threshold so users don't have to
        # think about it.
        self.config.economy_mode = True
        self.config.local_microbatch_mode = "auto"
        self.config.batch_size = 30          # legacy fallback only
        self.config.ambiguity_threshold = 0.15
        self.config.max_excerpt_chars = 1800
        # economy_max_files is kept as a soft cap; the planner now uses
        # the model's real context window when available.
        self.config.economy_max_files = max(self.config.economy_max_files or 120, 60)
        self.config.appearance = self.cmb_appearance.currentText()
        self.config.reasoning_mode = self.cmb_reasoning.currentData() or "off"
        save_config(self.config)
        self.config_changed.emit()

        # Modal "saved" toast — pin a min-width so the title can't get
        # ellipsised to "저" on small windows.
        box = QtWidgets.QMessageBox(self)
        box.setIcon(QtWidgets.QMessageBox.Information)
        box.setWindowTitle("설정 저장됨")
        box.setText("설정이 저장되었습니다.")
        box.setStandardButtons(QtWidgets.QMessageBox.Ok)
        # Force a roomy minimum so the title bar shows the full text on
        # any window size, including small / non-maximised states.
        box.setStyleSheet("QLabel{min-width:280px;}")
        box.exec()
