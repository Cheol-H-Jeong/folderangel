"""Background worker for the UI thread.

Runs the full pipeline in a QThread so the GUI stays responsive.  Exposes
Qt signals for stage changes, progress, and results.
"""
from __future__ import annotations

from pathlib import Path

from PySide6 import QtCore

from .config import Config
from .index import IndexDB
from .models import OperationResult
from .pipeline import run as run_pipeline


class OrganizeWorker(QtCore.QObject):
    stage_changed = QtCore.Signal(str, float)  # stage, progress 0..1 overall
    status = QtCore.Signal(str)  # free-form text
    finished = QtCore.Signal(object)  # OperationResult
    failed = QtCore.Signal(str)

    def __init__(
        self,
        target: Path,
        config: Config,
        recursive: bool,
        dry_run: bool,
        index_db: IndexDB | None,
    ) -> None:
        super().__init__()
        self.target = target
        self.config = config
        self.recursive = recursive
        self.dry_run = dry_run
        self.index_db = index_db
        self._cancel = False

    @QtCore.Slot()
    def run(self):
        try:
            def _progress(msg: str, pct: float):
                if self._cancel:
                    raise RuntimeError("canceled by user")
                # derive a rough overall progress by stage prefix
                stage = _stage_from_msg(msg)
                self.stage_changed.emit(stage, pct)
                self.status.emit(msg)

            op = run_pipeline(
                target_root=self.target,
                config=self.config,
                recursive=self.recursive,
                dry_run=self.dry_run,
                index_db=self.index_db,
                progress=_progress,
                cancel_check=lambda: self._cancel,
            )
            self.finished.emit(op)
        except Exception as exc:
            self.failed.emit(str(exc))

    def cancel(self):
        self._cancel = True
        self.status.emit("취소 중… (현재 작업이 안전하게 멈출 때까지 잠시 대기)")


def _stage_from_msg(msg: str) -> str:
    m = msg.strip().lower()
    if m.startswith("scan"):
        return "scan"
    if m.startswith("parse") or any(msg.lower().endswith(ext) for ext in (".pdf", ".docx", ".pptx", ".xlsx", ".hwp", ".hwpx", ".txt", ".md")):
        return "parse"
    if (
        m.startswith("plan")
        or m.startswith("stage-a")
        or m.startswith("stage-b")
        or m.startswith("stage-merge")
        or m.startswith("mock-planner")
        or m.startswith("plan-design")
        or m.startswith("plan-assign")
    ):
        return "plan"
    if m.startswith("organize") or m.startswith("move") or m.startswith("  ↳") or m.startswith("  ⚠"):
        return "organize"
    return "organize"
