"""Qt (PySide6) UI — works with Homebrew Python without system Tcl/Tk."""

from __future__ import annotations

import queue
import threading
from pathlib import Path

from PySide6.QtCore import QObject, Qt, Signal, Slot
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from sharp_local_batch.batch_runner import WatchController, scan_jobs
from sharp_local_batch.core import (
    PlySidecarResult,
    needs_ply_refresh,
    process_image_to_sidecar_ply,
    sidecar_ply_path,
)


class _Bridge(QObject):
    """Cross-thread signals (Qt delivers slots on the GUI thread)."""

    job_finished = Signal(object)
    watch_enqueue = Signal(object)


class SharpBatchQtWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("sharp_local_batch")
        self.resize(580, 520)

        self._job_q: queue.Queue[Path] = queue.Queue()
        self._quit_app = threading.Event()
        self._opts_lock = threading.Lock()
        self._snap_lim = False
        self._snap_max: int | None = None
        self._snap_skip = True
        self._batch_total = 0
        self._batch_done = 0
        self._watch: WatchController | None = None
        self._processed_session = 0

        self._bridge = _Bridge()
        self._bridge.job_finished.connect(self._on_job_done, Qt.QueuedConnection)
        self._bridge.watch_enqueue.connect(self._on_watch_enqueue, Qt.QueuedConnection)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Folder"))
        self._folder_edit = QLineEdit()
        row1.addWidget(self._folder_edit, stretch=1)
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._browse)
        row1.addWidget(browse)
        layout.addLayout(row1)

        row2 = QHBoxLayout()
        self._recursive_chk = QCheckBox("Include subfolders")
        self._recursive_chk.setChecked(True)
        row2.addWidget(self._recursive_chk)
        self._force_all_chk = QCheckBox("Reprocess all (ignore PLY freshness)")
        row2.addWidget(self._force_all_chk)
        layout.addLayout(row2)

        row3 = QHBoxLayout()
        self._limit_chk = QCheckBox("Limit splat count")
        row3.addWidget(self._limit_chk)
        row3.addWidget(QLabel("Max splats"))
        self._max_edit = QLineEdit("500000")
        self._max_edit.setMaximumWidth(120)
        row3.addWidget(self._max_edit)
        self._skip_chk = QCheckBox("Skip up-to-date PLY")
        self._skip_chk.setChecked(True)
        row3.addWidget(self._skip_chk)
        row3.addStretch()
        layout.addLayout(row3)
        self._limit_chk.toggled.connect(self._sync_limit_widgets)
        self._sync_limit_widgets()

        row4 = QHBoxLayout()
        scan_btn = QPushButton("Scan & queue jobs")
        scan_btn.clicked.connect(self._on_scan)
        row4.addWidget(scan_btn)
        stop_btn = QPushButton("Stop")
        stop_btn.clicked.connect(self._on_stop)
        row4.addWidget(stop_btn)
        self._watch_chk = QCheckBox("Watch folder (new / changed images)")
        self._watch_chk.toggled.connect(self._on_watch_toggled)
        row4.addWidget(self._watch_chk)
        layout.addLayout(row4)

        layout.addWidget(QLabel("Batch progress"))
        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        layout.addWidget(self._progress)
        self._progress_label = QLabel("—")
        layout.addWidget(self._progress_label)

        log_label = QLabel("Log")
        layout.addWidget(log_label)
        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setMinimumHeight(200)
        layout.addWidget(self._log, stretch=1)

        hint = (
            "Writes <name>.ply next to each image. "
            "Optional cap uses splat-transform (npm i -g @playcanvas/splat-transform)."
        )
        foot = QLabel(hint)
        foot.setWordWrap(True)
        foot.setStyleSheet("color: #666;")
        layout.addWidget(foot)

        threading.Thread(target=self._worker_loop, daemon=True).start()

    def _sync_limit_widgets(self) -> None:
        self._max_edit.setEnabled(self._limit_chk.isChecked())

    def _browse(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Select folder")
        if d:
            self._folder_edit.setText(d)

    def _snapshot_opts(self) -> None:
        lim = self._limit_chk.isChecked()
        mx = self._parse_max_splats() if lim else None
        if lim and mx is None:
            mx = 500_000
        skip = self._skip_chk.isChecked()
        with self._opts_lock:
            self._snap_lim = bool(lim)
            self._snap_max = mx if lim else None
            self._snap_skip = skip

    def _parse_max_splats(self) -> int | None:
        if not self._limit_chk.isChecked():
            return None
        raw = self._max_edit.text().strip()
        try:
            n = int(raw, 10)
        except ValueError:
            return None
        if n < 1:
            return None
        return min(n, 10_000_000)

    def _limit_options_valid(self) -> bool:
        if not self._limit_chk.isChecked():
            return True
        if self._parse_max_splats() is None:
            QMessageBox.critical(
                self,
                "Invalid max splats",
                "Enter a positive integer for max splats when limit is enabled.",
            )
            return False
        return True

    @Slot()
    def _on_scan(self) -> None:
        if not self._limit_options_valid():
            return
        raw = self._folder_edit.text().strip()
        if not raw:
            QMessageBox.warning(self, "Folder", "Choose a folder first.")
            return
        root = Path(raw).expanduser()
        if not root.is_dir():
            QMessageBox.critical(self, "Folder", f"Not a directory: {root}")
            return

        jobs = scan_jobs(
            root,
            self._recursive_chk.isChecked(),
            force_all=self._force_all_chk.isChecked(),
        )
        if not jobs:
            QMessageBox.information(self, "Scan", "No images need processing (PLY already fresh).")
            return

        self._snapshot_opts()
        self._batch_total = len(jobs)
        self._batch_done = 0
        self._progress.setMaximum(self._batch_total)
        self._progress.setValue(0)
        self._progress_label.setText(f"Queued {len(jobs)} job(s)…")
        self._log_line(f"--- Scan: {len(jobs)} job(s) ---")
        for p in jobs:
            self._job_q.put(p)

    @Slot()
    def _on_stop(self) -> None:
        try:
            while True:
                self._job_q.get_nowait()
        except queue.Empty:
            pass
        self._batch_total = 0
        self._batch_done = 0
        self._progress.setValue(0)
        self._progress.setMaximum(100)
        self._progress_label.setText("Stopped")
        self._log_line("--- Stop: queue cleared ---")

    @Slot(bool)
    def _on_watch_toggled(self, checked: bool) -> None:
        if checked:
            self._start_watch()
        else:
            self._stop_watch()

    def _start_watch(self) -> None:
        if not self._limit_options_valid():
            self._watch_chk.setChecked(False)
            return
        raw = self._folder_edit.text().strip()
        if not raw:
            QMessageBox.warning(self, "Watch", "Choose a folder first.")
            self._watch_chk.setChecked(False)
            return
        root = Path(raw).expanduser()
        if not root.is_dir():
            QMessageBox.critical(self, "Watch", f"Not a directory: {root}")
            self._watch_chk.setChecked(False)
            return
        if self._watch is not None:
            return
        self._snapshot_opts()

        def enqueue(p: Path) -> None:
            self._bridge.watch_enqueue.emit(p.resolve())

        self._watch = WatchController(
            root,
            enqueue,
            debounce_sec=0.6,
            recursive=self._recursive_chk.isChecked(),
        )
        self._watch.start()
        self._log_line(f"--- Watch started: {root} ---")

    def _stop_watch(self) -> None:
        if self._watch is not None:
            self._watch.stop()
            self._watch = None
            self._log_line("--- Watch stopped ---")

    @Slot(object)
    def _on_watch_enqueue(self, p: object) -> None:
        if not isinstance(p, Path):
            p = Path(p)
        self._snapshot_opts()
        self._job_q.put(p.resolve())

    def _worker_loop(self) -> None:
        while not self._quit_app.is_set():
            try:
                item = self._job_q.get(timeout=0.35)
            except queue.Empty:
                continue
            with self._opts_lock:
                lim = self._snap_lim
                max_s = self._snap_max
                skip = self._snap_skip
            if skip and not needs_ply_refresh(item):
                r = PlySidecarResult(
                    ok=True,
                    image_path=item,
                    ply_path=sidecar_ply_path(item),
                    message="Skipped (PLY up to date)",
                    skipped=True,
                )
            else:
                r = process_image_to_sidecar_ply(
                    item,
                    limit_splats=lim,
                    max_splats=max_s if lim else None,
                )
            self._bridge.job_finished.emit(r)

    @Slot(object)
    def _on_job_done(self, r: object) -> None:
        if not isinstance(r, PlySidecarResult):
            return
        if r.skipped:
            self._log_line(f"[skip] {r.image_path.name} — {r.message}")
        elif r.ok:
            self._processed_session += 1
            self._log_line(f"[ok] {r.image_path.name} — {r.message}")
        else:
            self._log_line(f"[err] {r.image_path.name} — {r.message}")

        if self._batch_total > 0:
            self._batch_done += 1
            self._progress.setValue(self._batch_done)
            self._progress_label.setText(
                f"{self._batch_done} / {self._batch_total} files"
            )
            if self._batch_done >= self._batch_total:
                self._batch_total = 0
                self._batch_done = 0
                self._progress.setValue(0)
                self._progress.setMaximum(100)
                self._progress_label.setText(
                    f"Batch done · session processed: {self._processed_session}"
                )

    def _log_line(self, text: str) -> None:
        self._log.append(text)

    def closeEvent(self, event: QCloseEvent) -> None:
        self._stop_watch()
        self._quit_app.set()
        event.accept()


def main() -> None:
    import sys

    app = QApplication(sys.argv)
    win = SharpBatchQtWindow()
    win.show()
    sys.exit(app.exec())
