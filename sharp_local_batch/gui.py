"""Tkinter UI for folder batch + optional filesystem watch."""

from __future__ import annotations

import queue
import sys
import threading
import time
import tkinter as tk
from collections.abc import Callable, Mapping
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from sharp_local_batch.batch_runner import WatchController, scan_jobs
from sharp_local_batch.gui_settings import (
    clear_saved_batch_gui_settings,
    default_batch_gui_settings,
    load_batch_gui_settings,
    save_batch_gui_settings,
)
from sharp_local_batch.core import (
    PHOTOS_LIBRARY_MIRROR_HELP,
    PlySidecarResult,
    default_macos_photos_library_path,
    format_elapsed_for_log,
    is_photos_library_bundle,
    output_ply_path_for_job,
    sidecar_ply_path,
    update_ply_sidecar,
)


class SharpBatchGui:
    def __init__(self) -> None:
        self.root = tk.Tk()
        from sharp_local_batch._version import __version__

        self.root.title(f"Sharp Local batch {__version__}")
        self.root.minsize(640, 440)

        self._folder_var = tk.StringVar(value="")
        self._recursive_var = tk.BooleanVar(value=True)
        self._force_all_var = tk.BooleanVar(value=False)
        self._limit_var = tk.BooleanVar(value=False)
        self._max_splats_var = tk.StringVar(value="500000")
        self._skip_up_to_date_var = tk.BooleanVar(value=True)
        self._spz_var = tk.BooleanVar(value=True)
        self._spz_only_var = tk.BooleanVar(value=False)
        self._remove_ply_after_spz_var = tk.BooleanVar(value=True)
        self._watch_var = tk.BooleanVar(value=False)
        self._mirror_var = tk.BooleanVar(value=False)
        self._output_mirror_var = tk.StringVar(value="")
        self._photos_lib_var = tk.BooleanVar(value=False)

        self._job_q: queue.Queue[Path] = queue.Queue()
        self._quit_app = threading.Event()
        self._opts_lock = threading.Lock()
        self._snap_lim = False
        self._snap_max: int | None = None
        self._snap_skip = True
        self._snap_spz = True
        self._snap_spz_only = False
        self._snap_remove_ply_after_spz = False
        self._snap_mirror_output: Path | None = None
        self._snap_input_root: Path | None = None
        self._scan_running = False
        self._batch_total = 0
        self._batch_done = 0
        self._batch_start_time = 0.0
        self._watch: WatchController | None = None
        self._processed_session = 0
        self._srv_running = False
        self._srv_thread: threading.Thread | None = None
        self._srv_httpd: object | None = None
        self._model_state = "idle"
        self._model_busy_download = False
        self._model_busy_cancel = False

        self._build_ui()

        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self._apply_persisted_settings(load_batch_gui_settings())
        self._poll_model_status()

    def _collect_persisted_settings(self) -> dict[str, object]:
        return {
            "folder": self._folder_var.get().strip(),
            "recursive": self._recursive_var.get(),
            "force_all": self._force_all_var.get(),
            "limit_splats": self._limit_var.get(),
            "max_splats": self._max_splats_var.get().strip() or "500000",
            "skip_up_to_date": self._skip_up_to_date_var.get(),
            "export_spz": self._spz_var.get(),
            "spz_only": self._spz_only_var.get(),
            "remove_ply_after_spz": self._remove_ply_after_spz_var.get(),
            "mirror": self._mirror_var.get(),
            "output_mirror": self._output_mirror_var.get().strip(),
            "use_photos_library": self._photos_lib_var.get(),
        }

    def _apply_persisted_settings(self, d: Mapping[str, object]) -> None:
        self._recursive_var.set(bool(d.get("recursive", True)))
        self._force_all_var.set(bool(d.get("force_all", False)))
        self._limit_var.set(bool(d.get("limit_splats", False)))
        self._max_splats_var.set(str(d.get("max_splats", "500000")))
        self._skip_up_to_date_var.set(bool(d.get("skip_up_to_date", True)))
        self._spz_var.set(bool(d.get("export_spz", True)))
        self._spz_only_var.set(bool(d.get("spz_only", False)))
        self._remove_ply_after_spz_var.set(bool(d.get("remove_ply_after_spz", True)))
        self._output_mirror_var.set(str(d.get("output_mirror", "")))

        use_pl = bool(d.get("use_photos_library", False)) and sys.platform == "darwin"
        lib = default_macos_photos_library_path()
        lib_ok = use_pl and lib.is_dir()

        if lib_ok:
            self._photos_lib_var.set(True)
            self._folder_var.set(str(lib))
            self._mirror_var.set(True)
            self._mirror_cb.state(["disabled"])
        else:
            self._photos_lib_var.set(False)
            self._mirror_cb.state(["!disabled"])
            self._folder_var.set(str(d.get("folder", "")))
            self._mirror_var.set(bool(d.get("mirror", False)))

        self._sync_force_skip_widgets()
        self._sync_limit_widgets()
        self._on_spz_only_toggle()
        self._sync_remove_ply_widgets()
        self._sync_mirror_widgets()

    # save_batch_gui_settings removes gui_settings.json when merged values match defaults.
    def _persist_gui_settings(self) -> None:
        try:
            save_batch_gui_settings(self._collect_persisted_settings())
        except OSError:
            pass

    def _on_reset_settings(self) -> None:
        if not messagebox.askyesno(
            "Reset settings",
            "Restore all options to defaults and forget saved preferences?",
            default=messagebox.NO,
        ):
            return
        clear_saved_batch_gui_settings()
        self._apply_persisted_settings(default_batch_gui_settings())
        self._log_line("--- Settings reset to defaults (saved preferences cleared) ---")

    def _build_ui(self) -> None:
        pad = {"padx": 10, "pady": 4}
        root_f = ttk.Frame(self.root, padding=10)
        root_f.pack(fill=tk.BOTH, expand=True)

        model_f = ttk.LabelFrame(root_f, text="SHARP model", padding=6)
        model_f.pack(fill=tk.X, **pad)
        self._model_status_label = ttk.Label(
            model_f, text="Checking SHARP model…", wraplength=560
        )
        self._model_status_label.pack(anchor=tk.W)
        self._model_progress = ttk.Progressbar(
            model_f, mode="determinate", maximum=100, value=0
        )
        self._model_progress.pack(fill=tk.X, pady=(4, 0))
        self._model_progress.pack_forget()
        model_btns = ttk.Frame(model_f)
        model_btns.pack(fill=tk.X, pady=(6, 0))
        self._model_download_btn = ttk.Button(
            model_btns, text="Download", command=self._on_model_download
        )
        self._model_download_btn.pack(side=tk.LEFT)
        self._model_cancel_btn = ttk.Button(
            model_btns, text="Cancel", command=self._on_model_cancel
        )
        self._model_remove_btn = ttk.Button(
            model_btns, text="Remove", command=self._on_model_remove
        )
        self._model_remove_btn.pack(side=tk.LEFT, padx=(8, 0))

        row1 = ttk.Frame(root_f)
        row1.pack(fill=tk.X, **pad)
        ttk.Label(row1, text="Folder").pack(side=tk.LEFT)
        ttk.Entry(row1, textvariable=self._folder_var, width=48).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=(8, 4)
        )
        ttk.Button(row1, text="Browse…", command=self._browse).pack(side=tk.LEFT)

        if sys.platform == "darwin":
            row_pl = ttk.Frame(root_f)
            row_pl.pack(fill=tk.X, **pad)
            ttk.Checkbutton(
                row_pl,
                text=(
                    "Use system Photos library as source folder "
                    "(Photos Library.photoslibrary; mirror required)"
                ),
                variable=self._photos_lib_var,
                command=self._on_macos_photos_library_toggle,
            ).pack(anchor=tk.W)
            ttk.Label(
                row_pl,
                text=(
                    "Fills the Folder field above — one source only, not an extra path. "
                    "Uncheck to browse your own folder."
                ),
                wraplength=520,
                foreground="#666",
            ).pack(anchor=tk.W, padx=(22, 0), pady=(0, 2))

        row1b = ttk.Frame(root_f)
        row1b.pack(fill=tk.X, **pad)
        self._mirror_cb = ttk.Checkbutton(
            row1b,
            text="Mirror PLY output (same subfolders under target)",
            variable=self._mirror_var,
        )
        self._mirror_cb.pack(anchor=tk.W)
        ttk.Label(
            root_f,
            text=(
                "Example: ~/Photos/source/folder1/example.jpg → "
                "target_mirror/Photos/source/folder1/example.ply "
                "(path under the mirror target repeats from your home folder)."
            ),
            wraplength=520,
            foreground="#666",
        ).pack(anchor=tk.W, padx=10, pady=(0, 2))

        row1c = ttk.Frame(root_f)
        row1c.pack(fill=tk.X, **pad)
        ttk.Label(row1c, text="Target folder for mirror").pack(side=tk.LEFT)
        self._output_mirror_entry = ttk.Entry(
            row1c, textvariable=self._output_mirror_var, width=40
        )
        self._output_mirror_entry.pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=(8, 4)
        )
        self._mirror_browse_btn = ttk.Button(
            row1c, text="Browse…", command=self._browse_mirror
        )
        self._mirror_browse_btn.pack(side=tk.LEFT)
        self._mirror_var.trace_add("write", lambda *_: self._sync_mirror_widgets())
        self._sync_mirror_widgets()

        row2 = ttk.Frame(root_f)
        row2.pack(fill=tk.X, **pad)
        ttk.Checkbutton(row2, text="Include subfolders", variable=self._recursive_var).pack(
            side=tk.LEFT
        )
        ttk.Checkbutton(
            row2,
            text="Reprocess all (ignore PLY freshness)",
            variable=self._force_all_var,
            command=self._sync_force_skip_widgets,
        ).pack(side=tk.LEFT, padx=(16, 0))

        row3 = ttk.Frame(root_f)
        row3.pack(fill=tk.X, **pad)
        self._limit_cb = ttk.Checkbutton(
            row3, text="Limit splat count", variable=self._limit_var
        )
        self._limit_cb.pack(side=tk.LEFT)
        ttk.Label(row3, text="Max splats").pack(side=tk.LEFT, padx=(8, 4))
        self._max_entry = ttk.Entry(row3, textvariable=self._max_splats_var, width=14)
        self._max_entry.pack(side=tk.LEFT, padx=(0, 8))
        self._skip_cb = ttk.Checkbutton(
            row3,
            text="Skip up-to-date PLY",
            variable=self._skip_up_to_date_var,
        )
        self._skip_cb.pack(side=tk.LEFT, padx=(8, 0))

        row3b = ttk.Frame(root_f)
        row3b.pack(fill=tk.X, **pad)
        self._spz_cb = ttk.Checkbutton(row3b, text="Export SPZ", variable=self._spz_var)
        self._spz_cb.pack(side=tk.LEFT)
        self._spz_only_cb = ttk.Checkbutton(
            row3b,
            text="SPZ from existing PLY only (no new render)",
            variable=self._spz_only_var,
            command=self._on_spz_only_toggle,
        )
        self._spz_only_cb.pack(side=tk.LEFT, padx=(16, 0))
        self._remove_ply_cb = ttk.Checkbutton(
            row3b,
            text="Remove PLY after successful SPZ",
            variable=self._remove_ply_after_spz_var,
        )
        self._remove_ply_cb.pack(side=tk.LEFT, padx=(16, 0))

        self._spz_var.trace_add("write", lambda *_: self._sync_remove_ply_widgets())
        self._limit_var.trace_add("write", lambda *_: self._sync_limit_widgets())
        self._sync_limit_widgets()
        self._sync_force_skip_widgets()
        self._on_spz_only_toggle()
        self._sync_remove_ply_widgets()

        row4 = ttk.Frame(root_f)
        row4.pack(fill=tk.X, **pad)
        self._scan_btn = ttk.Button(row4, text="Start batch", command=self._on_scan)
        self._scan_btn.pack(side=tk.LEFT)
        ttk.Button(row4, text="Stop", command=self._on_stop).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(row4, text="Reset settings", command=self._on_reset_settings).pack(
            side=tk.LEFT, padx=(8, 0)
        )
        self._watch_cb = ttk.Checkbutton(
            row4,
            text="Watch folder (new / changed images)",
            variable=self._watch_var,
            command=self._on_watch_toggle,
        )
        self._watch_cb.pack(side=tk.LEFT, padx=(16, 0))

        srv_row = ttk.Frame(root_f)
        srv_row.pack(fill=tk.X, **pad)
        self._srv_btn = ttk.Button(
            srv_row,
            text="Start server",
            command=self._on_toggle_server,
        )
        self._srv_btn.pack(side=tk.LEFT)
        self._srv_label = ttk.Label(srv_row, text="Server stopped", foreground="#666")
        self._srv_label.pack(side=tk.LEFT, padx=(12, 0))

        row5 = ttk.Frame(root_f)
        row5.pack(fill=tk.X, **pad)
        ttk.Label(row5, text="Batch progress").pack(anchor=tk.W)
        self._progress = ttk.Progressbar(row5, mode="determinate", maximum=100, value=0)
        self._progress.pack(fill=tk.X, pady=(4, 0))
        self._progress_label = ttk.Label(row5, text="—")
        self._progress_label.pack(anchor=tk.W, pady=(2, 0))

        hint = (
            "PLY next to each image when mirror output is off; with mirror on, PLY goes "
            "under the target folder for mirror. Optional Limit splat count uses "
            "PlayCanvas splat-transform (pairwise merge): "
            "https://github.com/playcanvas/splat-transform"
        )
        foot = ttk.Label(root_f, text=hint, wraplength=560, foreground="#666")
        foot.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=(8, 10))

        log_f = ttk.LabelFrame(root_f, text="Log", padding=6)
        log_f.pack(fill=tk.BOTH, expand=True, padx=10, pady=4)
        self._log = tk.Text(log_f, height=14, wrap=tk.WORD, font=("Courier", 11))
        scroll = ttk.Scrollbar(log_f, command=self._log.yview)
        self._log.configure(yscrollcommand=scroll.set)
        self._log.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)

        def _sync_foot_wrap(_event: tk.Event) -> str | None:
            try:
                w = root_f.winfo_width()
            except tk.TclError:
                return None
            if w > 40:
                foot.configure(wraplength=max(280, w - 36))
            return None

        root_f.bind("<Configure>", _sync_foot_wrap)

    def _sync_limit_widgets(self) -> None:
        on = self._limit_var.get()
        self._max_entry.configure(state=tk.NORMAL if on else "disabled")

    def _on_spz_only_toggle(self) -> None:
        if self._spz_only_var.get():
            self._spz_var.set(True)
            self._spz_cb.state(["disabled"])
        else:
            self._spz_cb.state(["!disabled"])
        self._sync_limit_widgets()
        self._sync_remove_ply_widgets()

    def _sync_remove_ply_widgets(self) -> None:
        if self._spz_var.get():
            self._remove_ply_cb.state(["!disabled"])
        else:
            self._remove_ply_after_spz_var.set(False)
            self._remove_ply_cb.state(["disabled"])

    def _sync_force_skip_widgets(self) -> None:
        if self._force_all_var.get():
            self._skip_up_to_date_var.set(False)
            self._skip_cb.state(["disabled"])
        else:
            self._skip_cb.state(["!disabled"])

    def _sync_mirror_widgets(self) -> None:
        on = self._mirror_var.get()
        state = tk.NORMAL if on else tk.DISABLED
        self._output_mirror_entry.configure(state=state)
        self._mirror_browse_btn.configure(state=state)

    def _browse(self) -> None:
        d = filedialog.askdirectory()
        if d:
            self._folder_var.set(d)

    def _browse_mirror(self) -> None:
        d = filedialog.askdirectory()
        if d:
            self._output_mirror_var.set(d)

    def _on_macos_photos_library_toggle(self) -> None:
        if not self._photos_lib_var.get():
            self._mirror_cb.state(["!disabled"])
            return
        lib = default_macos_photos_library_path()
        if not lib.is_dir():
            messagebox.showwarning(
                "Apple Photos library",
                f"Photos Library.photoslibrary not found:\n{lib}",
            )
            self._photos_lib_var.set(False)
            return
        self._folder_var.set(str(lib))
        self._mirror_var.set(True)
        self._mirror_cb.state(["disabled"])

    def _snapshot_opts(self) -> None:
        lim = self._limit_var.get()
        mx = self._parse_max_splats() if lim else None
        if lim and mx is None:
            mx = 500_000
        skip = self._skip_up_to_date_var.get() and not self._force_all_var.get()
        mirror = self._mirror_var.get()
        m_out: Path | None = None
        i_root: Path | None = None
        if mirror:
            mor = self._output_mirror_var.get().strip()
            if mor:
                m_out = Path(mor).expanduser().resolve()
            if sys.platform == "darwin" and self._photos_lib_var.get():
                lib = default_macos_photos_library_path().expanduser().resolve()
                if lib.is_dir():
                    i_root = lib
            else:
                fr = self._folder_var.get().strip()
                if fr:
                    i_root = Path(fr).expanduser().resolve()
        spz = self._spz_var.get()
        spz_only = self._spz_only_var.get()
        rm_ply = self._remove_ply_after_spz_var.get()
        with self._opts_lock:
            self._snap_lim = bool(lim)
            self._snap_max = mx if lim else None
            self._snap_skip = skip
            self._snap_spz = spz
            self._snap_spz_only = spz_only
            self._snap_remove_ply_after_spz = rm_ply
            self._snap_mirror_output = m_out
            self._snap_input_root = i_root

    def _parse_max_splats(self) -> int | None:
        if not self._limit_var.get():
            return None
        raw = self._max_splats_var.get().strip()
        try:
            n = int(raw, 10)
        except ValueError:
            return None
        if n < 1:
            return None
        return min(n, 10_000_000)

    def _limit_options(self) -> tuple[bool, int | None]:
        lim = self._limit_var.get()
        m = self._parse_max_splats()
        if lim and m is None:
            messagebox.showerror(
                "Invalid max splats",
                "Enter a positive integer for max splats when limit is enabled.",
            )
            return False, None
        return lim, m if lim else None

    @staticmethod
    def _format_model_bytes(n: int) -> str:
        if n <= 0:
            return "-"
        mb = n / (1024 * 1024)
        if mb >= 1024:
            gb = mb / 1024
            return f"{round(gb) if gb >= 10 else f'{gb:.1f}'} GB"
        if mb < 100:
            return f"{mb:.1f} MB"
        return f"{round(mb)} MB"

    def _jobs_busy(self) -> bool:
        if self._scan_running:
            return True
        if self._batch_total > 0 and self._batch_done < self._batch_total:
            return True
        if not self._job_q.empty():
            return True
        from sharp_local_batch.core import PREDICT_LOCK

        return PREDICT_LOCK.locked()

    def _sync_model_gated_widgets(self) -> None:
        ready = self._model_state == "ready"
        downloading = self._model_state == "downloading" or self._model_busy_download
        busy = self._jobs_busy()
        self._scan_btn.configure(state="normal" if ready else "disabled")
        if not ready and self._watch_var.get():
            self._watch_var.set(False)
            self._stop_watch()
        self._watch_cb.configure(state="normal" if ready else "disabled")
        if not self._srv_running:
            self._srv_btn.configure(state="normal" if ready else "disabled")
        if downloading:
            if self._model_download_btn.winfo_ismapped():
                self._model_download_btn.pack_forget()
            if self._model_remove_btn.winfo_ismapped():
                self._model_remove_btn.pack_forget()
            if not self._model_cancel_btn.winfo_ismapped():
                self._model_cancel_btn.pack(side=tk.LEFT)
            self._model_cancel_btn.configure(
                state="disabled" if self._model_busy_cancel else "normal"
            )
        elif self._model_state == "ready":
            if self._model_cancel_btn.winfo_ismapped():
                self._model_cancel_btn.pack_forget()
            self._model_download_btn.pack_forget()
            if not self._model_remove_btn.winfo_ismapped():
                self._model_remove_btn.pack(side=tk.LEFT, padx=(8, 0))
            self._model_remove_btn.configure(
                state="disabled" if busy else "normal"
            )
        else:
            if self._model_cancel_btn.winfo_ismapped():
                self._model_cancel_btn.pack_forget()
            if not self._model_download_btn.winfo_ismapped():
                self._model_download_btn.pack(side=tk.LEFT)
            self._model_remove_btn.pack_forget()
            self._model_download_btn.configure(state="normal")
            self._model_download_btn.configure(
                text="Retry" if self._model_state == "error" else "Download"
            )

    def _apply_model_status(self, status: dict) -> None:
        state = str(status.get("state") or "idle")
        self._model_state = state
        self._model_busy_download = state == "downloading"
        if state != "downloading":
            self._model_busy_cancel = False
        total = int(status.get("bytes_total") or 0)
        done = int(status.get("bytes_downloaded") or 0)
        percent = int(status.get("percent") or 0)
        size_exact = bool(status.get("size_exact"))
        size_label = self._format_model_bytes(total) if total > 0 else "~2.6 GB"
        if total > 0 and not size_exact and state != "ready":
            size_label = f"~{size_label}"

        if state == "ready":
            self._model_status_label.config(
                text=f"SHARP model ready · {size_label}"
            )
            self._model_progress.pack_forget()
        elif state == "downloading":
            msg = str(status.get("message") or "")
            if "cancel" in msg.lower():
                self._model_status_label.config(text=msg)
            else:
                left = (
                    f"{self._format_model_bytes(done)} / {self._format_model_bytes(total)}"
                    if total > 0
                    else self._format_model_bytes(done)
                )
                self._model_status_label.config(
                    text=f"Downloading SHARP model… {percent}% · {left}"
                )
            self._model_progress.configure(value=max(0, min(100, percent)))
            if not self._model_progress.winfo_ismapped():
                self._model_progress.pack(
                    fill=tk.X, pady=(4, 0), after=self._model_status_label
                )
        elif state == "error":
            err = status.get("error") or "unknown error"
            self._model_status_label.config(text=f"Download failed: {err}")
            self._model_progress.pack_forget()
        else:
            self._model_status_label.config(
                text=(
                    f"Download the SHARP model ({size_label}) before "
                    "batch or server work."
                )
            )
            self._model_progress.pack_forget()
        self._sync_model_gated_widgets()

    def _poll_model_status(self) -> None:
        try:
            from sharp_local_batch.model_download import get_download_manager

            status = get_download_manager().status_dict()
            self._apply_model_status(status)
        except Exception as exc:
            self._model_status_label.config(
                text=f"Model status unavailable: {exc}"
            )
        try:
            self.root.after(1000, self._poll_model_status)
        except tk.TclError:
            pass

    def _on_model_download(self) -> None:
        from sharp_local_batch.model_download import get_download_manager

        self._model_busy_download = True
        self._model_download_btn.configure(state="disabled")
        self._model_status_label.config(text="Starting SHARP model download…")
        mgr = get_download_manager()
        mgr.ensure_download_async()
        self._apply_model_status(mgr.status_dict())

    def _on_model_cancel(self) -> None:
        from sharp_local_batch.model_download import get_download_manager

        self._model_busy_cancel = True
        self._model_cancel_btn.configure(state="disabled")
        self._model_status_label.config(text="Cancelling download…")
        mgr = get_download_manager()
        mgr.cancel_download()
        self._log_line("--- SHARP model download cancelled ---")
        self._apply_model_status(mgr.status_dict())

    def _on_model_remove(self) -> None:
        if not messagebox.askyesno(
            "Remove SHARP model?",
            "Delete the downloaded SHARP checkpoint from this computer?\n\n"
            "You can download it again later. Batch and server generation "
            "will be blocked until then.",
            default=messagebox.NO,
        ):
            return
        from sharp_local_batch.model_download import get_download_manager

        try:
            result = get_download_manager().delete_checkpoint()
        except RuntimeError as exc:
            messagebox.showwarning("Remove model", str(exc))
            return
        freed = int(result.get("deleted_bytes") or 0)
        note = (
            f" Freed {self._format_model_bytes(freed)}."
            if freed > 0
            else ""
        )
        self._log_line(f"--- SHARP model removed.{note} ---")
        self._apply_model_status(get_download_manager().status_dict())

    def _on_scan(self) -> None:
        if self._model_state != "ready":
            messagebox.showwarning(
                "SHARP model",
                "Download the SHARP model first (see the SHARP model section above).",
            )
            return
        ok, max_s = self._limit_options()
        if not ok:
            return
        if sys.platform == "darwin" and self._photos_lib_var.get():
            root = default_macos_photos_library_path().expanduser().resolve()
            if not root.is_dir():
                messagebox.showerror(
                    "Apple Photos library",
                    f"Photos Library.photoslibrary not found:\n{root}",
                )
                return
            self._folder_var.set(str(root))
        else:
            raw = self._folder_var.get().strip()
            if not raw:
                messagebox.showwarning("Folder", "Choose a folder first.")
                return
            root = Path(raw).expanduser().resolve()
            if not root.is_dir():
                messagebox.showerror("Folder", f"Not a directory: {root}")
                return

        mirror_out: Path | None = None
        if self._mirror_var.get():
            mor = self._output_mirror_var.get().strip()
            if not mor:
                messagebox.showwarning(
                    "Mirror output",
                    "Choose a target folder for mirror, or turn off mirroring.",
                )
                return
            mirror_out = Path(mor).expanduser().resolve()
            if mirror_out == root:
                messagebox.showerror(
                    "Mirror output",
                    "Target folder for mirror must differ from the source folder.",
                )
                return
            mirror_out.mkdir(parents=True, exist_ok=True)

        if is_photos_library_bundle(root) and mirror_out is None:
            messagebox.showerror("Apple Photos library", PHOTOS_LIBRARY_MIRROR_HELP)
            return

        jobs, total_found = scan_jobs(
            root,
            self._recursive_var.get(),
            force_all=self._force_all_var.get(),
            mirror_output_root=mirror_out,
            export_spz=self._spz_var.get(),
            spz_only=self._spz_only_var.get(),
        )
        if not jobs:
            if total_found == 0:
                messagebox.showinfo(
                    "Scan",
                    "No supported images were found in the scan folder.\n\n"
                    "If you use iCloud Photos, originals may not be on disk yet — "
                    "open a photo in Photos to download it, or enable "
                    "Photos → Settings → iCloud → Download Originals to this Mac.",
                )
            else:
                if self._spz_only_var.get() and self._spz_var.get():
                    messagebox.showinfo(
                        "Scan",
                        "No work found: every image already has an .spz that is up "
                        "to date with its PLY. Turn on Reprocess all to refresh .spz "
                        "files anyway.",
                    )
                else:
                    messagebox.showinfo(
                        "Scan",
                        "No images need processing (PLY already up to date).",
                    )
            return

        self._snapshot_opts()
        self._scan_running = True
        self._batch_total = len(jobs)
        self._batch_done = 0
        self._batch_start_time = time.perf_counter()
        self._progress.configure(maximum=self._batch_total, value=0, mode="determinate")
        self._progress_label.config(text=f"Queued {len(jobs)} job(s)…")
        self._log_line(f"--- Batch started: {len(jobs)} job(s) ---")

        for p in jobs:
            self._job_q.put(p)

    def _on_stop(self) -> None:
        try:
            while True:
                self._job_q.get_nowait()
        except queue.Empty:
            pass
        self._scan_running = False
        self._batch_total = 0
        self._batch_done = 0
        self._progress.configure(value=0)
        self._progress_label.config(text="Stopped")
        self._log_line("--- Stop: queue cleared ---")

    def _stop_http_server(self, *, update_ui: bool = True) -> None:
        from app import set_gui_log_sink

        httpd = self._srv_httpd
        self._srv_httpd = None
        if httpd is not None:
            shutdown = getattr(httpd, "shutdown", None)
            if callable(shutdown):
                try:
                    shutdown()
                except Exception:
                    pass
        t = self._srv_thread
        if t is not None and t.is_alive():
            t.join(timeout=2.0)
        self._srv_thread = None
        if httpd is not None:
            close = getattr(httpd, "server_close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass
        set_gui_log_sink(None)
        self._srv_running = False
        if update_ui:
            self._srv_btn.config(text="Start server")
            self._srv_label.config(text="Server stopped", foreground="#666")
            self._log_line("--- Server stopped ---")
            self._sync_model_gated_widgets()

    def _on_toggle_server(self) -> None:
        if self._srv_running:
            self._stop_http_server()
            return
        if self._model_state != "ready":
            messagebox.showwarning(
                "SHARP model",
                "Download the SHARP model first before starting the HTTP server.",
            )
            return

        from app import (
            OUTPUTS_DIR,
            app,
            create_wsgi_server,
            set_gui_log_sink,
            _suppress_flask_startup_noise,
        )

        port = 8765
        url = f"http://127.0.0.1:{port}"
        limit_default = bool(self._limit_var.get())
        max_s: int | None = None
        try:
            n = int(self._max_splats_var.get().strip())
            if n >= 1:
                max_s = n
        except ValueError:
            max_s = None

        _suppress_flask_startup_noise()
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        app.config["DEFAULT_LIMIT_SPLATS"] = limit_default
        app.config["DEFAULT_MAX_SPLATS"] = max_s
        try:
            httpd = create_wsgi_server("127.0.0.1", port)
        except OSError as exc:
            messagebox.showerror(
                "Server",
                f"Could not start server on {url}:\n{exc}",
            )
            return

        def _server_log_sink(text: str) -> None:
            self.root.after(0, lambda t=text: self._log_line(t))

        self._srv_httpd = httpd
        self._srv_running = True
        self._srv_btn.config(text="Stop server")
        self._srv_label.config(text=f"Running: {url}", foreground="#2a2")
        self._log_line(f"--- Server starting at {url} ---")
        if limit_default and max_s is not None:
            self._log_line(f"Splat limit: {max_s:,}")

        def _run() -> None:
            set_gui_log_sink(_server_log_sink)
            try:
                httpd.serve_forever()
            finally:
                set_gui_log_sink(None)

        self._srv_thread = threading.Thread(
            target=_run, name="sharp-http-server", daemon=True
        )
        self._srv_thread.start()

    def _on_watch_toggle(self) -> None:
        if self._watch_var.get():
            if self._model_state != "ready":
                messagebox.showwarning(
                    "SHARP model",
                    "Download the SHARP model first before watching a folder.",
                )
                self._watch_var.set(False)
                return
            self._start_watch()
        else:
            self._stop_watch()

    def _start_watch(self) -> None:
        ok, _mx = self._limit_options()
        if not ok:
            self._watch_var.set(False)
            return
        if sys.platform == "darwin" and self._photos_lib_var.get():
            root = default_macos_photos_library_path().expanduser().resolve()
            if not root.is_dir():
                messagebox.showerror(
                    "Watch",
                    f"Photos Library.photoslibrary not found:\n{root}",
                )
                self._watch_var.set(False)
                return
            self._folder_var.set(str(root))
        else:
            raw = self._folder_var.get().strip()
            if not raw:
                messagebox.showwarning("Watch", "Choose a folder first.")
                self._watch_var.set(False)
                return
            root = Path(raw).expanduser().resolve()
            if not root.is_dir():
                messagebox.showerror("Watch", f"Not a directory: {root}")
                self._watch_var.set(False)
                return

        if self._mirror_var.get():
            mor = self._output_mirror_var.get().strip()
            if not mor:
                messagebox.showwarning(
                    "Watch",
                    "Choose a target folder for mirror, or disable mirroring.",
                )
                self._watch_var.set(False)
                return
            if Path(mor).expanduser().resolve() == root:
                messagebox.showerror(
                    "Watch",
                    "Target folder for mirror must differ from the watched folder.",
                )
                self._watch_var.set(False)
                return

        mirror_out_watch: Path | None = None
        if self._mirror_var.get():
            mirror_out_watch = Path(self._output_mirror_var.get().strip()).expanduser().resolve()
        if is_photos_library_bundle(root) and mirror_out_watch is None:
            messagebox.showerror("Watch", PHOTOS_LIBRARY_MIRROR_HELP)
            self._watch_var.set(False)
            return

        if self._watch is not None:
            return

        if is_photos_library_bundle(root):
            messagebox.showinfo(
                "Watch",
                "Watching Photos Library.photoslibrary can fire often while the Photos "
                "app updates its database. Exporting to a normal folder is gentler if "
                "you hit issues.",
            )

        self._snapshot_opts()

        def enqueue(p: Path) -> None:
            path = p.resolve()

            def push() -> None:
                self._snapshot_opts()
                self._job_q.put(path)

            self._safe_after(0, push)

        self._watch = WatchController(
            root,
            enqueue,
            debounce_sec=0.6,
            recursive=self._recursive_var.get(),
        )
        self._watch.start()
        self._log_line(f"--- Watch started: {root} ---")

    def _stop_watch(self) -> None:
        if self._watch is not None:
            self._watch.stop()
            self._watch = None
            self._log_line("--- Watch stopped ---")

    def _worker_loop(self) -> None:
        while not self._quit_app.is_set():
            try:
                item = self._job_q.get(timeout=0.35)
            except queue.Empty:
                continue
            p = item
            with self._opts_lock:
                lim = self._snap_lim
                max_s = self._snap_max
                skip = self._snap_skip
                spz = self._snap_spz
                spz_only = self._snap_spz_only
                rm_ply = self._snap_remove_ply_after_spz
                m_out = self._snap_mirror_output
                i_root = self._snap_input_root

            try:
                ply_target = output_ply_path_for_job(
                    p,
                    mirror_output_root=m_out,
                    mirror_input_root=i_root,
                )
            except ValueError as e:
                r = PlySidecarResult(
                    ok=False,
                    image_path=p,
                    ply_path=sidecar_ply_path(p),
                    message=str(e),
                )
                self._safe_after(0, self._on_job_done, r)
                continue

            r = update_ply_sidecar(
                p,
                skip_up_to_date=skip,
                limit_splats=lim,
                max_splats=max_s if lim else None,
                ply_output_path=ply_target,
                export_spz=spz,
                spz_only=spz_only,
                remove_ply_after_spz=rm_ply,
            )

            self._safe_after(0, self._on_job_done, r)

    def _on_job_done(self, r: PlySidecarResult) -> None:
        t_note = (
            f" ({format_elapsed_for_log(r.elapsed_seconds)})"
            if r.elapsed_seconds is not None
            else ""
        )
        if r.skipped:
            self._log_line(f"[skip] {r.image_path.name} — {r.message}{t_note}")
        elif r.ok:
            self._processed_session += 1
            self._log_line(f"[ok] {r.image_path.name} — {r.message}{t_note}")
        else:
            self._log_line(f"[err] {r.image_path.name} — {r.message}{t_note}")

        if self._batch_total > 0:
            self._batch_done += 1
            self._progress["value"] = self._batch_done
            self._progress_label.config(
                text=f"{self._batch_done} / {self._batch_total} files"
            )
            if self._batch_done >= self._batch_total:
                n_jobs = self._batch_total
                batch_elapsed = time.perf_counter() - self._batch_start_time
                self._batch_total = 0
                self._batch_done = 0
                self._scan_running = False
                self._progress.configure(value=0, maximum=100)
                self._progress_label.config(
                    text=(
                        f"Batch done in {format_elapsed_for_log(batch_elapsed, decimals=1)} · "
                        f"session processed: {self._processed_session}"
                    )
                )
                self._log_line(
                    f"--- Batch finished: {n_jobs} job(s) in "
                    f"{format_elapsed_for_log(batch_elapsed)} ---"
                )

    def _log_line(self, text: str) -> None:
        self._log.insert(tk.END, text + "\n")
        self._log.see(tk.END)

    def _safe_after(self, delay_ms: int, func: Callable[..., object], *args: object) -> None:
        """Schedule on the Tk main loop; no-op if the root is already destroyed."""
        try:
            self.root.after(delay_ms, func, *args)
        except tk.TclError:
            pass

    def _on_close(self) -> None:
        self._persist_gui_settings()
        self._stop_watch()
        self._stop_http_server(update_ui=False)
        self._quit_app.set()
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    SharpBatchGui().run()
