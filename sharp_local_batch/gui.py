"""Tkinter UI for folder batch + optional filesystem watch."""

from __future__ import annotations

import queue
import sys
import threading
import time
import tkinter as tk
from collections.abc import Callable
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from sharp_local_batch.batch_runner import WatchController, scan_jobs
from sharp_local_batch.core import (
    PHOTOS_LIBRARY_MIRROR_HELP,
    PlySidecarResult,
    default_macos_photos_library_path,
    is_photos_library_bundle,
    output_ply_path_for_job,
    sidecar_ply_path,
    update_ply_sidecar,
)


class SharpBatchGui:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("sharp_local_batch")
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

        self._build_ui()

        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self) -> None:
        pad = {"padx": 10, "pady": 4}
        root_f = ttk.Frame(self.root, padding=10)
        root_f.pack(fill=tk.BOTH, expand=True)

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
        ttk.Button(row4, text="Scan & queue jobs", command=self._on_scan).pack(
            side=tk.LEFT
        )
        ttk.Button(row4, text="Stop", command=self._on_stop).pack(side=tk.LEFT, padx=(8, 0))
        self._watch_cb = ttk.Checkbutton(
            row4,
            text="Watch folder (new / changed images)",
            variable=self._watch_var,
            command=self._on_watch_toggle,
        )
        self._watch_cb.pack(side=tk.LEFT, padx=(16, 0))

        row5 = ttk.Frame(root_f)
        row5.pack(fill=tk.X, **pad)
        ttk.Label(row5, text="Batch progress").pack(anchor=tk.W)
        self._progress = ttk.Progressbar(row5, mode="determinate", maximum=100, value=0)
        self._progress.pack(fill=tk.X, pady=(4, 0))
        self._progress_label = ttk.Label(row5, text="—")
        self._progress_label.pack(anchor=tk.W, pady=(2, 0))

        hint = (
            "PLY next to each image when mirror output is off; with mirror on, PLY goes "
            "under the target folder for mirror. Optional cap uses splat-transform "
            "(npm i -g @playcanvas/splat-transform)."
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

    def _on_scan(self) -> None:
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
        self._log_line(f"--- Scan: {len(jobs)} job(s) ---")

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

    def _on_watch_toggle(self) -> None:
        if self._watch_var.get():
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
            f" ({r.elapsed_seconds:.2f}s)"
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
                        f"Batch done in {batch_elapsed:.1f}s · "
                        f"session processed: {self._processed_session}"
                    )
                )
                self._log_line(
                    f"--- Batch finished: {n_jobs} job(s) in {batch_elapsed:.2f}s ---"
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
        self._stop_watch()
        self._quit_app.set()
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    SharpBatchGui().run()
