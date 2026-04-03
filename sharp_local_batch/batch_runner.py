"""Job queue worker and optional filesystem watch (watchdog)."""

from __future__ import annotations

import queue
import threading
from pathlib import Path
from typing import Callable, Optional

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from sharp_local_batch.core import (
    PlySidecarResult,
    effective_batch_scan_root,
    is_supported_image,
    list_image_paths,
    mirrored_ply_path,
    needs_ply_refresh,
    output_ply_path_for_job,
    process_image_to_sidecar_ply,
    sidecar_ply_path,
)


def scan_jobs(
    root: Path,
    recursive: bool,
    *,
    force_all: bool,
    mirror_output_root: Optional[Path] = None,
) -> list[Path]:
    """Images under ``root`` that need work (or all images if ``force_all``)."""
    scan_root = effective_batch_scan_root(root)
    paths = list_image_paths(scan_root, recursive)
    if force_all:
        return paths
    root_r = root.resolve()
    if mirror_output_root is None:
        return [p for p in paths if needs_ply_refresh(p)]
    out_r = mirror_output_root.resolve()
    return [
        p
        for p in paths
        if needs_ply_refresh(p, mirrored_ply_path(p, root_r, out_r))
    ]


class DebouncedScheduler:
    """Call ``on_ready(path)`` after ``delay_sec`` quiet period per path."""

    def __init__(self, delay_sec: float, on_ready: Callable[[Path], None]) -> None:
        self.delay_sec = delay_sec
        self.on_ready = on_ready
        self._timers: dict[str, threading.Timer] = {}
        self._lock = threading.Lock()

    def schedule(self, path: Path) -> None:
        key = str(path.resolve())
        with self._lock:
            old = self._timers.pop(key, None)
            if old is not None:
                old.cancel()

            def fire() -> None:
                with self._lock:
                    self._timers.pop(key, None)
                self.on_ready(path)

            t = threading.Timer(self.delay_sec, fire)
            t.daemon = True
            self._timers[key] = t
            t.start()

    def cancel_all(self) -> None:
        with self._lock:
            for t in self._timers.values():
                t.cancel()
            self._timers.clear()


class _ImageWatchHandler(FileSystemEventHandler):
    def __init__(self, scheduler: DebouncedScheduler) -> None:
        self._scheduler = scheduler

    def on_created(self, event: object) -> None:
        self._handle(event)

    def on_modified(self, event: object) -> None:
        self._handle(event)

    def _handle(self, event: object) -> None:
        if getattr(event, "is_directory", False):
            return
        src = getattr(event, "src_path", None)
        if not src:
            return
        p = Path(src)
        if not is_supported_image(p):
            return
        if any(part.startswith(".") for part in p.parts):
            return
        self._scheduler.schedule(p)


class WatchController:
    """Watchdog observer; debounces and enqueues paths via ``enqueue``."""

    def __init__(
        self,
        root: Path,
        enqueue: Callable[[Path], None],
        *,
        debounce_sec: float = 0.6,
        recursive: bool = True,
    ) -> None:
        self._root = root.resolve()
        self._recursive = recursive
        self._scheduler = DebouncedScheduler(debounce_sec, enqueue)
        self._handler = _ImageWatchHandler(self._scheduler)
        self._observer: Optional[Observer] = None

    def start(self) -> None:
        if self._observer is not None:
            return
        obs = Observer()
        watch_path = effective_batch_scan_root(self._root)
        obs.schedule(self._handler, str(watch_path), recursive=self._recursive)
        obs.start()
        self._observer = obs

    def stop(self) -> None:
        self._scheduler.cancel_all()
        if self._observer is not None:
            self._observer.stop()
            self._observer.join(timeout=5.0)
            self._observer = None


def worker_loop(
    job_q: "queue.Queue[Optional[Path]]",
    stop_event: threading.Event,
    *,
    limit_splats: bool,
    max_splats: Optional[int],
    skip_up_to_date: bool,
    mirror_output_root: Optional[Path],
    mirror_input_root: Optional[Path],
    on_result: Callable[[PlySidecarResult], None],
) -> None:
    """Drain ``job_q`` until ``None`` sentinel or ``stop_event``."""
    while True:
        if stop_event.is_set():
            try:
                while True:
                    job_q.get_nowait()
            except queue.Empty:
                pass
            break
        try:
            item = job_q.get(timeout=0.35)
        except queue.Empty:
            continue
        if item is None:
            break
        if stop_event.is_set():
            break
        image_path = item.resolve()
        try:
            ply_target = output_ply_path_for_job(
                image_path,
                mirror_output_root=mirror_output_root,
                mirror_input_root=mirror_input_root,
            )
        except ValueError as e:
            on_result(
                PlySidecarResult(
                    ok=False,
                    image_path=image_path,
                    ply_path=sidecar_ply_path(image_path),
                    message=str(e),
                )
            )
            continue
        if skip_up_to_date and not needs_ply_refresh(image_path, ply_target):
            on_result(
                PlySidecarResult(
                    ok=True,
                    image_path=image_path,
                    ply_path=ply_target,
                    message="Skipped (PLY up to date)",
                    skipped=True,
                )
            )
            continue
        on_result(
            process_image_to_sidecar_ply(
                image_path,
                limit_splats=limit_splats,
                max_splats=max_splats,
                ply_output_path=ply_target,
            )
        )
