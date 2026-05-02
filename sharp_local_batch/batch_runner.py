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
    needs_spz_refresh,
    output_ply_path_for_job,
    sidecar_ply_path,
    update_ply_sidecar,
)


def scan_jobs(
    root: Path,
    recursive: bool,
    *,
    force_all: bool,
    mirror_output_root: Optional[Path] = None,
    export_spz: bool = True,
    spz_only: bool = False,
) -> tuple[list[Path], int]:
    """Images under ``root`` that need work (or all images if ``force_all``).

    When ``export_spz`` is set, files whose PLY is current but ``.spz`` sidecar
    is missing are still queued so the worker can top up the SPZ.

    When ``spz_only`` is set (requires ``export_spz``), images **without** a
    target PLY are still queued so the worker can run the full SHARP pipeline
    once.  Images that already have a PLY are queued when ``.spz`` is missing,
    stale vs that PLY, or when ``force_all`` is set.  The worker then converts
    PLY → SPZ without SHARP only when the PLY already exists.

    Returns ``(jobs, total_supported_images)`` where *total_supported_images* is the
    count of supported files found before the PLY-freshness filter — useful to tell
    “nothing on disk” from “everything already processed”.
    """
    scan_root = effective_batch_scan_root(root)
    paths = list_image_paths(scan_root, recursive)
    total = len(paths)
    effective_spz_only = bool(spz_only and export_spz)

    if effective_spz_only:
        root_r = root.resolve()
        if mirror_output_root is None:

            def _spz_only_need(p: Path) -> bool:
                ply = sidecar_ply_path(p)
                if not ply.is_file():
                    # PLY missing — check if SPZ already exists and is current
                    # (PLY may have been removed after a previous SPZ export).
                    spz = ply.with_suffix(".spz")
                    if spz.is_file():
                        try:
                            if spz.stat().st_mtime >= p.stat().st_mtime:
                                return False
                        except OSError:
                            pass
                    return True
                if force_all:
                    return True
                return needs_spz_refresh(ply)

            return [p for p in paths if _spz_only_need(p)], total

        out_r = mirror_output_root.resolve()

        def _spz_only_need_m(p: Path) -> bool:
            try:
                target = mirrored_ply_path(p, root_r, out_r)
            except ValueError:
                return True
            if not target.is_file():
                # PLY missing — check if SPZ already exists and is current.
                spz = target.with_suffix(".spz")
                if spz.is_file():
                    try:
                        if spz.stat().st_mtime >= p.stat().st_mtime:
                            return False
                    except OSError:
                        pass
                return True
            if force_all:
                return True
            return needs_spz_refresh(target)

        return [p for p in paths if _spz_only_need_m(p)], total

    if force_all:
        return paths, total
    root_r = root.resolve()
    if mirror_output_root is None:
        return [
            p for p in paths if needs_ply_refresh(p, require_spz=export_spz)
        ], total
    out_r = mirror_output_root.resolve()

    def _needs_work(p: Path) -> bool:
        try:
            target = mirrored_ply_path(p, root_r, out_r)
        except ValueError:
            return True
        return needs_ply_refresh(p, target, require_spz=export_spz)

    return [p for p in paths if _needs_work(p)], total


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

    def on_moved(self, event: object) -> None:
        self._handle(event, path_attr="dest_path")

    def _handle(self, event: object, *, path_attr: str = "src_path") -> None:
        if getattr(event, "is_directory", False):
            return
        src = getattr(event, path_attr, None)
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
    export_spz: bool = True,
    spz_only: bool = False,
    remove_ply_after_spz: bool = True,
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
        on_result(
            update_ply_sidecar(
                image_path,
                skip_up_to_date=skip_up_to_date,
                limit_splats=limit_splats,
                max_splats=max_splats,
                ply_output_path=ply_target,
                export_spz=export_spz,
                spz_only=spz_only,
                remove_ply_after_spz=remove_ply_after_spz,
            )
        )
