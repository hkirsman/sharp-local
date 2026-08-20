"""Background download manager for the Apple SHARP checkpoint.

Mirrors Vaela's AI stereo download pattern for a single torch-hub URL:
progress is thread-safe, inference never fetches from the network, and
delete only touches an allowlisted cache path.
"""

from __future__ import annotations

import logging
import threading
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, Optional
from urllib.parse import urlparse

LOGGER = logging.getLogger(__name__)

# Documented size when Content-Length is unavailable (README / Apple CDN).
FALLBACK_BYTES = int(2.6 * 1024 * 1024 * 1024)

_USER_AGENT = "SharpLocal/1.0"


class _DownloadCancelled(Exception):
    """Internal: ``cancel_download()`` interrupted the fetch."""


class ModelNotReadyError(RuntimeError):
    """Raised when inference is attempted before the checkpoint is on disk."""

    def __init__(self, message: str = "model_not_ready") -> None:
        super().__init__(message)
        self.code = "model_not_ready"


def default_model_url() -> str:
    """Return Apple SHARP default checkpoint URL (requires ml-sharp on path)."""
    from sharp.cli.predict import DEFAULT_MODEL_URL

    return str(DEFAULT_MODEL_URL)


def checkpoint_filename(url: Optional[str] = None) -> str:
    target = url or default_model_url()
    name = Path(urlparse(target).path).name
    if not name or not name.endswith(".pt"):
        raise RuntimeError(f"Unexpected SHARP model URL basename: {name!r}")
    return name


def checkpoint_cache_path(url: Optional[str] = None) -> Path:
    """Path where ``torch.hub.load_state_dict_from_url`` stores the checkpoint."""
    from torch.hub import get_dir

    return Path(get_dir()) / "checkpoints" / checkpoint_filename(url)


def checkpoint_tmp_path(url: Optional[str] = None) -> Path:
    dest = checkpoint_cache_path(url)
    return dest.with_suffix(dest.suffix + ".tmp")


def _fetch_remote_bytes(url: str) -> int:
    try:
        req = urllib.request.Request(
            url,
            method="HEAD",
            headers={"User-Agent": _USER_AGENT},
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            length = resp.headers.get("Content-Length")
            if length:
                return int(length)
    except Exception:
        return 0
    return 0


class ModelDownloadManager:
    """Process-wide SHARP checkpoint download / status / delete."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._state = "idle"  # idle | downloading | ready | error
        self._percent = 0
        self._bytes_downloaded = 0
        self._bytes_total = 0
        self._size_exact = False
        self._message = ""
        self._error: Optional[str] = None
        self._thread: Optional[threading.Thread] = None
        self._remote_probed = False
        self._known_total = 0
        self._probe_thread: Optional[threading.Thread] = None
        self._cancel = threading.Event()
        self._active_resp: Optional[Any] = None
        self.refresh_local_state()

    def refresh_local_state(self) -> None:
        """Update ready/idle from disk without starting a download."""
        path = checkpoint_cache_path()
        with self._lock:
            if self._state == "downloading":
                return
            if path.is_file():
                size = path.stat().st_size
                self._state = "ready"
                self._percent = 100
                self._bytes_downloaded = size
                self._bytes_total = size
                self._size_exact = True
                self._known_total = size
                self._message = "SHARP model ready"
                self._error = None
            elif self._state == "error":
                # Keep last error until disk is ready or a new download starts.
                return
            else:
                self._state = "idle"
                self._percent = 0
                self._bytes_downloaded = 0
                if self._known_total <= 0:
                    self._bytes_total = FALLBACK_BYTES
                    self._size_exact = False
                else:
                    self._bytes_total = self._known_total
                    self._size_exact = True
                self._message = ""

    def is_ready(self) -> bool:
        self.refresh_local_state()
        with self._lock:
            return self._state == "ready"

    def is_downloading(self) -> bool:
        with self._lock:
            return self._state == "downloading"

    def _ensure_remote_size(self) -> None:
        """Kick off a background size probe; never block status on the network."""
        with self._lock:
            if self._remote_probed or self._state in ("downloading", "ready"):
                return
            if self._known_total > 0:
                self._remote_probed = True
                return
            if getattr(self, "_probe_thread", None) is not None and self._probe_thread.is_alive():
                return
            self._probe_thread = threading.Thread(
                target=self._probe_remote_size_bg,
                name="sharp-model-size-probe",
                daemon=True,
            )
            self._probe_thread.start()

    def _probe_remote_size_bg(self) -> None:
        try:
            remote = _fetch_remote_bytes(default_model_url())
        except Exception:
            remote = 0
        with self._lock:
            self._remote_probed = True
            if remote > 0:
                self._known_total = remote
                if self._state not in ("ready", "downloading"):
                    self._bytes_total = remote
                    self._size_exact = True

    def status_dict(self) -> Dict[str, Any]:
        self.refresh_local_state()
        self._ensure_remote_size()
        path = checkpoint_cache_path()
        with self._lock:
            payload: Dict[str, Any] = {
                "state": self._state,
                "percent": self._percent,
                "bytes_downloaded": self._bytes_downloaded,
                "bytes_total": self._bytes_total,
                "size_exact": self._size_exact,
                "message": self._message,
                "error": self._error,
                "cache_path": str(path) if path.is_file() else None,
                "url": default_model_url(),
            }
            return payload

    def ensure_download_async(self) -> None:
        """Start a background download if the checkpoint is not already ready."""
        self.refresh_local_state()
        with self._lock:
            if self._state == "ready":
                return
            if self._thread is not None and self._thread.is_alive():
                return
            self._cancel.clear()
            self._state = "downloading"
            self._error = None
            self._message = "Starting SHARP model download…"
            self._percent = 0
            self._bytes_downloaded = 0
            if self._known_total > 0:
                self._bytes_total = self._known_total
                self._size_exact = True
            else:
                self._bytes_total = FALLBACK_BYTES
                self._size_exact = False
            self._thread = threading.Thread(
                target=self._run_download,
                name="sharp-model-download",
                daemon=True,
            )
            self._thread.start()

    def download_sync(
        self,
        *,
        progress_cb: Optional[Callable[[int, int, int], None]] = None,
    ) -> None:
        """Blocking download (CLI). Raises on failure."""
        self.refresh_local_state()
        with self._lock:
            if self._state == "ready":
                return
            if self._thread is not None and self._thread.is_alive():
                raise RuntimeError("A model download is already in progress")
            self._cancel.clear()
            self._state = "downloading"
            self._error = None
            self._message = "Downloading SHARP model…"
            self._percent = 0
            self._bytes_downloaded = 0
            if self._known_total > 0:
                self._bytes_total = self._known_total
                self._size_exact = True
            else:
                self._bytes_total = FALLBACK_BYTES
                self._size_exact = False
        try:
            self._download_file(progress_cb=progress_cb)
        except _DownloadCancelled:
            self._note_cancelled()
            raise RuntimeError("Download cancelled")
        except Exception as exc:
            if self._cancel.is_set():
                self._note_cancelled()
                raise RuntimeError("Download cancelled") from exc
            with self._lock:
                self._state = "error"
                self._error = str(exc)
                self._message = f"Download failed: {exc}"
                self._percent = 0
            raise
        self.refresh_local_state()

    def delete_checkpoint(self) -> Dict[str, Any]:
        """Remove the cached checkpoint. Refuses while downloading or predicting."""
        with self._lock:
            if self._state == "downloading":
                raise RuntimeError(
                    "Cannot delete SHARP model while a download is in progress"
                )
            if self._thread is not None and self._thread.is_alive():
                raise RuntimeError(
                    "Cannot delete SHARP model while a download is in progress"
                )

        from sharp_local_batch.core import PREDICT_LOCK, unload_predictor

        if not PREDICT_LOCK.acquire(blocking=False):
            raise RuntimeError(
                "Cannot delete SHARP model while inference is running"
            )
        try:
            unload_predictor()
            dest = checkpoint_cache_path()
            tmp = checkpoint_tmp_path()
            deleted_bytes = 0
            deleted_paths: list[str] = []
            if dest.is_file():
                deleted_bytes += dest.stat().st_size
                dest.unlink()
                deleted_paths.append(str(dest))
            if tmp.is_file():
                deleted_bytes += tmp.stat().st_size
                tmp.unlink()
                deleted_paths.append(str(tmp))
        finally:
            PREDICT_LOCK.release()

        with self._lock:
            self._state = "idle"
            self._percent = 0
            self._bytes_downloaded = 0
            self._message = ""
            self._error = None
            if self._known_total > 0:
                self._bytes_total = self._known_total
                self._size_exact = True
            else:
                self._bytes_total = FALLBACK_BYTES
                self._size_exact = False
                self._remote_probed = False

        self.refresh_local_state()
        LOGGER.info(
            "SHARP checkpoint deleted (%s bytes, paths=%s)",
            deleted_bytes,
            deleted_paths,
        )
        return {
            "deleted_bytes": deleted_bytes,
            "deleted_paths": deleted_paths,
        }

    def cancel_download(self) -> Dict[str, Any]:
        """Stop an in-progress download and discard the partial tmp file.

        Idempotent: if nothing is downloading, returns current status.
        State stays ``downloading`` until the worker exits, then ``idle``.
        """
        resp: Optional[Any] = None
        with self._lock:
            if self._state == "downloading":
                self._cancel.set()
                self._message = "Cancelling download…"
                resp = self._active_resp
        if resp is not None:
            try:
                resp.close()
            except Exception:
                pass
        return self.status_dict()

    def _cleanup_partial_download(self) -> None:
        tmp = checkpoint_tmp_path()
        if tmp.is_file():
            try:
                tmp.unlink()
            except OSError:
                pass

    def _note_cancelled(self) -> None:
        self._cleanup_partial_download()
        with self._lock:
            self._state = "idle"
            self._percent = 0
            self._bytes_downloaded = 0
            self._error = None
            self._message = ""
            if self._known_total > 0:
                self._bytes_total = self._known_total
                self._size_exact = True
            else:
                self._bytes_total = FALLBACK_BYTES
                self._size_exact = False
        LOGGER.info("SHARP model download cancelled")

    def _raise_if_cancelled(self) -> None:
        if self._cancel.is_set():
            raise _DownloadCancelled()

    def _run_download(self) -> None:
        try:
            self._download_file()
            with self._lock:
                self._state = "ready"
                self._percent = 100
                self._message = "SHARP model ready"
                self._error = None
            LOGGER.info("SHARP model download complete")
        except _DownloadCancelled:
            self._note_cancelled()
        except Exception as exc:
            if self._cancel.is_set():
                self._note_cancelled()
            else:
                LOGGER.exception("SHARP model download failed")
                with self._lock:
                    self._state = "error"
                    self._error = str(exc)
                    self._message = f"Download failed: {exc}"
                    self._percent = 0
        finally:
            with self._lock:
                self._thread = None
            self.refresh_local_state()

    def _download_file(
        self,
        *,
        progress_cb: Optional[Callable[[int, int, int], None]] = None,
    ) -> None:
        url = default_model_url()
        dest = checkpoint_cache_path()
        tmp = checkpoint_tmp_path()
        dest.parent.mkdir(parents=True, exist_ok=True)
        if tmp.is_file():
            try:
                tmp.unlink()
            except OSError:
                pass

        self._raise_if_cancelled()
        req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
        with urllib.request.urlopen(req, timeout=120) as resp:
            with self._lock:
                self._active_resp = resp
            try:
                self._raise_if_cancelled()
                total = int(resp.headers.get("Content-Length") or 0)
                with self._lock:
                    if total > 0:
                        self._known_total = total
                        self._bytes_total = total
                        self._size_exact = True
                    else:
                        total = self._bytes_total or FALLBACK_BYTES
                    if not self._cancel.is_set():
                        self._message = "Downloading SHARP model…"
                downloaded = 0
                with open(tmp, "wb") as out:
                    while True:
                        self._raise_if_cancelled()
                        chunk = resp.read(1024 * 256)
                        if not chunk:
                            break
                        out.write(chunk)
                        downloaded += len(chunk)
                        percent = (
                            min(99, int(100 * downloaded / total))
                            if total > 0
                            else 0
                        )
                        with self._lock:
                            self._bytes_downloaded = downloaded
                            self._percent = percent
                            if self._bytes_total < downloaded:
                                self._bytes_total = downloaded
                        if progress_cb is not None:
                            progress_cb(downloaded, total, percent)
            finally:
                with self._lock:
                    if self._active_resp is resp:
                        self._active_resp = None
        self._raise_if_cancelled()
        tmp.replace(dest)
        size = dest.stat().st_size
        with self._lock:
            self._bytes_downloaded = size
            self._bytes_total = size
            self._known_total = size
            self._size_exact = True
            self._percent = 100
            self._state = "ready"
            self._message = "SHARP model ready"
            self._error = None
        if progress_cb is not None:
            progress_cb(size, size, 100)


_manager: Optional[ModelDownloadManager] = None
_manager_lock = threading.Lock()


def get_download_manager() -> ModelDownloadManager:
    """Return (or create) the process-wide download manager singleton."""
    global _manager
    with _manager_lock:
        if _manager is None:
            _manager = ModelDownloadManager()
        return _manager
