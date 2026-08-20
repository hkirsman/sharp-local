"""One-shot stderr (+ optional file) logging for CLI/GUI."""

from __future__ import annotations

import logging
import sys
from pathlib import Path


def sharp_local_data_dir() -> Path:
    """Per-user data root (logs, etc.) for Sharp Local apps."""
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "SharpLocal"
    if sys.platform == "win32":
        local = __import__("os").environ.get("LOCALAPPDATA")
        if local:
            return Path(local) / "SharpLocal"
        return Path.home() / "AppData" / "Local" / "SharpLocal"
    return Path.home() / ".local" / "share" / "SharpLocal"


def batch_log_path() -> Path:
    return sharp_local_data_dir() / "logs" / "sharp-batch.log"


def web_log_path() -> Path:
    return sharp_local_data_dir() / "logs" / "sharp-web.log"


def attach_file_handler(log: logging.Logger, log_file_name: str) -> Path | None:
    """Append a FileHandler to ``log`` if one for this path is not already present."""
    try:
        log_dir = sharp_local_data_dir() / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / log_file_name
    except OSError:
        return None

    resolved = str(log_path.resolve())
    for existing in log.handlers:
        if isinstance(existing, logging.FileHandler):
            try:
                if str(Path(existing.baseFilename).resolve()) == resolved:
                    return log_path
            except OSError:
                continue

    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s:%(name)s:%(message)s")
    )
    log.addHandler(handler)
    log.info("Logging to %s", log_path)
    return log_path


def ensure_stderr_info_logging(*, log_file_name: str | None = None) -> Path | None:
    """Attach INFO handlers if the root logger is bare.

    When ``log_file_name`` is set (or the process is frozen), also append to a
    file under Application Support / LocalAppData so double-clicked apps leave
    a readable trail. Returns the log file path when file logging is enabled.
    """
    frozen = bool(getattr(sys, "frozen", False))
    name = log_file_name or ("sharp-batch.log" if frozen else None)
    log_path: Path | None = None
    if name:
        try:
            log_dir = sharp_local_data_dir() / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            log_path = log_dir / name
        except OSError:
            log_path = None

    root = logging.getLogger()
    if root.handlers:
        return log_path

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if log_path is not None:
        handlers.append(logging.FileHandler(log_path, encoding="utf-8"))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s:%(name)s:%(message)s",
        handlers=handlers,
    )
    if log_path is not None:
        logging.getLogger(__name__).info("Logging to %s", log_path)
    return log_path
