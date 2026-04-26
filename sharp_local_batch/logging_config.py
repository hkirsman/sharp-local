"""One-shot stderr logging for CLI/GUI when the root logger has no handlers."""

from __future__ import annotations

import logging
import sys


def ensure_stderr_info_logging() -> None:
    """If nothing configured the root logger yet, attach a single INFO StreamHandler."""
    root = logging.getLogger()
    if root.handlers:
        return
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s:%(name)s:%(message)s",
        stream=sys.stderr,
    )
