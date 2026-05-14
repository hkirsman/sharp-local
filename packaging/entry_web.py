"""PyInstaller entrypoint: run the Flask web UI (``app.py``)."""

from __future__ import annotations

import logging
import multiprocessing


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    from sharp_local_batch._version import __version__
    from app import LOGGER, OUTPUTS_DIR, app

    LOGGER.info("Sharp Local web %s — http://127.0.0.1:8765", __version__)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    app.run(host="127.0.0.1", port=8765, debug=False, threaded=True)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
