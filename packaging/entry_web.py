"""PyInstaller entrypoint: run the Flask web UI (``app.py``)."""

from __future__ import annotations

import logging
import multiprocessing


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    from app import OUTPUTS_DIR, app

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    app.run(host="127.0.0.1", port=8765, debug=False, threaded=True)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
