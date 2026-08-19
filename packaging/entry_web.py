"""PyInstaller entrypoint: run the Flask web UI (``app.py``)."""

from __future__ import annotations

import logging
import multiprocessing


def main() -> None:
    import argparse

    logging.basicConfig(level=logging.INFO)
    from sharp_local_batch._version import __version__
    from app import LOGGER, OUTPUTS_DIR, app

    parser = argparse.ArgumentParser(description="Sharp Local web server")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8765, help="Port (default: 8765)")
    args = parser.parse_args()

    LOGGER.info("Sharp Local web %s - http://%s:%d", __version__, args.host, args.port)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
