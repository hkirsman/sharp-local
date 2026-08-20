"""PyInstaller entrypoint: run the Flask web UI (``app.py``)."""

from __future__ import annotations

import logging
import multiprocessing


def main() -> None:
    import argparse

    from sharp_local_batch.logging_config import ensure_stdio

    # Before any import that may touch logging or torch progress bars.
    ensure_stdio()
    logging.basicConfig(level=logging.INFO)
    from sharp_local_batch._version import __version__
    from app import LOGGER, OUTPUTS_DIR, app

    parser = argparse.ArgumentParser(description="Sharp Local web server")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8765, help="Port (default: 8765)")
    parser.add_argument(
        "--headless",
        action="store_true",
        help="No launch dialog or browser; log the URL and run the server only",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Show the launch dialog but do not open the browser automatically",
    )
    args = parser.parse_args()

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Sharp Local web %s - http://%s:%d", __version__, args.host, args.port)
    from app import WEB_LOG_PATH

    if WEB_LOG_PATH is not None:
        LOGGER.info("Log file: %s", WEB_LOG_PATH)

    def _start_server() -> None:
        app.run(host=args.host, port=args.port, debug=False, threaded=True)

    if args.headless:
        _start_server()
        return

    from sharp_local_batch.web_launch_dialog import run_server_with_launch_dialog

    run_server_with_launch_dialog(
        host=args.host,
        port=args.port,
        version=__version__,
        start_server=_start_server,
        open_browser=not args.no_browser,
    )


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
