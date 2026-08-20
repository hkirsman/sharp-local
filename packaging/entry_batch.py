"""PyInstaller entrypoint: run ``python -m sharp_local_batch`` (GUI or --cli)."""

from __future__ import annotations

import multiprocessing


def main() -> None:
    import logging

    from sharp_local_batch._version import __version__
    from sharp_local_batch.logging_config import ensure_stderr_info_logging

    ensure_stderr_info_logging(log_file_name="sharp-batch.log")
    logging.getLogger("SharpBatch").info("Sharp Local batch %s", __version__)

    from sharp_local_batch.__main__ import main as batch_main

    batch_main()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
