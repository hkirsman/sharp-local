"""PyInstaller entrypoint: run ``python -m sharp_local_batch`` (GUI or --cli)."""

from __future__ import annotations

import multiprocessing


def main() -> None:
    from sharp_local_batch.__main__ import main as batch_main

    batch_main()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
