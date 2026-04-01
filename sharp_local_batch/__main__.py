"""Run ``python -m sharp_local_batch`` (GUI default) or ``--cli`` for headless."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from sharp_local_batch.batch_runner import scan_jobs
from sharp_local_batch.core import process_image_to_sidecar_ply


def _cli_main() -> int:
    p = argparse.ArgumentParser(description="SHARP batch: PLY next to each image.")
    p.add_argument(
        "--folder",
        type=Path,
        required=True,
        help="Root folder to scan",
    )
    p.add_argument(
        "--recursive",
        action="store_true",
        help="Include subfolders",
    )
    p.add_argument(
        "--force-all",
        action="store_true",
        help="Reprocess every image (ignore PLY freshness)",
    )
    p.add_argument(
        "--limit-splats",
        action="store_true",
        help="Decimate with splat-transform after full PLY",
    )
    p.add_argument(
        "--max-splats",
        type=int,
        default=500_000,
        help="Target splat count when --limit-splats (default 500000)",
    )
    args = p.parse_args()

    root = args.folder.expanduser().resolve()
    if not root.is_dir():
        print(f"Not a directory: {root}", file=sys.stderr)
        return 1

    jobs = scan_jobs(root, args.recursive, force_all=args.force_all)
    if not jobs:
        print("No images need processing.")
        return 0

    max_s = args.max_splats if args.limit_splats else None
    if args.limit_splats and args.max_splats < 1:
        print("--max-splats must be >= 1", file=sys.stderr)
        return 1

    n = len(jobs)
    exit_code = 0
    for i, path in enumerate(jobs, start=1):
        print(f"[{i}/{n}] {path}", flush=True)
        r = process_image_to_sidecar_ply(
            path,
            limit_splats=args.limit_splats,
            max_splats=max_s,
        )
        tag = "ok" if r.ok else "err"
        if r.skipped:
            tag = "skip"
        print(f"    {tag}: {r.message}", flush=True)
        if not r.ok and not r.skipped:
            exit_code = 1
    return exit_code


_GUI_HELP = """\
No batch GUI available.

Fix (pick one):
  • pip install PySide6-Essentials
    Then: python -m sharp_local_batch
    (Qt UI; smaller than full PySide6; no system Tcl/Tk.)
  • brew install python-tk@3.14
    Match `python3 --version`, recreate .venv, then the Tk fallback can load.
  • Headless: python -m sharp_local_batch --cli --folder /path/to/photos --recursive
"""


def main() -> None:
    if len(sys.argv) >= 2 and sys.argv[1] == "--cli":
        sys.argv.pop(1)
        raise SystemExit(_cli_main())
    try:
        from sharp_local_batch.gui_qt import main as gui_main

        gui_main()
        return
    except ImportError:
        pass
    try:
        from sharp_local_batch.gui import main as gui_main

        gui_main()
    except ImportError as e:
        err = str(e).lower()
        if "_tkinter" in err or "tkinter" in err:
            print(_GUI_HELP, file=sys.stderr)
            raise SystemExit(1) from e
        raise


if __name__ == "__main__":
    main()
