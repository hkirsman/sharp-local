"""Run ``python -m sharp_local_batch`` (GUI default) or ``--cli`` for headless."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from sharp_local_batch.logging_config import ensure_stderr_info_logging

ensure_stderr_info_logging()

from sharp_local_batch.batch_runner import scan_jobs
from sharp_local_batch.core import (
    PHOTOS_LIBRARY_MIRROR_HELP,
    format_elapsed_for_log,
    is_photos_library_bundle,
    output_ply_path_for_job,
    update_ply_sidecar,
)


def _format_model_bytes(n: int) -> str:
    if n <= 0:
        return "-"
    mb = n / (1024 * 1024)
    if mb >= 1024:
        gb = mb / 1024
        return f"{round(gb) if gb >= 10 else f'{gb:.1f}'} GB"
    if mb < 100:
        return f"{mb:.1f} MB"
    return f"{round(mb)} MB"


def _ensure_model_ready_for_cli() -> int:
    """Download the SHARP checkpoint synchronously if missing. Returns exit code."""
    from sharp_local_batch.core import ensure_sharp_imports
    from sharp_local_batch.model_download import get_download_manager

    try:
        ensure_sharp_imports()
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    mgr = get_download_manager()
    if mgr.is_ready():
        status = mgr.status_dict()
        total = int(status.get("bytes_total") or 0)
        print(
            f"SHARP model ready ({_format_model_bytes(total)}).",
            flush=True,
        )
        return 0

    print("SHARP model missing - downloading…", flush=True)
    last_pct = -1

    def _progress(done: int, total: int, percent: int) -> None:
        nonlocal last_pct
        if percent == last_pct and percent not in (0, 100):
            return
        last_pct = percent
        if total > 0:
            print(
                f"  {percent}% · {_format_model_bytes(done)} / {_format_model_bytes(total)}",
                flush=True,
            )
        else:
            print(f"  {_format_model_bytes(done)}…", flush=True)

    try:
        mgr.download_sync(progress_cb=_progress)
    except Exception as exc:
        print(f"SHARP model download failed: {exc}", file=sys.stderr)
        return 1
    status = mgr.status_dict()
    total = int(status.get("bytes_total") or 0)
    print(f"SHARP model ready ({_format_model_bytes(total)}).", flush=True)
    return 0


def _download_model_main() -> int:
    return _ensure_model_ready_for_cli()


def _remove_model_main() -> int:
    from sharp_local_batch.core import ensure_sharp_imports
    from sharp_local_batch.model_download import get_download_manager

    try:
        ensure_sharp_imports()
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    mgr = get_download_manager()
    try:
        result = mgr.delete_checkpoint()
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    freed = int(result.get("deleted_bytes") or 0)
    if freed > 0:
        print(f"Removed SHARP model ({_format_model_bytes(freed)}).")
    else:
        print("No SHARP model checkpoint found to remove.")
    return 0


def _cli_main() -> int:
    from sharp_local_batch._version import __version__

    p = argparse.ArgumentParser(
        description="SHARP batch: PLY beside each image, or under --output-root when mirroring.",
    )
    p.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    p.add_argument(
        "--download-model",
        action="store_true",
        help="Download the SHARP checkpoint (~2.6 GB) and exit",
    )
    p.add_argument(
        "--remove-model",
        action="store_true",
        help="Delete the cached SHARP checkpoint and exit",
    )
    p.add_argument(
        "--folder",
        type=Path,
        required=False,
        help=(
            "Root path to scan (a directory of images, or a macOS "
            "Photos Library.photoslibrary bundle - the latter requires --output-root)"
        ),
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
    p.add_argument(
        "--no-export-spz",
        action="store_false",
        dest="export_spz",
        help="Skip Niantic .spz export (SPZ is exported by default alongside PLY)",
    )
    p.add_argument(
        "--spz-only",
        action="store_true",
        help=(
            "SPZ from existing PLY when possible: if a PLY exists, skip SHARP and "
            "only refresh .spz (optional decimate first). If no PLY yet, run full "
            "SHARP then .spz. Implies SPZ export (incompatible with --no-export-spz)"
        ),
    )
    p.add_argument(
        "--keep-ply",
        action="store_false",
        dest="remove_ply_after_spz",
        help=(
            "Keep the .ply file after a successful .spz export (by default "
            "the PLY is deleted to save space; next run without a PLY will "
            "run SHARP again). Incompatible with --no-export-spz"
        ),
    )
    p.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help=(
            "Write PLY files under this folder; under ~ mirror path from home, else "
            "relative to --folder (default: next to each image)"
        ),
    )
    args = p.parse_args()

    if args.download_model and args.remove_model:
        print(
            "Use only one of --download-model or --remove-model.",
            file=sys.stderr,
        )
        return 1
    if args.download_model:
        return _download_model_main()
    if args.remove_model:
        return _remove_model_main()

    if args.folder is None:
        p.error("--folder is required unless --download-model or --remove-model")

    if args.spz_only and not args.export_spz:
        print(
            "--spz-only cannot be combined with --no-export-spz "
            "(SPZ-only mode always exports .spz).",
            file=sys.stderr,
        )
        return 1

    if args.remove_ply_after_spz and not args.export_spz:
        print(
            "PLY removal requires SPZ export "
            "(do not pass --no-export-spz).",
            file=sys.stderr,
        )
        return 1

    root = args.folder.expanduser().resolve()
    if not root.is_dir():
        print(f"Not a directory: {root}", file=sys.stderr)
        return 1

    mirror_out: Path | None = None
    if args.output_root is not None:
        mirror_out = args.output_root.expanduser().resolve()
        if mirror_out == root:
            print("--output-root must differ from --folder", file=sys.stderr)
            return 1
        mirror_out.mkdir(parents=True, exist_ok=True)

    if is_photos_library_bundle(root) and mirror_out is None:
        print(PHOTOS_LIBRARY_MIRROR_HELP, file=sys.stderr)
        print("Use: --output-root /path/outside/library", file=sys.stderr)
        return 1

    model_rc = _ensure_model_ready_for_cli()
    if model_rc != 0:
        return model_rc

    jobs, total_found = scan_jobs(
        root,
        args.recursive,
        force_all=args.force_all,
        mirror_output_root=mirror_out,
        export_spz=args.export_spz,
        spz_only=args.spz_only,
    )
    if not jobs:
        if total_found == 0:
            print(
                "No supported images found under the source folder.",
                file=sys.stderr,
            )
        else:
            if args.spz_only:
                print(
                    "No work: every image has .spz up to date with its PLY. "
                    "Use --force-all to refresh every .spz.",
                )
            else:
                print("No images need processing (PLY already up to date).")
        return 0

    max_s = args.max_splats if args.limit_splats else None
    if args.limit_splats and args.max_splats < 1:
        print("--max-splats must be >= 1", file=sys.stderr)
        return 1

    n = len(jobs)
    exit_code = 0
    batch_t0 = time.perf_counter()
    for i, path in enumerate(jobs, start=1):
        print(f"[{i}/{n}] {path}", flush=True)
        try:
            ply_target = output_ply_path_for_job(
                path,
                mirror_output_root=mirror_out,
                mirror_input_root=root if mirror_out is not None else None,
            )
        except ValueError as e:
            print(f"    err: {e}", flush=True)
            exit_code = 1
            continue
        r = update_ply_sidecar(
            path,
            skip_up_to_date=not args.force_all,
            limit_splats=args.limit_splats,
            max_splats=max_s,
            ply_output_path=ply_target,
            export_spz=args.export_spz,
            spz_only=args.spz_only,
            remove_ply_after_spz=args.remove_ply_after_spz,
        )
        tag = "ok" if r.ok else "err"
        if r.skipped:
            tag = "skip"
        t_note = (
            f" ({format_elapsed_for_log(r.elapsed_seconds)})"
            if r.elapsed_seconds is not None
            else ""
        )
        print(f"    {tag}: {r.message}{t_note}", flush=True)
        if not r.ok and not r.skipped:
            exit_code = 1
    batch_elapsed = time.perf_counter() - batch_t0
    print(
        f"Batch finished: {n} item(s) in {format_elapsed_for_log(batch_elapsed)}.",
        flush=True,
    )
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


def _is_expected_missing_qt(exc: ImportError) -> bool:
    """True only when PySide6 (or a PySide6.* submodule) is absent - not other import bugs."""
    if isinstance(exc, ModuleNotFoundError):
        name = (exc.name or "").lower()
        if name == "pyside6" or name.startswith("pyside6."):
            return True
    return "pyside6" in str(exc).lower()


def main() -> None:
    if len(sys.argv) == 2 and sys.argv[1] in ("--version", "-V"):
        from sharp_local_batch._version import __version__

        print(f"sharp-local-batch {__version__}")
        raise SystemExit(0)
    # Top-level model helpers (no --cli required).
    if len(sys.argv) >= 2 and sys.argv[1] in ("--download-model", "--remove-model"):
        if sys.argv[1] == "--download-model":
            raise SystemExit(_download_model_main())
        raise SystemExit(_remove_model_main())
    if len(sys.argv) >= 2 and sys.argv[1] == "--cli":
        sys.argv.pop(1)
        raise SystemExit(_cli_main())
    try:
        from sharp_local_batch.gui_qt import main as gui_main

        gui_main()
        return
    except ImportError as e:
        if not _is_expected_missing_qt(e):
            raise
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
