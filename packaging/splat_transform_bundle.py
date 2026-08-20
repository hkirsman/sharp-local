# -*- mode: python ; coding: utf-8 -*-
"""Shared helpers for Sharp Local PyInstaller specs."""

from __future__ import annotations

import pathlib
import sys


def splat_transform_binary(repo: pathlib.Path) -> tuple[str, str]:
    """Return ``(src_path, dest_dir)`` for Analysis.binaries, or exit if missing.

    Dest ``.`` places the file at the root of ``sys._MEIPASS`` so
    ``resolve_splat_transform_exe`` can find it when frozen.
    """
    vendor = repo / "vendor" / "splat-transform"
    version_file = vendor / "VERSION"
    if not version_file.is_file():
        raise SystemExit(
            f"Missing {version_file} - run: ./packaging/compile-splat-transform.sh"
        )
    version = version_file.read_text(encoding="utf-8").strip()
    if not version:
        raise SystemExit(f"Empty {version_file}")

    if sys.platform == "darwin":
        name = f"splat-transform-{version}-darwin-arm64"
    elif sys.platform == "win32":
        name = f"splat-transform-{version}-windows-x64.exe"
    else:
        raise SystemExit(
            f"No vendored splat-transform binary for platform {sys.platform!r}. "
            "Supported: darwin (arm64), win32 (x64)."
        )

    src = vendor / name
    if not src.is_file():
        raise SystemExit(
            f"Missing {src}\n"
            "Rebuild helpers with: ./packaging/compile-splat-transform.sh\n"
            "(requires bun + npm)"
        )
    return (str(src), ".")
