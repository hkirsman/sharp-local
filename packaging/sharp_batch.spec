# -*- mode: python ; coding: utf-8 -*-
#
# Build a standalone app for sharp_local_batch (Qt GUI + CLI).
#
# From the repository root (with .venv activated and deps installed):
#   pip install pyinstaller
#   pyinstaller packaging/sharp_batch.spec
#
# Output: dist/SharpBatch/  (Windows: binary + _internal; macOS: SharpBatch.app via BUNDLE).
#
# Expect a large bundle (PyTorch + Qt). The SHARP checkpoint still downloads on first inference
# unless you ship it separately and point TORCH_HOME / cache.
#
import pathlib
import sys

from PyInstaller.utils.hooks import collect_submodules, copy_metadata

block_cipher = None

REPO = pathlib.Path(SPECPATH).resolve().parent
_PACKAGING = pathlib.Path(SPECPATH).resolve()

if sys.platform == "win32":
    ICON = (_PACKAGING / "sharp-local.ico").resolve()
elif sys.platform == "darwin":
    ICON = (_PACKAGING / "sharp-local.icns").resolve()
else:
    ICON = None

if ICON is not None and not ICON.is_file():
    raise SystemExit(f"Missing {ICON} - run: python packaging/brand_icon.py")

datas = [
    # App version (read by sharp_local_batch/_version.py at import time).
    (str(REPO / "version.txt"), "."),
]
if (REPO / "ml-sharp" / "src").is_dir():
    datas.append((str(REPO / "ml-sharp" / "src"), "ml-sharp/src"))

# imageio reads importlib.metadata.version("imageio") at import time; frozen apps omit dist-info otherwise.
datas += copy_metadata("imageio")
try:
    datas += copy_metadata("imageio-ffmpeg")
except Exception:
    pass

# Omit sharp_local_batch.gui: frozen bundle targets PySide6 only; Tk needs _tkinter
# and breaks analysis/build on Homebrew Pythons without Tcl/Tk.
hiddenimports = (
    collect_submodules("sharp")
    + [
        "sharp_local_batch",
        "sharp_local_batch._version",
        "sharp_local_batch.core",
        "sharp_local_batch.batch_runner",
        "sharp_local_batch.gui_qt",
        "sharp_local_batch.logging_config",
        "watchdog",
        "watchdog.observers",
        "watchdog.events",
        "plyfile",
        "PIL",
        "PIL.Image",
        "gaussforge",
        "imageio",
        "imageio.v2",
        "imageio.core",
        "imageio.plugins",
    ]
)

a = Analysis(
    [str(REPO / "packaging" / "entry_batch.py")],
    pathex=[str(REPO)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="SharpBatch",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    # Windowed app so Finder double-click shows a Dock icon (CLI still works
    # when run from Terminal via Contents/MacOS/SharpBatch --cli ...).
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(ICON) if ICON is not None else None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="SharpBatch",
)

if sys.platform == "darwin":
    _app_version = (REPO / "version.txt").read_text(encoding="utf-8").strip()
    app = BUNDLE(
        coll,
        name="SharpBatch.app",
        icon=str(ICON),
        bundle_identifier="io.sharplocal.batch",
        version=_app_version,
        info_plist={
            "CFBundleDisplayName": "Sharp Local batch",
            "CFBundleName": "SharpBatch",
            "CFBundleShortVersionString": _app_version,
            "CFBundleVersion": _app_version,
            "NSHighResolutionCapable": True,
            "NSPrincipalClass": "NSApplication",
        },
    )
