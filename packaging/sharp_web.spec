# -*- mode: python ; coding: utf-8 -*-
#
# Build a standalone Flask web UI (app.py → browser at http://127.0.0.1:8765).
#
# From the repository root (with .venv and deps installed):
#   pip install pyinstaller
#   pyinstaller packaging/sharp_web.spec
#
# Output: dist/SharpWeb/  (Windows: binary + _internal; macOS: SharpWeb.app via BUNDLE).
#
import pathlib
import sys

from PyInstaller.utils.hooks import collect_data_files, collect_submodules, copy_metadata

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
if (REPO / "static").is_dir():
    datas.append((str(REPO / "static"), "static"))

datas += copy_metadata("imageio")
try:
    datas += copy_metadata("imageio-ffmpeg")
except Exception:
    pass

for pkg in ("flask", "werkzeug", "jinja2"):
    try:
        datas += collect_data_files(pkg)
    except Exception:
        pass

hiddenimports = collect_submodules("sharp") + [
    "app",
    "sharp_local_batch",
    "sharp_local_batch._version",
    "sharp_local_batch.core",
    "sharp_local_batch.logging_config",
    "sharp_local_batch.web_launch_dialog",
    "plyfile",
    "PIL",
    "PIL.Image",
    "gaussforge",
    "imageio",
    "imageio.v2",
    "imageio.core",
    "imageio.plugins",
    "flask",
    "werkzeug",
    "jinja2",
    "PySide6",
    "PySide6.QtCore",
    "PySide6.QtGui",
    "PySide6.QtWidgets",
]

a = Analysis(
    [str(REPO / "packaging" / "entry_web.py")],
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
    name="SharpWeb",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    # No console window - launch dialog shows the URL (use --headless for logs-only).
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
    name="SharpWeb",
)

if sys.platform == "darwin":
    _app_version = (REPO / "version.txt").read_text(encoding="utf-8").strip()
    app = BUNDLE(
        coll,
        name="SharpWeb.app",
        icon=str(ICON),
        bundle_identifier="io.vaela.sharplocal.web",
        version=_app_version,
        info_plist={
            "CFBundleDisplayName": "Sharp Local web",
            "CFBundleName": "SharpWeb",
            "CFBundleShortVersionString": _app_version,
            "CFBundleVersion": _app_version,
            "NSHighResolutionCapable": True,
            "NSPrincipalClass": "NSApplication",
        },
    )
