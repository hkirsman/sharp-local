# -*- mode: python ; coding: utf-8 -*-
#
# Build a standalone app for sharp_local_batch (Qt GUI + CLI).
#
# From the repository root (with .venv activated and deps installed):
#   pip install pyinstaller
#   pyinstaller packaging/sharp_batch.spec
#
# Output: dist/SharpBatch/  (macOS: SharpBatch.app if you add --windowed; this spec uses console for CLI.)
#
# Expect a large bundle (PyTorch + Qt). The SHARP checkpoint still downloads on first inference
# unless you ship it separately and point TORCH_HOME / cache.
#
import pathlib

from PyInstaller.utils.hooks import collect_submodules, copy_metadata

block_cipher = None

REPO = pathlib.Path(SPECPATH).resolve().parent

datas = []
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
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
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
