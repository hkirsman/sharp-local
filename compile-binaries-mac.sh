#!/usr/bin/env bash
# Build both standalone bundles on macOS:
#   dist/SharpBatch/SharpBatch.app  - Qt GUI + CLI batch tool
#   dist/SharpWeb/SharpWeb.app      - Flask web UI (open http://127.0.0.1:8765)
#
# Creates .venv and installs deps if needed (same idea as compile-binaries-win.bat).
# See docs/mac-setup.md for full developer setup instructions.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

if ! command -v python3 >/dev/null 2>&1; then
  echo "ERROR: python3 not found on PATH." >&2
  echo "Install Python 3.13 (or 3.11) - see docs/mac-setup.md" >&2
  exit 1
fi

PYTHON=".venv/bin/python3"

if [[ ! -x "$ROOT/$PYTHON" ]]; then
  echo "Creating .venv..."
  python3 -m venv .venv
fi

echo "Installing project dependencies into .venv..."
"$ROOT/$PYTHON" -m pip install -U pip
"$ROOT/$PYTHON" -m pip install -e ./ml-sharp -r requirements.txt

echo "Installing PyInstaller into .venv (if needed)..."
"$ROOT/$PYTHON" -m pip install -q -U pyinstaller

echo "Checking vendored splat-transform helper..."
"$ROOT/$PYTHON" -c "
from pathlib import Path
import sys
sys.path.insert(0, str(Path('packaging').resolve()))
from splat_transform_bundle import splat_transform_binary
src, _ = splat_transform_binary(Path('.').resolve())
print(f'OK: {src}')
"

echo "Generating application icons..."
"$ROOT/$PYTHON" packaging/brand_icon.py

echo "Building SharpBatch..."
rm -rf dist/SharpBatch dist/SharpBatch.app
"$ROOT/$PYTHON" -m PyInstaller --noconfirm packaging/sharp_batch.spec
# PyInstaller emits dist/SharpBatch/ (bare binary + _internal) and
# dist/SharpBatch.app at the dist root. Keep only the .app.
rm -rf dist/SharpBatch
mkdir -p dist/SharpBatch
mv dist/SharpBatch.app dist/SharpBatch/

echo "Building SharpWeb..."
rm -rf dist/SharpWeb dist/SharpWeb.app
"$ROOT/$PYTHON" -m PyInstaller --noconfirm packaging/sharp_web.spec
rm -rf dist/SharpWeb
mkdir -p dist/SharpWeb
mv dist/SharpWeb.app dist/SharpWeb/

VERSION=$("$ROOT/$PYTHON" -c "from sharp_local_batch._version import __version__; print(__version__)")

echo "Packaging archives..."
(
  cd "$ROOT/dist"
  rm -f "SharpBatch-${VERSION}-mac.zip" "SharpWeb-${VERSION}-mac.zip"
  zip -r -y -q "SharpBatch-${VERSION}-mac.zip" SharpBatch
  zip -r -y -q "SharpWeb-${VERSION}-mac.zip" SharpWeb
)

echo ""
echo "========================================================================"
echo "Build finished OK - version ${VERSION}."
echo ""
echo "Batch tool (GUI/CLI):"
echo "  $ROOT/dist/SharpBatch/SharpBatch.app"
echo "  Folder: $ROOT/dist/SharpBatch/"
echo "  CLI: $ROOT/dist/SharpBatch/SharpBatch.app/Contents/MacOS/SharpBatch --cli ..."
echo ""
echo "Web UI (Flask server - open http://127.0.0.1:8765 after starting):"
echo "  $ROOT/dist/SharpWeb/SharpWeb.app"
echo "  Folder: $ROOT/dist/SharpWeb/"
echo ""
echo "Archives (upload these to the GitHub release):"
echo "  $ROOT/dist/SharpBatch-${VERSION}-mac.zip"
echo "  $ROOT/dist/SharpWeb-${VERSION}-mac.zip"
echo ""
echo "NOTE: bundles built locally are not notarised. To open them the first"
echo "  time: right-click -> Open -> Open, or run:"
echo "  xattr -dr com.apple.quarantine dist/SharpBatch/ dist/SharpWeb/"
echo "========================================================================"
