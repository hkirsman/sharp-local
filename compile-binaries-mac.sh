#!/usr/bin/env bash
# Build both standalone bundles on macOS:
#   dist/SharpBatch/SharpBatch   — Qt GUI + CLI batch tool
#   dist/SharpWeb/SharpWeb       — Flask web UI (open http://127.0.0.1:8765)
#
# Prerequisites: .venv already created and deps installed.
#   Run ./bootstrap.sh first if you haven't yet.
#   See docs/mac-setup.md for full developer setup instructions.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

PYTHON=".venv/bin/python3"

if [[ ! -x "$ROOT/$PYTHON" ]]; then
  echo "ERROR: .venv not found at $ROOT/.venv" >&2
  echo "Create the venv and install dependencies first — see docs/mac-setup.md" >&2
  exit 1
fi

echo "Installing PyInstaller into .venv (if needed)..."
"$ROOT/$PYTHON" -m pip install -q -U pyinstaller

echo "Building SharpBatch..."
"$ROOT/$PYTHON" -m PyInstaller packaging/sharp_batch.spec

echo "Building SharpWeb..."
"$ROOT/$PYTHON" -m PyInstaller packaging/sharp_web.spec

echo ""
echo "========================================================================"
echo "Build finished OK."
echo ""
echo "Batch tool (GUI/CLI):"
echo "  $ROOT/dist/SharpBatch/SharpBatch"
echo "  Folder: $ROOT/dist/SharpBatch/"
echo ""
echo "Web UI (Flask server — open http://127.0.0.1:8765 after starting):"
echo "  $ROOT/dist/SharpWeb/SharpWeb"
echo "  Folder: $ROOT/dist/SharpWeb/"
echo ""
echo "Ship each app by zipping the whole folder above (include _internal/)."
echo ""
echo "NOTE: bundles built locally are not notarised. To open them the first"
echo "  time: right-click → Open → Open, or run:"
echo "  xattr -dr com.apple.quarantine dist/SharpBatch/"
echo "========================================================================"
