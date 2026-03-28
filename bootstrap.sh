#!/usr/bin/env bash
# Create .venv and install ml-sharp + Flask (avoids Homebrew PEP 668 "externally-managed-environment").
# ml-sharp is a git submodule at ./ml-sharp (see .gitmodules).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

if ! command -v git >/dev/null 2>&1; then
  echo "git is required (ml-sharp is a submodule)." >&2
  exit 1
fi

if ! git -C "$ROOT" rev-parse --git-dir >/dev/null 2>&1; then
  echo "Run bootstrap from a git clone of this repo (ml-sharp is a submodule)." >&2
  exit 1
fi

if [[ ! -f "$ROOT/.gitmodules" ]] || ! grep -qE '^[[:space:]]*path = ml-sharp[[:space:]]*$' "$ROOT/.gitmodules"; then
  echo "Missing submodule config for ml-sharp (.gitmodules). This repository expects ml-sharp as a submodule." >&2
  exit 1
fi

# Always sync: if ml-sharp exists but was checked out manually or on the wrong
# commit, a pyproject-only check would skip fixing it. update --init is cheap when
# the tree already matches the superproject’s recorded gitlink.
echo "Syncing ml-sharp submodule to pinned commit..." >&2
git -C "$ROOT" submodule update --init ml-sharp

if [[ ! -f "$ROOT/ml-sharp/pyproject.toml" ]]; then
  echo "ml-sharp still missing after submodule update. Try: git submodule update --init ml-sharp" >&2
  exit 1
fi

ML_SHARP="$ROOT/ml-sharp"

if [[ ! -d .venv ]]; then
  python3 -m venv .venv
fi
# shellcheck source=/dev/null
source .venv/bin/activate
python3 -m pip install -U pip
python3 -m pip install -e "$ML_SHARP" -r requirements.txt

echo ""
echo "Done. Run:"
echo "  cd $ROOT && source .venv/bin/activate && python app.py"
echo "Then open http://127.0.0.1:8765"
