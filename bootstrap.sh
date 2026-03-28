#!/usr/bin/env bash
# Create .venv and install ml-sharp + Flask (avoids Homebrew PEP 668 "externally-managed-environment").
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

ML_SHARP=""
if [[ -d "$ROOT/ml-sharp" ]]; then
  ML_SHARP="$ROOT/ml-sharp"
elif [[ -d "$ROOT/../ml-sharp" ]]; then
  ML_SHARP="$(cd "$ROOT/../ml-sharp" && pwd)"
else
  echo "Clone Apple ml-sharp into this folder (./ml-sharp) or next to experiments/ (../ml-sharp)." >&2
  exit 1
fi

python3 -m venv .venv
# shellcheck source=/dev/null
source .venv/bin/activate
python3 -m pip install -U pip
python3 -m pip install -e "$ML_SHARP" -r requirements.txt

echo ""
echo "Done. Run:"
echo "  cd $ROOT && source .venv/bin/activate && python app.py"
echo "Then open http://127.0.0.1:8765"
