#!/usr/bin/env bash
# Compile standalone splat-transform helpers for Sharp Local packaging.
#
# Produces (under vendor/splat-transform/, npm version from VERSION file):
#   splat-transform-<ver>-darwin-arm64
#   splat-transform-<ver>-windows-x64.exe
#
# Requires: bun, npm. Does NOT run as part of compile-binaries-mac.sh /
# compile-binaries-win.bat - only when bumping the pinned helper.
#
# Uses a stub webgpu package so Bun can compile without embedding Dawn
# native addons (~20MB+ per platform). Sharp Local always invokes the CLI
# with -g cpu for --decimate (CPU path; GPU not required).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

VENDOR="$ROOT/vendor/splat-transform"
VERSION_FILE="$VENDOR/VERSION"

if [[ ! -f "$VERSION_FILE" ]]; then
  echo "ERROR: missing $VERSION_FILE" >&2
  exit 1
fi
VERSION="$(tr -d '[:space:]' < "$VERSION_FILE")"
if [[ -z "$VERSION" ]]; then
  echo "ERROR: VERSION file is empty" >&2
  exit 1
fi

if ! command -v bun >/dev/null 2>&1; then
  echo "ERROR: bun not found on PATH." >&2
  echo "Install: brew install oven-sh/bun/bun  (or https://bun.sh)" >&2
  exit 1
fi
if ! command -v npm >/dev/null 2>&1; then
  echo "ERROR: npm not found on PATH." >&2
  exit 1
fi

PKG="@playcanvas/splat-transform@${VERSION}"
OUT_MAC="$VENDOR/splat-transform-${VERSION}-darwin-arm64"
OUT_WIN="$VENDOR/splat-transform-${VERSION}-windows-x64.exe"

WORKDIR="$(mktemp -d "${TMPDIR:-/tmp}/sharp-splat-transform.XXXXXX")"
cleanup() { rm -rf "$WORKDIR"; }
trap cleanup EXIT

echo "Fetching ${PKG}..."
cd "$WORKDIR"
TARBALL="$(npm pack "$PKG" | tail -n 1)"
tar -xzf "$TARBALL"
cd package

echo "Installing @adobe/spz (omit real webgpu)..."
npm install --omit=dev --no-package-lock >/dev/null

# Replace webgpu with a stub so Bun --compile does not need Dawn .node files.
# Decimation uses -g cpu; GPU features are intentionally unavailable.
rm -rf node_modules/webgpu
mkdir -p node_modules/webgpu
cat > node_modules/webgpu/package.json <<'EOF'
{"name":"webgpu","version":"0.4.0","type":"module","main":"index.js"}
EOF
cat > node_modules/webgpu/index.js <<'EOF'
export const isMac = process.platform === "darwin";
export function create() {
  throw new Error(
    "webgpu not bundled in Sharp Local splat-transform helper; use -g cpu"
  );
}
export const globals = {};
EOF

echo "Compiling darwin-arm64 -> ${OUT_MAC}"
mkdir -p "$VENDOR"
bun build ./bin/cli.mjs --compile --minify \
  --target=bun-darwin-arm64 \
  --outfile "${OUT_MAC}.tmp"
# Rewrite via a new inode and clear xattrs. Bun-produced Mach-Os can carry
# com.apple.provenance that makes later `git hash-object` / `git status` get
# SIGKILL'd when reading the original path on some macOS setups.
rm -f "$OUT_MAC"
cp "${OUT_MAC}.tmp" "$OUT_MAC"
rm -f "${OUT_MAC}.tmp"
xattr -c "$OUT_MAC" 2>/dev/null || true
chmod +x "$OUT_MAC"

echo "Compiling windows-x64 -> ${OUT_WIN}"
bun build ./bin/cli.mjs --compile --minify \
  --target=bun-windows-x64 \
  --outfile "${OUT_WIN}.tmp"
rm -f "$OUT_WIN"
cp "${OUT_WIN}.tmp" "$OUT_WIN"
rm -f "${OUT_WIN}.tmp"
xattr -c "$OUT_WIN" 2>/dev/null || true

echo "Smoke-testing macOS binary (--help)..."
"$OUT_MAC" --help >/dev/null

echo ""
echo "========================================================================"
echo "splat-transform helpers ready (npm ${VERSION})."
echo "  $OUT_MAC"
echo "  $OUT_WIN"
ls -lh "$OUT_MAC" "$OUT_WIN"
echo ""
echo "Commit the vendor binaries with the rest of the repo. PyInstaller specs"
echo "expect these paths when building SharpBatch / SharpWeb."
echo "========================================================================"
