# macOS development setup

Developer instructions for building Sharp Local standalone bundles on macOS (batch tool and optional web UI). To run from source without PyInstaller, see the main [README](../README.md).

## Prerequisites

### Git

Git ships with Xcode Command Line Tools. If it is not already installed:

```bash
xcode-select --install
```

Alternatively install via Homebrew: `brew install git`.

### Python 3.13 (recommended) or 3.11 (fallback)

**Option A – python.org installer (recommended for PyInstaller builds)**

Download the universal2 macOS installer from <https://www.python.org/downloads/> and run it. This avoids Homebrew's PEP 668 `externally-managed-environment` restriction and produces cleaner standalone bundles.

**Option B – Homebrew**

```bash
brew install python@3.13
```

Homebrew Python works fine for running from source. For PyInstaller builds it also works, but use the venv exclusively (never install packages globally).

> After installing Python, close and reopen your terminal so the new `PATH` entries take effect.

### Homebrew (optional but recommended)

If you don't have Homebrew yet and want it for other tools:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

## Clone and install

```bash
git clone --recurse-submodules https://github.com/hkirsman/sharp-local.git
cd sharp-local
./bootstrap.sh
```

`bootstrap.sh` creates `.venv`, syncs the `ml-sharp` submodule to the pinned commit, and installs all Python dependencies. Manual equivalent if you need it:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -U pip
python3 -m pip install -e ./ml-sharp -r requirements.txt
```

## Build both bundles (shortcut)

From the repo root with the venv **activated**:

```bash
./compile-binaries-mac.sh
```

This installs PyInstaller into `.venv` if needed, then builds both specs. For use in CI (no interactive pause), the script exits cleanly on its own.

## Build standalone batch bundle

```bash
source .venv/bin/activate
python -m pip install pyinstaller
pyinstaller packaging/sharp_batch.spec
```

Output: `dist/SharpBatch/SharpBatch` (with `_internal/` beside it).

Run it directly or pass CLI flags:

```bash
./dist/SharpBatch/SharpBatch
./dist/SharpBatch/SharpBatch --cli --folder ~/Photos --recursive
```

To distribute, zip the entire `dist/SharpBatch/` folder (including `_internal/`) — recipients unzip the whole folder and run `SharpBatch` (no Python or Git required).

## Build standalone web UI bundle

Same venv and dependencies as above, then:

```bash
pyinstaller packaging/sharp_web.spec
```

Output: `dist/SharpWeb/SharpWeb`. Run it, then open **http://127.0.0.1:8765** in a browser.

When frozen, generated scenes are stored under `~/Library/Application Support/SharpLocal/outputs/` (not next to the binary). Zip `dist/SharpWeb/` the same way as the batch bundle.

## Notes

- **First inference** downloads the SHARP model checkpoint (~2.6 GB) into `~/.cache/torch/hub/checkpoints/`. Make sure you have internet access and enough disk space before the first run.
- **Apple Silicon (MPS acceleration):** PyTorch uses the Metal Performance Shaders (MPS) backend automatically on Apple Silicon Macs. No extra setup is needed. Inference is substantially faster than CPU.
- **Intel Mac:** inference runs on CPU. GPU acceleration via CUDA is not available on macOS.
- **Gatekeeper / "can't be opened" warning:** bundles built locally are not notarised. To open them the first time, right-click → **Open** → **Open** in the dialog, or run `xattr -dr com.apple.quarantine dist/SharpBatch/` after building.
- **Bundle size:** standalone bundles are large (PyTorch; the batch build also includes Qt). This is expected.
- **Python version mismatch:** if you switch Python versions, delete `.venv` and rerun `bootstrap.sh` before rebuilding.
- **_tkinter missing:** Homebrew Pythons may lack Tcl/Tk. Install `brew install python-tk@3.13` (match your Python minor) and recreate `.venv`, or use `--cli`, or rely on the PySide6 GUI (already in `requirements.txt`).
- **macOS Photos Library:** the GUI can enable **Use system Photos library as source folder** (`~/Pictures/Photos Library.photoslibrary`). PLY output must be mirrored outside the bundle — enable **Mirror PLY output** and pick a target folder. For the CLI pass `--output-root /path/to/mirror`.
