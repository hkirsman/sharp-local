# Windows development setup

Developer instructions for building Sharp Local standalone `.exe` bundles on Windows (batch tool and optional web UI). To run from source without PyInstaller, see the main [README](../README.md).

## Prerequisites

### Git

Download from <https://git-scm.com/download/win> and run the installer (defaults are fine).

### Python 3.13 (recommended) or 3.11 (fallback)

Download from <https://www.python.org/downloads/>.

Use the latest Python 3.13 installer on Windows.

If package install issues occur in your environment, use Python 3.11 instead.

**Important:** on the first installer screen, check **"Add python.exe to PATH"** before clicking Install.

> After installing both, close and reopen your terminal so the new `PATH` entries take effect.

## Clone and install

Open **PowerShell** (or Command Prompt) and run:

```powershell
git clone --recurse-submodules https://github.com/hkirsman/sharp-local.git
cd sharp-local

python -m venv .venv
.venv\Scripts\activate

python -m pip install -U pip
python -m pip install -e ./ml-sharp -r requirements.txt
```

## Build both `.exe` bundles (shortcut)

From the repo root in **Command Prompt**, or by double‑clicking in Explorer:

`compile-binaries-win.bat`

This installs PyInstaller into `.venv` if needed, then runs `packaging\sharp_batch.spec` and `packaging\sharp_web.spec`. For unattended use (no `pause` at the end), run `compile-binaries-win.bat nopause`.

## Build standalone batch `.exe`

```powershell
.\.venv\Scripts\python -m pip install pyinstaller
.\.venv\Scripts\pyinstaller packaging/sharp_batch.spec
```

Output: `dist\SharpBatch\SharpBatch.exe` (with `_internal\` beside it).

Run it directly or pass CLI flags:

```powershell
dist\SharpBatch\SharpBatch.exe
dist\SharpBatch\SharpBatch.exe --cli --folder C:\photos --recursive
```

To distribute, zip the entire `dist\SharpBatch\` folder (including `_internal`) — recipients unzip the whole folder and run `SharpBatch.exe` (no Python or Git needed on their machine).

## Build standalone web UI `.exe`

Same venv and dependencies as above, then:

```powershell
.\.venv\Scripts\pyinstaller packaging/sharp_web.spec
```

Output: `dist\SharpWeb\SharpWeb.exe`. Run it, then open **http://127.0.0.1:8765** in a browser.

When frozen, generated scenes are stored under `%LOCALAPPDATA%\SharpLocal\outputs\` (not next to the `.exe`). Zip `dist\SharpWeb\` the same way as the batch bundle (include `_internal`).

## Notes

- The **first inference** downloads the SHARP model checkpoint (~2.6 GB) into the user cache (`%LOCALAPPDATA%\torch\hub\checkpoints\`). Make sure you have internet access and enough disk space.
- Default PyTorch includes CPU support, which works fine. For GPU acceleration with an NVIDIA card, see <https://pytorch.org/get-started/locally/> to install the CUDA-enabled version instead.
- The standalone bundles are large (PyTorch; the batch build also includes Qt). This is expected.
- If command discovery differs between terminals (for example Cursor terminal vs external PowerShell), run tools with explicit paths like `.\.venv\Scripts\python ...` and `.\.venv\Scripts\pyinstaller ...`.
- For automated CI builds, see the GitHub Actions example in the main [README](../README.md).
