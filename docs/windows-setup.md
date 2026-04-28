# Windows development setup

Developer instructions for building and running Sharp Local from source on Windows.

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

## Run from source

### Batch tool — GUI

```powershell
.venv\Scripts\activate
python -m sharp_local_batch
```

### Batch tool — CLI (no GUI needed)

```powershell
.venv\Scripts\activate
python -m sharp_local_batch --cli --folder C:\path\to\photos --recursive
```

With mirrored output:

```powershell
python -m sharp_local_batch --cli --folder C:\path\to\photos --recursive ^
  --output-root C:\path\to\splat_mirror
```

### Web UI

```powershell
.venv\Scripts\activate
python app.py
```

Then open **http://127.0.0.1:8765** in a browser.

## Build a standalone `.exe`

```powershell
.\.venv\Scripts\python -m pip install pyinstaller
.\.venv\Scripts\pyinstaller packaging/sharp_batch.spec
```

Output: `dist\SharpBatch\SharpBatch.exe`

Run it directly or pass CLI flags:

```powershell
dist\SharpBatch\SharpBatch.exe
dist\SharpBatch\SharpBatch.exe --cli --folder C:\photos --recursive
```

To distribute to end users, zip the entire `dist\SharpBatch\` folder — recipients just unzip and run `SharpBatch.exe` (no Python or Git needed on their machine).

## Notes

- The **first inference** downloads the SHARP model checkpoint (~2.6 GB) into the user cache (`%LOCALAPPDATA%\torch\hub\checkpoints\`). Make sure you have internet access and enough disk space.
- Default PyTorch includes CPU support, which works fine. For GPU acceleration with an NVIDIA card, see <https://pytorch.org/get-started/locally/> to install the CUDA-enabled version instead.
- The standalone bundle is large (PyTorch + Qt). This is expected.
- If command discovery differs between terminals (for example Cursor terminal vs external PowerShell), run tools with explicit paths like `.\.venv\Scripts\python ...` and `.\.venv\Scripts\pyinstaller ...`.
- For automated CI builds, see the GitHub Actions example in the main [README](../README.md).
