# SHARP ocal web experiment

Small local app: upload an image → run [Apple SHARP](https://github.com/apple/ml-sharp) (PyTorch) → view the 3D Gaussian splat in the browser with Three.js ([GaussianSplats3D](https://github.com/mkkellogg/GaussianSplats3D)).

Inference is **not** in the browser; only the UI and viewer are.

## Layout

Put **this folder** and a clone of **ml-sharp** together in either form:

- **Recommended:** `ml-sharp/` inside this directory (same level as `app.py`).
- **Alternative:** `ml-sharp/` next to `experiments/` (i.e. `resources/ml-sharp`).

## Setup (macOS / Homebrew Python)

Use a virtual environment (PEP 668 blocks global `pip`).

```bash
cd resources/experiments
./bootstrap.sh
```

If you do not have `ml-sharp` yet:

```bash
cd resources/experiments
git clone https://github.com/apple/ml-sharp.git
./bootstrap.sh
```

Manual equivalent:

```bash
cd resources/experiments
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -U pip
python3 -m pip install -e ./ml-sharp -r requirements.txt
```

(Use `../ml-sharp` instead of `./ml-sharp` if the repo lives beside `experiments/`.)

## Run

```bash
cd resources/experiments
source .venv/bin/activate
python app.py
```

Open **http://127.0.0.1:8765**

The first run downloads the SHARP checkpoint (~2.6 GB) into `~/.cache/torch/hub/checkpoints/`.

## Using the UI

1. Drop an image (or browse). HEIC is supported if `pillow-heif` is installed (comes with ml-sharp).
2. **Generate** — wait for inference (MPS/CUDA/CPU depending on your machine).
3. Orbit with the mouse; **arrow keys** move along the view (↑/→ forward, ↓/← back); **Splat size** slider adjusts screen-space scale; **Previous scenes** reloads saved `.ply` files from `outputs/`.

## References

- [ml-sharp](https://github.com/apple/ml-sharp) · [paper](https://arxiv.org/abs/2512.10685)
>>>>>>> 1f6b7f4 (GH-1: Add local SHARP web UI MVP)
