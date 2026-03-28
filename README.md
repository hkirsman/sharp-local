# Sharp local (SHARP web experiment)

Small local app: upload an image → run [Apple SHARP](https://github.com/apple/ml-sharp) (PyTorch) → view the 3D Gaussian splat in the browser with Three.js and [GaussianSplats3D](https://github.com/mkkellogg/GaussianSplats3D).

Inference is **not** in the browser; only the UI and viewer are.

## Upstream: `ml-sharp` (git submodule)

Apple’s inference code is **[ml-sharp](https://github.com/apple/ml-sharp)**, vendored as a **git submodule** at `./ml-sharp` (pinned to the commit recorded in this repo).

Clone **with submodules**:

```bash
git clone --recurse-submodules https://github.com/hkirsman/sharp-local.git
cd sharp-local
```

If you already cloned **without** submodules:

```bash
git submodule update --init ml-sharp
```

`./bootstrap.sh` always runs `git submodule update --init ml-sharp` so the checkout matches the **pinned** commit recorded in this repo (cheap no-op when already in sync).

Avoid `git submodule update --init --depth 1 ml-sharp` unless you know that pinned commit is reachable from the remote’s default tip—a shallow fetch can omit older pins and fail checkout.

## Setup (macOS / Homebrew Python)

Use a virtual environment (PEP 668 blocks global `pip`). From the repo root:

```bash
./bootstrap.sh
```

You need **git** on your `PATH` so the submodule can be initialized.

Manual equivalent (submodule already checked out):

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -U pip
python3 -m pip install -e ./ml-sharp -r requirements.txt
```

## Run

From the directory that contains `app.py`:

```bash
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

## Credits

- **UI / idea:** The two-panel layout (upload + preview, generate, local run) and overall workflow were inspired by [**Rob de Winter**](https://www.youtube.com/watch?v=8S57bfQ9w9A)’s walkthrough (“Apple SHARP + custom Three.js interface” on YouTube). This repository is independent community work—not affiliated with Apple, ml-sharp’s authors, or the video creator.
- **3D Gaussian splat preview (browser):** [GaussianSplats3D](https://github.com/mkkellogg/GaussianSplats3D) by Mark Kellogg, consumed as [`@mkkellogg/gaussian-splats-3d@0.4.6`](https://unpkg.com/@mkkellogg/gaussian-splats-3d@0.4.6/) from the [unpkg](https://unpkg.com/) CDN (imported in `static/main.js`). npm listing: [`@mkkellogg/gaussian-splats-3d`](https://www.npmjs.com/package/@mkkellogg/gaussian-splats-3d). See the upstream [license](https://github.com/mkkellogg/GaussianSplats3D/blob/main/LICENSE).
- **WebGL / math (browser):** [Three.js](https://threejs.org/) **r170** (npm `three@0.170.0`) as an ES module, loaded via an import map from unpkg in `static/index.html` (unpkg URLs use that semver, not `three@r170`).
- **Monocular splats (local inference):** [Apple ml-sharp](https://github.com/apple/ml-sharp) and its model weights; use is subject to their [LICENSE](https://github.com/apple/ml-sharp/blob/main/LICENSE) and [LICENSE_MODEL](https://github.com/apple/ml-sharp/blob/main/LICENSE_MODEL).

Third-party scripts are fetched from **unpkg** at runtime in development-style setups; for stricter deployments, vendor these files or use your own CDN and integrity checks.
