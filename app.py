"""Local web UI for Apple SHARP (ml-sharp): browser + Flask + Three.js viewer.

Inference runs in Python with PyTorch (CPU / MPS / CUDA), not in the browser.

Homebrew Python blocks global pip (PEP 668). Use a venv — from this directory:

  ./bootstrap.sh
  (Initializes the ml-sharp git submodule and installs into a venv.)

Or manually:

  git submodule update --init --depth 1
  python3 -m venv .venv
  source .venv/bin/activate
  python3 -m pip install -U pip
  python3 -m pip install -e ./ml-sharp -r requirements.txt
  python app.py
"""

from __future__ import annotations

import json
import logging
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from flask import Flask, jsonify, request, send_file, send_from_directory

EXPERIMENT_ROOT = Path(__file__).resolve().parent
ML_SHARP_SRC = EXPERIMENT_ROOT / "ml-sharp" / "src"
OUTPUTS_DIR = EXPERIMENT_ROOT / "outputs"
STATIC_DIR = EXPERIMENT_ROOT / "static"

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("sharp-web")
# ml-sharp pulls in matplotlib; on macOS its font scan is noisy and harmless.
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)

if ML_SHARP_SRC.is_dir():
    import sys

    sys.path.insert(0, str(ML_SHARP_SRC))
else:
    LOGGER.warning(
        "ml-sharp not found at %s — run: git submodule update --init --depth 1",
        ML_SHARP_SRC.parent,
    )

app = Flask(__name__, static_folder=str(STATIC_DIR), static_url_path="")
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024

_predict_lock = threading.Lock()
_predictor: Any = None
_device: Optional[str] = None


def _ensure_sharp_imports() -> None:
    if not ML_SHARP_SRC.is_dir():
        raise RuntimeError(
            f"ml-sharp missing at {ML_SHARP_SRC}. Clone to resources/experiments/ml-sharp "
            "or run: pip install -e <path-to-ml-sharp>"
        )
    import sharp  # noqa: F401


def get_predictor() -> tuple[Any, str]:
    global _predictor, _device
    _ensure_sharp_imports()
    if _predictor is not None and _device is not None:
        return _predictor, _device

    import torch
    from sharp.cli.predict import DEFAULT_MODEL_URL
    from sharp.models import PredictorParams, create_predictor

    if torch.cuda.is_available():
        _device = "cuda"
    elif torch.mps.is_available():
        _device = "mps"
    else:
        _device = "cpu"

    LOGGER.info("Loading SHARP checkpoint (first run may download weights) on %s", _device)
    state_dict = torch.hub.load_state_dict_from_url(DEFAULT_MODEL_URL, progress=True)
    predictor = create_predictor(PredictorParams())
    predictor.load_state_dict(state_dict)
    predictor.eval()
    predictor.to(_device)
    _predictor = predictor
    return _predictor, _device


def _scene_id_ok(scene_id: str) -> bool:
    try:
        uuid.UUID(scene_id)
        return True
    except ValueError:
        return False


@app.route("/favicon.ico")
def favicon() -> Any:
    """Avoid 404 spam when the browser requests a default favicon."""
    return "", 204


@app.route("/")
def index() -> Any:
    return send_from_directory(STATIC_DIR, "index.html")


@app.route("/api/health")
def health() -> Any:
    ok = ML_SHARP_SRC.is_dir()
    return jsonify(
        {
            "ok": True,
            "ml_sharp_path": str(ML_SHARP_SRC),
            "ml_sharp_present": ok,
            "model_loaded": _predictor is not None,
            "device": _device,
        }
    )


@app.route("/api/scenes", methods=["GET"])
def list_scenes() -> Any:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    scenes: list[dict[str, Any]] = []
    for d in OUTPUTS_DIR.iterdir():
        if not d.is_dir() or not _scene_id_ok(d.name):
            continue
        ply = d / "splat.ply"
        if not ply.is_file():
            continue
        meta_path = d / "meta.json"
        label = d.name[:8] + "…"
        original_name = ""
        if meta_path.is_file():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                original_name = meta.get("original_name", "") or ""
                if original_name:
                    label = original_name
            except (json.JSONDecodeError, OSError):
                pass
        scenes.append(
            {
                "id": d.name,
                "label": label,
                "original_name": original_name,
                "mtime": ply.stat().st_mtime,
            }
        )
    scenes.sort(key=lambda s: s["mtime"], reverse=True)
    for s in scenes:
        del s["mtime"]
    return jsonify(scenes)


@app.route("/api/scenes/<scene_id>/splat.ply", methods=["GET"])
def get_splat(scene_id: str) -> Any:
    if not _scene_id_ok(scene_id):
        return jsonify({"error": "Invalid scene id"}), 400
    ply = OUTPUTS_DIR / scene_id / "splat.ply"
    if not ply.is_file():
        return jsonify({"error": "Not found"}), 404
    return send_file(ply, mimetype="application/octet-stream", as_attachment=False)


@app.route("/api/generate", methods=["POST"])
def generate() -> Any:
    if "file" not in request.files:
        return jsonify({"error": "Missing file field"}), 400
    upload = request.files["file"]
    if not upload.filename:
        return jsonify({"error": "Empty filename"}), 400

    _ensure_sharp_imports()
    from sharp.utils import io as sharp_io
    from sharp.utils.gaussians import save_ply

    from sharp.cli.predict import predict_image

    ext = Path(upload.filename).suffix
    allowed = set(sharp_io.get_supported_image_extensions())
    if ext not in allowed:
        return jsonify({"error": f"Unsupported image type: {ext or '(none)'}"}), 400

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    scene_id = str(uuid.uuid4())
    scene_dir = OUTPUTS_DIR / scene_id
    scene_dir.mkdir(parents=True, exist_ok=True)
    safe_suffix = ext if ext else ".jpg"
    input_path = scene_dir / f"input{safe_suffix}"
    upload.save(str(input_path))

    import torch

    try:
        with _predict_lock:
            predictor, device_str = get_predictor()
            image, _, f_px = sharp_io.load_rgb(input_path)
            height, width = int(image.shape[0]), int(image.shape[1])
            device = torch.device(device_str)
            gaussians = predict_image(predictor, image, f_px, device)
            ply_path = scene_dir / "splat.ply"
            save_ply(gaussians, f_px, (height, width), ply_path)
    except Exception:
        LOGGER.exception("Inference failed for %s", input_path)
        try:
            import shutil

            shutil.rmtree(scene_dir, ignore_errors=True)
        except OSError:
            pass
        return jsonify({"error": "Inference failed; check server logs."}), 500

    meta = {
        "id": scene_id,
        "original_name": upload.filename,
        "created": datetime.now(timezone.utc).isoformat(),
        "width": width,
        "height": height,
    }
    (scene_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return jsonify(
        {
            "id": scene_id,
            "ply_url": f"/api/scenes/{scene_id}/splat.ply",
            "label": upload.filename,
        }
    )


if __name__ == "__main__":
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    app.run(host="127.0.0.1", port=8765, debug=False, threaded=True)
