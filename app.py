"""Local web UI for Apple SHARP (ml-sharp): browser + Flask + Three.js viewer.

Inference runs in Python with PyTorch (CPU / MPS / CUDA), not in the browser.

Homebrew Python blocks global pip (PEP 668). Use a venv — from this directory:

  ./bootstrap.sh
  (Initializes the ml-sharp git submodule and installs into a venv.)

Or manually:

  git submodule update --init ml-sharp
  python3 -m venv .venv
  source .venv/bin/activate
  python3 -m pip install -U pip
  python3 -m pip install -e ./ml-sharp -r requirements.txt
  python app.py
"""

from __future__ import annotations

import json
import logging
import shutil
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from flask import Flask, jsonify, request, send_file, send_from_directory

from sharp_local_batch.logging_config import ensure_stderr_info_logging

ensure_stderr_info_logging()

from sharp_local_batch.core import (
    ML_SHARP_SRC,
    PREDICT_LOCK,
    count_ply_vertices,
    decimate_ply_splat_transform,
    ensure_sharp_imports,
    export_ply_to_spz,
    get_predictor,
    inference_device,
    predictor_loaded,
)

EXPERIMENT_ROOT = Path(__file__).resolve().parent
OUTPUTS_DIR = EXPERIMENT_ROOT / "outputs"
STATIC_DIR = EXPERIMENT_ROOT / "static"


def _configure_logger(name: str) -> logging.Logger:
    """This app’s logger at INFO, no propagation to root; add StreamHandler if none."""
    log = logging.getLogger(name)
    log.setLevel(logging.INFO)
    log.propagate = False
    if not log.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
        log.addHandler(handler)
    return log


LOGGER = _configure_logger("sharp-web")
# ml-sharp pulls in matplotlib; on macOS its font scan is noisy and harmless.
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)

app = Flask(__name__, static_folder=str(STATIC_DIR), static_url_path="")
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024


def _scene_id_ok(scene_id: str) -> bool:
    try:
        uuid.UUID(scene_id)
        return True
    except ValueError:
        return False


def _parse_splat_limit_form() -> tuple[bool, Optional[int], Optional[str]]:
    """From multipart form: (limit_enabled, max_splats or None, error message or None)."""
    flag = (request.form.get("limit_splats") or "").strip().lower()
    active = flag in ("1", "true", "on", "yes")
    if not active:
        return False, None, None
    raw = (request.form.get("max_splats") or "").strip()
    if not raw:
        return True, None, "max_splats is required when limit splats is enabled"
    try:
        n = int(raw, 10)
    except ValueError:
        return True, None, "max_splats must be an integer"
    if n < 1:
        return True, None, "max_splats must be at least 1"
    cap = 10_000_000
    if n > cap:
        return True, None, f"max_splats cannot exceed {cap:,}"
    return True, n, None


def _parse_export_spz_form() -> bool:
    """From multipart form: whether SPZ export is requested (default True)."""
    flag = (request.form.get("export_spz") or "").strip().lower()
    if not flag:
        return True
    return flag in ("1", "true", "on", "yes")


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
            "model_loaded": predictor_loaded(),
            "device": inference_device(),
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
        meta: dict[str, Any] = {}
        if meta_path.is_file():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                original_name = meta.get("original_name", "") or ""
                if original_name:
                    label = original_name
            except (json.JSONDecodeError, OSError):
                meta = {}
        entry: dict[str, Any] = {
            "id": d.name,
            "label": label,
            "original_name": original_name,
            "mtime": ply.stat().st_mtime,
        }
        if "splat_count" in meta:
            entry["splat_count"] = meta["splat_count"]
        if "splat_count_full" in meta:
            entry["splat_count_full"] = meta["splat_count_full"]
        if meta.get("splat_limit_applied"):
            entry["splat_limit_applied"] = True
        if meta.get("decimate_error"):
            entry["decimate_error"] = meta["decimate_error"]
        if "elapsed_seconds" in meta and type(meta["elapsed_seconds"]) in (int, float):
            entry["elapsed_seconds"] = float(meta["elapsed_seconds"])
        if (d / "splat.spz").is_file():
            entry["spz_url"] = f"/api/scenes/{d.name}/splat.spz"
        scenes.append(entry)
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


@app.route("/api/scenes/<scene_id>/splat.spz", methods=["GET"])
def get_splat_spz(scene_id: str) -> Any:
    if not _scene_id_ok(scene_id):
        return jsonify({"error": "Invalid scene id"}), 400
    spz = OUTPUTS_DIR / scene_id / "splat.spz"
    if not spz.is_file():
        return jsonify({"error": "Not found"}), 404
    return send_file(spz, mimetype="application/octet-stream", as_attachment=False)


@app.route("/api/generate", methods=["POST"])
def generate() -> Any:
    if "file" not in request.files:
        return jsonify({"error": "Missing file field"}), 400
    upload = request.files["file"]
    if not upload.filename:
        return jsonify({"error": "Empty filename"}), 400

    ensure_sharp_imports()
    from sharp.utils import io as sharp_io
    from sharp.utils.gaussians import save_ply

    from sharp.cli.predict import predict_image

    ext = Path(upload.filename).suffix
    allowed = set(sharp_io.get_supported_image_extensions())
    if ext not in allowed:
        return jsonify({"error": f"Unsupported image type: {ext or '(none)'}"}), 400

    limit_on, max_splats, limit_err = _parse_splat_limit_form()
    if limit_err:
        return jsonify({"error": limit_err}), 400
    do_spz = _parse_export_spz_form()

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    scene_id = str(uuid.uuid4())
    scene_dir = OUTPUTS_DIR / scene_id
    scene_dir.mkdir(parents=True, exist_ok=True)
    safe_suffix = ext if ext else ".jpg"
    input_path = scene_dir / f"input{safe_suffix}"
    upload.save(str(input_path))

    import torch

    t0 = time.perf_counter()
    try:
        with PREDICT_LOCK:
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
            shutil.rmtree(scene_dir, ignore_errors=True)
        except OSError:
            pass
        return jsonify({"error": "Inference failed; check server logs."}), 500

    splat_count_full = count_ply_vertices(ply_path)
    splat_count = splat_count_full
    limit_applied = False
    decimate_error: Optional[str] = None

    if limit_on and max_splats is not None:
        if splat_count_full > max_splats:
            if decimate_ply_splat_transform(ply_path, max_splats):
                splat_count = count_ply_vertices(ply_path)
                limit_applied = splat_count < splat_count_full
            else:
                decimate_error = (
                    "Decimation failed or splat-transform missing; "
                    "install: npm install -g @playcanvas/splat-transform"
                )

    spz_path = scene_dir / "splat.spz"
    has_spz = export_ply_to_spz(ply_path, spz_path) if do_spz else False

    elapsed_seconds = round(time.perf_counter() - t0, 3)

    meta = {
        "id": scene_id,
        "original_name": upload.filename,
        "created": datetime.now(timezone.utc).isoformat(),
        "width": width,
        "height": height,
        "has_spz": has_spz,
        "splat_count": splat_count,
        "splat_count_full": splat_count_full,
        "splat_limit_applied": limit_applied,
        "elapsed_seconds": elapsed_seconds,
    }
    if decimate_error:
        meta["decimate_error"] = decimate_error
    (scene_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    payload: dict[str, Any] = {
        "id": scene_id,
        "ply_url": f"/api/scenes/{scene_id}/splat.ply",
        "label": upload.filename,
        "splat_count": splat_count,
        "splat_count_full": splat_count_full,
        "splat_limit_applied": limit_applied,
        "elapsed_seconds": elapsed_seconds,
    }
    if decimate_error:
        payload["decimate_error"] = decimate_error
    if has_spz:
        payload["spz_url"] = f"/api/scenes/{scene_id}/splat.spz"
    return jsonify(payload)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    app.run(host="127.0.0.1", port=8765, debug=False, threaded=True)
