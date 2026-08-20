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
import os
import shutil
import sys
import time
import uuid
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from flask import Flask, Response, jsonify, request, send_file, send_from_directory

from sharp_local_batch.logging_config import ensure_stderr_info_logging

ensure_stderr_info_logging(log_file_name="sharp-web.log")

from sharp_local_batch._version import __version__ as SHARP_LOCAL_VERSION
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
from sharp_local_batch.model_download import (
    ModelNotReadyError,
    get_download_manager,
)

def _dev_root() -> Path:
    return Path(__file__).resolve().parent


def _bundle_root() -> Path:
    """PyInstaller extract dir when frozen; repo root in development."""
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS)
    return _dev_root()


def _outputs_dir() -> Path:
    """Scene PLY/SPZ under repo ``outputs/`` in dev; user-writable dir when frozen."""
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        if sys.platform == "win32":
            local = os.environ.get("LOCALAPPDATA")
            if local:
                return Path(local) / "SharpLocal" / "outputs"
            return Path.home() / "AppData" / "Local" / "SharpLocal" / "outputs"
        if sys.platform == "darwin":
            return (
                Path.home()
                / "Library"
                / "Application Support"
                / "SharpLocal"
                / "outputs"
            )
        xdg = os.environ.get("XDG_DATA_HOME")
        if xdg:
            return Path(xdg) / "sharp-local" / "outputs"
        return Path.home() / ".local" / "share" / "sharp-local" / "outputs"
    return _dev_root() / "outputs"


STATIC_DIR = _bundle_root() / "static"
OUTPUTS_DIR = _outputs_dir()


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
WEB_LOG_PATH: Path | None = None
try:
    from sharp_local_batch.logging_config import attach_file_handler, web_log_path

    WEB_LOG_PATH = attach_file_handler(LOGGER, "sharp-web.log") or web_log_path()
except Exception:
    WEB_LOG_PATH = None
# ml-sharp pulls in matplotlib; on macOS its font scan is noisy and harmless.
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)

_gui_log_sink: Callable[[str], None] | None = None


def set_gui_log_sink(sink: Callable[[str], None] | None) -> None:
    """When set, [gui_log] lines appear in the batch app log window."""
    global _gui_log_sink
    _gui_log_sink = sink


def gui_log(message: str) -> None:
    """User-facing line for the batch GUI log (not stderr / werkzeug)."""
    if _gui_log_sink is not None:
        try:
            _gui_log_sink(message)
        except Exception:
            pass


def _inference_failed_payload() -> dict[str, str]:
    """JSON body for failed inference; points at the on-disk log when available."""
    if WEB_LOG_PATH is not None:
        return {
            "error": f"Inference failed. Check logs at {WEB_LOG_PATH}",
            "log_path": str(WEB_LOG_PATH),
            "log_url": "/api/logs",
        }
    return {"error": "Inference failed; check server logs."}


def _suppress_flask_startup_noise() -> None:
    """Keep werkzeug dev-server banners off the GUI log and stderr."""
    logging.getLogger("werkzeug").setLevel(logging.ERROR)
    try:
        import flask.cli

        flask.cli.show_server_banner = lambda *args, **kwargs: None  # type: ignore[method-assign]
    except Exception:
        pass


app = Flask(__name__, static_folder=str(STATIC_DIR), static_url_path="")
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024


@app.errorhandler(413)
def request_entity_too_large(_e: Exception) -> Any:
    return jsonify({"error": "File too large"}), 413


def _scene_id_ok(scene_id: str) -> bool:
    try:
        uuid.UUID(scene_id)
        return True
    except ValueError:
        return False


def _parse_splat_limit_form() -> tuple[bool, Optional[int], Optional[str]]:
    """From multipart form: (limit_enabled, max_splats or None, error message or None).

    If the client omits ``limit_splats``, fall back to defaults set when the
    batch GUI started the HTTP server (``DEFAULT_LIMIT_SPLATS`` / ``DEFAULT_MAX_SPLATS``).
    """
    flag = (request.form.get("limit_splats") or "").strip().lower()
    if not flag:
        if app.config.get("DEFAULT_LIMIT_SPLATS"):
            n = app.config.get("DEFAULT_MAX_SPLATS")
            if isinstance(n, int) and n >= 1:
                return True, n, None
        return False, None, None
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
    """Serve the packaged favicon (falls back to empty if missing)."""
    path = STATIC_DIR / "favicon.ico"
    if path.is_file():
        return send_file(path, mimetype="image/x-icon")
    return "", 204


@app.route("/")
def index() -> Any:
    """Serve ``index.html`` with title and heading tied to ``SHARP_LOCAL_VERSION``."""
    path = STATIC_DIR / "index.html"
    branding = f"Sharp Local web {SHARP_LOCAL_VERSION}"
    body = path.read_text(encoding="utf-8").replace("__SHARP_LOCAL_WEB_TITLE__", branding)
    return Response(body, mimetype="text/html; charset=utf-8")


def _model_status_payload() -> dict[str, Any]:
    return get_download_manager().status_dict()


def _model_not_ready_response() -> tuple[Any, int]:
    """503 when checkpoint is missing or still downloading."""
    status = _model_status_payload()
    code = (
        "model_downloading"
        if status.get("state") == "downloading"
        else "model_not_ready"
    )
    return jsonify({"error": code, "model": status}), 503


def _require_model_ready() -> Optional[tuple[Any, int]]:
    if get_download_manager().is_ready():
        return None
    return _model_not_ready_response()


@app.route("/api/health")
def api_health() -> Any:
    ok = ML_SHARP_SRC.is_dir()
    model = _model_status_payload()
    payload: dict[str, Any] = {
        "ok": True,
        "app": "sharp-local-web",
        "version": SHARP_LOCAL_VERSION,
        "ml_sharp_path": str(ML_SHARP_SRC),
        "ml_sharp_present": ok,
        "model_loaded": predictor_loaded(),
        "model_ready": model.get("state") == "ready",
        "model_state": model.get("state"),
        "device": inference_device(),
    }
    if WEB_LOG_PATH is not None:
        payload["log_path"] = str(WEB_LOG_PATH)
        payload["log_url"] = "/api/logs"
    return jsonify(payload)


@app.route("/api/model/status", methods=["GET"])
def api_model_status() -> Any:
    """SHARP checkpoint download status (idle / downloading / ready / error)."""
    return jsonify(_model_status_payload())


@app.route("/api/model/download", methods=["POST"])
def api_model_download() -> Any:
    """Start a background download of the SHARP checkpoint if not ready."""
    mgr = get_download_manager()
    mgr.ensure_download_async()
    return jsonify(mgr.status_dict())


@app.route("/api/model/delete", methods=["POST"])
def api_model_delete() -> Any:
    """Delete the cached SHARP checkpoint to free disk space."""
    mgr = get_download_manager()
    try:
        result = mgr.delete_checkpoint()
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 409
    status = mgr.status_dict()
    status["deleted_bytes"] = result.get("deleted_bytes", 0)
    status["deleted_paths"] = result.get("deleted_paths", [])
    return jsonify(status)


@app.route("/api/logs", methods=["GET"])
def download_logs() -> Any:
    """Download the Sharp Local web log file (for packaged apps with no console)."""
    path = WEB_LOG_PATH
    if path is None or not path.is_file():
        return jsonify({"error": "Log file not available"}), 404
    for handler in LOGGER.handlers:
        try:
            handler.flush()
        except Exception:
            pass
    return send_file(
        path,
        mimetype="text/plain; charset=utf-8",
        as_attachment=True,
        download_name="sharp-web.log",
    )


@app.route("/health")
def health() -> Any:
    """HTTP splat service: readiness probe (GET /health).

    ``ok`` means the service process is up (ml-sharp present). Model weights
    may still be missing - clients should check ``model_ready`` or accept
    503 from ``POST /transform``.
    """
    ok = ML_SHARP_SRC.is_dir()
    model = _model_status_payload()
    return jsonify({
        "ok": ok,
        "kind": "splat",
        "name": "sharp-local",
        "version": SHARP_LOCAL_VERSION,
        "model_ready": model.get("state") == "ready",
        "model_state": model.get("state"),
    })


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

    blocked = _require_model_ready()
    if blocked is not None:
        return blocked

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
    except ModelNotReadyError:
        try:
            shutil.rmtree(scene_dir, ignore_errors=True)
        except OSError:
            pass
        return _model_not_ready_response()
    except Exception:
        LOGGER.exception("Inference failed for %s", input_path)
        try:
            shutil.rmtree(scene_dir, ignore_errors=True)
        except OSError:
            pass
        return jsonify(_inference_failed_payload()), 500

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
                    "Decimation failed or splat-transform helper missing"
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


@app.route("/transform", methods=["POST"])
def transform() -> Any:
    """HTTP splat service: image in, PLY or SPZ bytes out (POST /transform)."""
    if "file" not in request.files:
        return jsonify({"error": "Missing file field"}), 400
    upload = request.files["file"]
    if not upload.filename:
        return jsonify({"error": "Empty filename"}), 400

    fmt = (request.form.get("format") or "ply").strip().lower()
    if fmt not in ("ply", "spz"):
        fmt = "ply"
    name = Path(upload.filename).name
    blocked = _require_model_ready()
    if blocked is not None:
        gui_log("  Failed - SHARP model not ready (download required)")
        return blocked
    limit_on, max_splats, limit_err = _parse_splat_limit_form()
    if limit_err:
        gui_log(f"  Failed - {limit_err}")
        return jsonify({"error": limit_err}), 400
    gui_log(f"Generating splat for {name}…")
    t0 = time.perf_counter()

    ext = Path(name).suffix.lower()
    try:
        ensure_sharp_imports()
        from sharp.utils import io as sharp_io
        from sharp.utils.gaussians import save_ply
        from sharp.cli.predict import predict_image
        allowed = {e.lower() for e in sharp_io.get_supported_image_extensions()}
    except RuntimeError as e:
        LOGGER.warning("SHARP not ready for %s: %s", name, e)
        gui_log("  Failed - service not ready")
        return jsonify({"error": "Service not ready"}), 503

    if ext not in allowed:
        gui_log(f"  Failed - unsupported image type ({ext or 'none'})")
        return jsonify({"error": f"Unsupported image type: {ext or '(none)'}"}), 400

    import tempfile
    import torch

    # Read output into memory before the temp dir is deleted. Returning
    # send_file(path) from inside TemporaryDirectory deletes the file before
    # Flask streams it, so the client gets an empty/failed body.
    payload: bytes | None = None
    download_name = "splat.ply"

    with tempfile.TemporaryDirectory(prefix="sharp_transform_") as tmpdir:
        tmpdir_path = Path(tmpdir)
        safe_suffix = ext if ext else ".jpg"
        input_path = tmpdir_path / f"input{safe_suffix}"
        upload.save(str(input_path))

        ply_path = tmpdir_path / "splat.ply"
        try:
            with PREDICT_LOCK:
                predictor, device_str = get_predictor()
                image, _, f_px = sharp_io.load_rgb(input_path)
                height, width = int(image.shape[0]), int(image.shape[1])
                device = torch.device(device_str)
                gui_log(f"  Running SHARP ({width}x{height}, {device_str})…")
                gaussians = predict_image(predictor, image, f_px, device)
                save_ply(gaussians, f_px, (height, width), ply_path)
        except ModelNotReadyError:
            gui_log("  Failed - SHARP model not ready")
            return _model_not_ready_response()
        except Exception:
            LOGGER.exception("Inference failed for %s", name)
            gui_log(f"  Failed - {name} (see terminal for details)")
            return jsonify(_inference_failed_payload()), 500

        splat_count_full = count_ply_vertices(ply_path)
        splat_count = splat_count_full
        if limit_on and max_splats is not None and splat_count_full > max_splats:
            gui_log(f"  Limiting {splat_count_full:,} -> {max_splats:,} splats…")
            if decimate_ply_splat_transform(ply_path, max_splats):
                splat_count = count_ply_vertices(ply_path)
            else:
                gui_log("  Limit skipped (splat-transform missing or failed)")

        infer_s = time.perf_counter() - t0
        count_note = f"{splat_count:,} splats"
        if splat_count < splat_count_full:
            count_note += f" (from {splat_count_full:,})"

        if fmt == "spz":
            spz_path = tmpdir_path / "splat.spz"
            if export_ply_to_spz(ply_path, spz_path):
                payload = spz_path.read_bytes()
                download_name = "splat.spz"
                gui_log(f"  Done — {name} ({count_note}, {infer_s:.1f}s, SPZ)")
            else:
                LOGGER.warning("SPZ export failed for %s; returning PLY", name)

        if payload is None:
            payload = ply_path.read_bytes()
            download_name = "splat.ply"
            gui_log(f"  Done — {name} ({count_note}, {infer_s:.1f}s)")

    return Response(
        payload,
        mimetype="application/octet-stream",
        headers={
            "Content-Disposition": f'attachment; filename="{download_name}"',
            "Content-Length": str(len(payload)),
        },
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Sharp Local web server")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8765, help="Port (default: 8765)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Sharp Local web — http://%s:%d", args.host, args.port)
    app.run(host=args.host, port=args.port, debug=False, threaded=True)
