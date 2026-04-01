"""Shared SHARP inference, PLY sidecar output, and splat-transform decimation."""

from __future__ import annotations

import logging
import shutil
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


def _bundle_root() -> Path:
    """Repo root in dev; PyInstaller extract dir when frozen (see packaging/sharp_batch.spec)."""
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS)
    return Path(__file__).resolve().parent.parent


REPO_ROOT = _bundle_root()
ML_SHARP_SRC = REPO_ROOT / "ml-sharp" / "src"

if ML_SHARP_SRC.is_dir():
    sys.path.insert(0, str(ML_SHARP_SRC))

LOGGER = logging.getLogger("sharp_local_batch.core")
if not LOGGER.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
    LOGGER.addHandler(_h)
    LOGGER.setLevel(logging.INFO)

if not ML_SHARP_SRC.is_dir():
    LOGGER.warning(
        "ml-sharp not found at %s — run: git submodule update --init ml-sharp",
        ML_SHARP_SRC.parent,
    )

PREDICT_LOCK = threading.Lock()
_predictor: Any = None
_device: Optional[str] = None


def ensure_sharp_imports() -> None:
    if not ML_SHARP_SRC.is_dir():
        raise RuntimeError(
            f"ml-sharp missing at {ML_SHARP_SRC}. From the repo root run: "
            "git submodule update --init ml-sharp or ./bootstrap.sh (see README)."
        )
    import sharp  # noqa: F401


def predictor_loaded() -> bool:
    return _predictor is not None


def inference_device() -> Optional[str]:
    return _device


def get_predictor() -> tuple[Any, str]:
    global _predictor, _device
    ensure_sharp_imports()
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


def count_ply_vertices(ply_path: Path) -> int:
    """Vertex count from the PLY ASCII header only (no full file decode)."""
    vertex_count: Optional[int] = None
    with ply_path.open("r", encoding="ascii", errors="replace") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line == "end_header":
                break
            parts = line.split()
            if len(parts) == 3 and parts[0] == "element" and parts[1] == "vertex":
                try:
                    vertex_count = int(parts[2])
                except ValueError:
                    pass
    if vertex_count is None:
        raise ValueError(
            f"PLY header missing valid 'element vertex N' before end_header: {ply_path}"
        )
    return vertex_count


def decimate_ply_splat_transform(
    ply_path: Path, target_count: int, timeout_sec: int = 7200
) -> bool:
    """Decimate PLY in place via PlayCanvas splat-transform CLI."""
    exe = shutil.which("splat-transform")
    if not exe:
        LOGGER.warning(
            "splat-transform not on PATH; install: npm install -g @playcanvas/splat-transform"
        )
        return False
    tmp_out = ply_path.with_name("_splat_decimated_tmp.ply")
    try:
        tmp_out.unlink(missing_ok=True)
        cmd = [exe, str(ply_path), "--decimate", str(target_count), str(tmp_out)]
        r = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
        if r.returncode != 0:
            err = (r.stderr or r.stdout or "").strip()
            LOGGER.warning(
                "splat-transform failed (exit %s): %s",
                r.returncode,
                err[:2000] if err else "(no output)",
            )
            tmp_out.unlink(missing_ok=True)
            return False
        if not tmp_out.is_file():
            LOGGER.warning("splat-transform produced no output file")
            return False
        tmp_out.replace(ply_path)
        return True
    except subprocess.TimeoutExpired:
        LOGGER.warning("splat-transform timed out after %s s", timeout_sec)
        tmp_out.unlink(missing_ok=True)
        return False
    except OSError as e:
        LOGGER.warning("splat-transform output handling failed: %s", e)
        tmp_out.unlink(missing_ok=True)
        return False


def supported_image_suffixes() -> set[str]:
    ensure_sharp_imports()
    from sharp.utils import io as sharp_io

    return {ext.lower() for ext in sharp_io.get_supported_image_extensions()}


def is_supported_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in supported_image_suffixes()


def list_image_paths(root: Path, recursive: bool) -> list[Path]:
    allowed = supported_image_suffixes()
    out: list[Path] = []
    if recursive:
        for p in root.rglob("*"):
            if p.is_file() and p.suffix.lower() in allowed and not _path_is_skipped(p):
                out.append(p)
    else:
        for p in root.iterdir():
            if p.is_file() and p.suffix.lower() in allowed and not _path_is_skipped(p):
                out.append(p)
    out.sort(key=lambda x: str(x).lower())
    return out


def _path_is_skipped(path: Path) -> bool:
    return any(part.startswith(".") for part in path.parts)


def sidecar_ply_path(image_path: Path) -> Path:
    return image_path.with_suffix(".ply")


def needs_ply_refresh(image_path: Path) -> bool:
    ply = sidecar_ply_path(image_path)
    if not ply.is_file():
        return True
    try:
        return image_path.stat().st_mtime > ply.stat().st_mtime
    except OSError:
        return True


@dataclass
class PlySidecarResult:
    ok: bool
    image_path: Path
    ply_path: Path
    message: str
    splat_count: Optional[int] = None
    splat_count_full: Optional[int] = None
    limit_applied: bool = False
    decimate_error: Optional[str] = None
    skipped: bool = False


def process_image_to_sidecar_ply(
    image_path: Path,
    *,
    limit_splats: bool = False,
    max_splats: Optional[int] = None,
) -> PlySidecarResult:
    """Run SHARP on ``image_path`` and write ``<stem>.ply`` beside it; optional decimate."""
    image_path = image_path.resolve()
    ply_path = sidecar_ply_path(image_path)

    if not is_supported_image(image_path):
        return PlySidecarResult(
            ok=False,
            image_path=image_path,
            ply_path=ply_path,
            message=f"Unsupported or missing image: {image_path}",
        )

    if limit_splats and max_splats is not None:
        if max_splats < 1:
            return PlySidecarResult(
                ok=False,
                image_path=image_path,
                ply_path=ply_path,
                message="max_splats must be at least 1",
            )

    ensure_sharp_imports()
    from sharp.cli.predict import predict_image
    from sharp.utils import io as sharp_io
    from sharp.utils.gaussians import save_ply

    import torch

    try:
        ply_path.parent.mkdir(parents=True, exist_ok=True)
        with PREDICT_LOCK:
            predictor, device_str = get_predictor()
            image, _, f_px = sharp_io.load_rgb(image_path)
            height, width = int(image.shape[0]), int(image.shape[1])
            device = torch.device(device_str)
            gaussians = predict_image(predictor, image, f_px, device)
            save_ply(gaussians, f_px, (height, width), ply_path)
    except Exception as e:
        LOGGER.exception("Inference failed for %s", image_path)
        try:
            ply_path.unlink(missing_ok=True)
        except OSError:
            pass
        return PlySidecarResult(
            ok=False,
            image_path=image_path,
            ply_path=ply_path,
            message=str(e),
        )

    splat_count_full = count_ply_vertices(ply_path)
    splat_count = splat_count_full
    limit_applied = False
    decimate_error: Optional[str] = None

    if limit_splats and max_splats is not None and splat_count_full > max_splats:
        if decimate_ply_splat_transform(ply_path, max_splats):
            splat_count = count_ply_vertices(ply_path)
            limit_applied = splat_count < splat_count_full
        else:
            decimate_error = (
                "Decimation failed or splat-transform missing; "
                "install: npm install -g @playcanvas/splat-transform"
            )

    msg = f"OK — {splat_count:,} splats"
    if limit_applied and splat_count_full > splat_count:
        msg += f" (from {splat_count_full:,})"
    if decimate_error:
        msg += f"; {decimate_error}"

    return PlySidecarResult(
        ok=True,
        image_path=image_path,
        ply_path=ply_path,
        message=msg,
        splat_count=splat_count,
        splat_count_full=splat_count_full,
        limit_applied=limit_applied,
        decimate_error=decimate_error,
    )
