"""Shared SHARP inference, PLY sidecar output, and splat-transform decimation."""

from __future__ import annotations

import functools
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

LOGGER = logging.getLogger(__name__)

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


@functools.lru_cache(maxsize=1)
def supported_image_suffixes() -> frozenset[str]:
    ensure_sharp_imports()
    from sharp.utils import io as sharp_io

    return frozenset(ext.lower() for ext in sharp_io.get_supported_image_extensions())


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


def default_macos_photos_library_path() -> Path:
    """Default Apple Photos library bundle path (macOS)."""
    return Path.home() / "Pictures" / "Photos Library.photoslibrary"


def is_photos_library_bundle(path: Path) -> bool:
    """True if ``path`` is a ``.photoslibrary`` package (do not write PLY inside)."""
    try:
        p = path.resolve()
        return p.is_dir() and p.suffix.lower() == ".photoslibrary"
    except OSError:
        return False


def effective_batch_scan_root(library_or_folder: Path) -> Path:
    """Directory to enumerate for images.

    Apple Photos stores downloadable originals under ``originals/`` (and older
    libraries may use ``Masters/``) inside the ``.photoslibrary`` bundle — same
    as ``backend/api.py`` (``PHOTO_DIRS`` / iCloud discovery). Scanning the
    bundle root alone often misses those files.
    """
    try:
        p = library_or_folder.resolve()
    except OSError:
        return library_or_folder
    if not is_photos_library_bundle(p):
        return p
    originals = p / "originals"
    if originals.is_dir():
        return originals
    masters = p / "Masters"
    if masters.is_dir():
        return masters
    return p


PHOTOS_LIBRARY_MIRROR_HELP = (
    "A Photos Library.photoslibrary bundle cannot store PLY files next to originals. "
    "Enable mirror output and choose a target folder for mirror outside that package."
)


def mirrored_ply_path(image_path: Path, input_root: Path, output_root: Path) -> Path:
    """PLY path under ``output_root`` mirroring a relative path under the home folder.

    When the image resolves under the user's home directory (e.g.
    ``~/Pictures/Photos Library.photoslibrary/originals/…``), the mirror
    layout is ``<output_root>/Pictures/Photos Library.photoslibrary/originals/…``
    so the tree is unambiguous and not only ``originals/…``. Otherwise the
    path is relative to ``input_root`` (unchanged behavior for paths outside
    home, e.g. external volumes).
    """
    img = image_path.resolve()
    out_r = output_root.resolve()
    try:
        home_r = Path.home().resolve()
        rel = img.relative_to(home_r)
    except (ValueError, OSError):
        rel = img.relative_to(input_root.resolve())
    return (out_r / rel).with_suffix(".ply")


def output_ply_path_for_job(
    image_path: Path,
    *,
    mirror_output_root: Optional[Path],
    mirror_input_root: Optional[Path],
) -> Path:
    """Sidecar next to the image, or mirrored under ``mirror_output_root`` when set."""
    img = image_path.resolve()
    if mirror_output_root is None:
        return sidecar_ply_path(img)
    if mirror_input_root is None:
        raise ValueError("mirror_input_root is required when mirror_output_root is set")
    return mirrored_ply_path(img, mirror_input_root, mirror_output_root)


def needs_ply_refresh(image_path: Path, ply_path: Optional[Path] = None) -> bool:
    ply = sidecar_ply_path(image_path) if ply_path is None else ply_path
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
    ply_output_path: Optional[Path] = None,
) -> PlySidecarResult:
    """Run SHARP on ``image_path`` and write PLY beside it or at ``ply_output_path``; optional decimate."""
    image_path = image_path.resolve()
    ply_path = sidecar_ply_path(image_path) if ply_output_path is None else ply_output_path.resolve()

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
