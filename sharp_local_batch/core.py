"""Shared SHARP inference, PLY sidecar output, and splat-transform decimation."""

from __future__ import annotations

import functools
import io
import logging
import os
import shutil
import subprocess
import sys
import tempfile
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
    tmp_fd, tmp_name = tempfile.mkstemp(
        prefix=f"._splat_decimated_{ply_path.stem}_",
        suffix=".ply",
        dir=str(ply_path.parent),
    )
    os.close(tmp_fd)
    tmp_out = Path(tmp_name)
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


def export_ply_to_spz(ply_path: Path, spz_path: Path) -> bool:
    """Write Niantic .spz from a SHARP splat PLY (vertex-only PLY for GaussForge).

    Returns ``True`` on success.  Logs a warning and returns ``False`` when
    ``gaussforge`` / ``plyfile`` is not importable or conversion fails.
    """
    try:
        import gaussforge
        from plyfile import PlyData, PlyElement
    except ImportError:
        LOGGER.warning(
            "gaussforge or plyfile not importable; run: pip install gaussforge plyfile"
        )
        return False
    try:
        plydata = PlyData.read(str(ply_path))
        vertex_el = plydata["vertex"]
        minimal = PlyData([PlyElement.describe(vertex_el.data, "vertex")])
        buf = io.BytesIO()
        minimal.write(buf)
        result = gaussforge.GaussForge().convert(buf.getvalue(), "ply", "spz")
        if "error" in result:
            LOGGER.warning("SPZ conversion failed: %s", result.get("error"))
            return False
        raw = result["data"]
        spz_path.parent.mkdir(parents=True, exist_ok=True)
        spz_path.write_bytes(raw if isinstance(raw, (bytes, bytearray)) else bytes(raw))
        return True
    except Exception:
        LOGGER.exception("SPZ export failed for %s", ply_path)
        return False


def try_remove_ply_after_spz(
    ply_path: Path,
    spz_path: Optional[Path],
    *,
    remove_requested: bool,
) -> tuple[bool, Optional[str]]:
    """Delete ``ply_path`` after a successful SPZ export when ``remove_requested``.

    Returns ``(removed, note)`` where *note* is a short user-facing reason if
    removal was requested but the PLY was left on disk; otherwise ``None``.
    """
    if not remove_requested or spz_path is None:
        return False, None
    if not spz_path.is_file():
        return False, "PLY kept (.spz missing after export)"
    try:
        ply_path.unlink()
        return True, None
    except OSError as e:
        LOGGER.warning("Could not remove PLY after SPZ: %s", e)
        return False, f"PLY kept (could not remove: {e})"


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
        try:
            root_r = root.resolve()
        except OSError:
            root_r = root
        if not root_r.is_dir():
            return []
        # followlinks=True matches pathlib.rglob: descend symlinked dirs (Photos may use
        # them). Rare symlink cycles are acceptable for typical library trees.
        for dirpath, dirnames, filenames in os.walk(
            root_r, topdown=True, followlinks=True
        ):
            dirnames[:] = [d for d in dirnames if not d.startswith(".")]
            base = Path(dirpath)
            for name in filenames:
                if name.startswith("."):
                    continue
                p = base / name
                try:
                    if (
                        p.is_file()
                        and p.suffix.lower() in allowed
                        and not _path_is_skipped(p)
                    ):
                        out.append(p)
                except OSError:
                    continue
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
    input_r = input_root.resolve()
    try:
        home_r: Optional[Path] = Path.home().resolve()
    except (OSError, RuntimeError):
        home_r = None
    if home_r is not None:
        try:
            return (out_r / img.relative_to(home_r)).with_suffix(".ply")
        except ValueError:
            pass
    try:
        rel = img.relative_to(input_r)
    except ValueError as exc:
        raise ValueError(
            f"Cannot mirror PLY path for {img}: not under "
            f"home ({home_r}) or input root {input_r}"
        ) from exc
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


def needs_ply_refresh(
    image_path: Path,
    ply_path: Optional[Path] = None,
    *,
    require_spz: bool = False,
) -> bool:
    """True if the PLY (and optionally the .spz sidecar) needs (re)generating."""
    ply = sidecar_ply_path(image_path) if ply_path is None else ply_path
    if not ply.is_file():
        # PLY missing — but if SPZ exists and is current vs the image,
        # the work was already done (PLY was removed after SPZ export).
        if require_spz and is_spz_current_for_image(ply, image_path):
            return False
        return True
    try:
        if image_path.stat().st_mtime > ply.stat().st_mtime:
            return True
    except OSError:
        return True
    if require_spz and needs_spz_refresh(ply):
        return True
    return False


def needs_spz_refresh(ply_path: Path) -> bool:
    """True when ``ply_path`` exists and the ``.spz`` sidecar is missing or stale.

    Stale means the SPZ file is older than the PLY (e.g. PLY was regenerated).
    """
    if not ply_path.is_file():
        return False
    spz_path = ply_path.with_suffix(".spz")
    if not spz_path.is_file():
        return True
    try:
        return spz_path.stat().st_mtime < ply_path.stat().st_mtime
    except OSError:
        return True


def is_spz_current_for_image(ply_path: Path, image_path: Path) -> bool:
    """True when ``ply_path`` has an ``.spz`` sidecar current vs ``image_path``."""
    spz_path = ply_path.with_suffix(".spz")
    if not spz_path.is_file():
        return False
    try:
        return spz_path.stat().st_mtime >= image_path.stat().st_mtime
    except OSError:
        return False


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
    spz_path: Optional[Path] = None
    spz_error: Optional[str] = None
    ply_removed: bool = False


def process_image_to_sidecar_ply(
    image_path: Path,
    *,
    limit_splats: bool = False,
    max_splats: Optional[int] = None,
    ply_output_path: Optional[Path] = None,
    export_spz: bool = True,
    remove_ply_after_spz: bool = True,
) -> PlySidecarResult:
    """Run SHARP on ``image_path`` and write PLY beside it or at ``ply_output_path``; optional decimate + SPZ."""
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

    spz_path: Optional[Path] = None
    spz_error: Optional[str] = None
    if export_spz:
        spz_target = ply_path.with_suffix(".spz")
        if export_ply_to_spz(ply_path, spz_target):
            spz_path = spz_target
        else:
            spz_error = "SPZ export failed (gaussforge missing or conversion error)"

    msg = f"OK — {splat_count:,} splats"
    if limit_applied and splat_count_full > splat_count:
        msg += f" (from {splat_count_full:,})"
    if decimate_error:
        msg += f"; {decimate_error}"
    if spz_path:
        msg += "; SPZ exported"
    elif spz_error:
        msg += f"; {spz_error}"

    ply_removed = False
    if remove_ply_after_spz and export_spz and spz_path is not None:
        ply_removed, note = try_remove_ply_after_spz(
            ply_path, spz_path, remove_requested=True
        )
        if note:
            msg += f"; {note}"
        elif ply_removed:
            msg += "; PLY removed (only .spz kept)"

    return PlySidecarResult(
        ok=True,
        image_path=image_path,
        ply_path=ply_path,
        message=msg,
        splat_count=splat_count,
        splat_count_full=splat_count_full,
        limit_applied=limit_applied,
        decimate_error=decimate_error,
        spz_path=spz_path,
        spz_error=spz_error,
        ply_removed=ply_removed,
    )


def _skip_if_spz_current(
    image_path: Path, ply_path: Path
) -> Optional[PlySidecarResult]:
    """Return a skipped result if a current .spz exists for a missing PLY, else None."""
    if not is_spz_current_for_image(ply_path, image_path):
        return None
    return PlySidecarResult(
        ok=True,
        image_path=image_path,
        ply_path=ply_path,
        message="Skipped (SPZ up to date, PLY previously removed)",
        skipped=True,
        spz_path=ply_path.with_suffix(".spz"),
    )


def update_ply_sidecar(
    image_path: Path,
    *,
    skip_up_to_date: bool,
    limit_splats: bool = False,
    max_splats: Optional[int] = None,
    ply_output_path: Optional[Path] = None,
    export_spz: bool = True,
    spz_only: bool = False,
    remove_ply_after_spz: bool = True,
) -> PlySidecarResult:
    """Skip, top up missing SPZ, SPZ-only from existing PLY, or run the full pipeline.

    When ``skip_up_to_date`` is set and the PLY is already current, this
    avoids re-running inference: if ``export_spz`` is on but the ``.spz``
    sidecar is missing, only the SPZ conversion runs against the existing
    PLY; otherwise the file is reported as skipped.  In all other cases the
    full :func:`process_image_to_sidecar_ply` pipeline runs.

    When ``spz_only`` is set (with ``export_spz``) and a PLY already exists at
    the target path, SHARP is not run: optional ``limit_splats`` /
    ``max_splats`` runs ``decimate_ply_splat_transform`` on that file, then
    ``export_ply_to_spz``.  If there is **no** PLY yet but a current ``.spz``
    sidecar exists (e.g. PLY was removed after a previous export), the file is
    skipped when ``skip_up_to_date`` is set.  Otherwise this falls back to the
    full :func:`process_image_to_sidecar_ply` pipeline.

    When ``remove_ply_after_spz`` is set, the target ``.ply`` is deleted after a
    successful ``.spz`` write (batch use; keeps disk usage down).
    """
    image_path = image_path.resolve()
    ply_path = (
        sidecar_ply_path(image_path)
        if ply_output_path is None
        else ply_output_path.resolve()
    )

    if spz_only:
        if not export_spz:
            return PlySidecarResult(
                ok=False,
                image_path=image_path,
                ply_path=ply_path,
                message="SPZ-only mode requires Export SPZ to be enabled",
            )
        if not ply_path.is_file():
            if skip_up_to_date:
                skipped = _skip_if_spz_current(image_path, ply_path)
                if skipped is not None:
                    return skipped
            return process_image_to_sidecar_ply(
                image_path,
                limit_splats=limit_splats,
                max_splats=max_splats,
                ply_output_path=ply_path,
                export_spz=export_spz,
                remove_ply_after_spz=remove_ply_after_spz,
            )
        if limit_splats and max_splats is not None and max_splats < 1:
            return PlySidecarResult(
                ok=False,
                image_path=image_path,
                ply_path=ply_path,
                message="max_splats must be at least 1",
            )

        spz_target = ply_path.with_suffix(".spz")
        if skip_up_to_date and spz_target.is_file():
            try:
                if spz_target.stat().st_mtime >= ply_path.stat().st_mtime:
                    return PlySidecarResult(
                        ok=True,
                        image_path=image_path,
                        ply_path=ply_path,
                        message="Skipped (SPZ up to date with PLY)",
                        skipped=True,
                    )
            except OSError:
                pass

        splat_count_full: Optional[int] = None
        try:
            splat_count_full = count_ply_vertices(ply_path)
        except (OSError, ValueError):
            pass

        splat_count = splat_count_full
        limit_applied = False
        decimate_error: Optional[str] = None

        if (
            limit_splats
            and max_splats is not None
            and splat_count_full is not None
            and splat_count_full > max_splats
        ):
            if decimate_ply_splat_transform(ply_path, max_splats):
                splat_count = count_ply_vertices(ply_path)
                limit_applied = (
                    splat_count is not None and splat_count < splat_count_full
                )
            else:
                decimate_error = (
                    "Decimation failed or splat-transform missing; "
                    "install: npm install -g @playcanvas/splat-transform"
                )

        if export_ply_to_spz(ply_path, spz_target):
            try:
                count = count_ply_vertices(ply_path)
            except (OSError, ValueError):
                count = splat_count
            msg = "SPZ exported (existing PLY only)"
            if count is not None:
                msg += f" — {count:,} splats"
            if limit_applied and splat_count_full is not None and count is not None:
                if splat_count_full > count:
                    msg += f" (from {splat_count_full:,})"
            if decimate_error:
                msg += f"; {decimate_error}"
            ply_removed = False
            if remove_ply_after_spz:
                ply_removed, note = try_remove_ply_after_spz(
                    ply_path, spz_target, remove_requested=True
                )
                if note:
                    msg += f"; {note}"
                elif ply_removed:
                    msg += "; PLY removed (only .spz kept)"
            return PlySidecarResult(
                ok=True,
                image_path=image_path,
                ply_path=ply_path,
                message=msg,
                splat_count=count,
                splat_count_full=splat_count_full if splat_count_full is not None else count,
                limit_applied=limit_applied,
                decimate_error=decimate_error,
                spz_path=spz_target,
                ply_removed=ply_removed,
            )
        err = "SPZ export failed (gaussforge missing or conversion error)"
        fail_msg = err
        if decimate_error:
            fail_msg = f"{err}; {decimate_error}"
        return PlySidecarResult(
            ok=False,
            image_path=image_path,
            ply_path=ply_path,
            message=fail_msg,
            splat_count=splat_count,
            splat_count_full=splat_count_full,
            limit_applied=limit_applied,
            decimate_error=decimate_error,
            spz_error=err,
        )

    if skip_up_to_date and not ply_path.is_file() and export_spz:
        skipped = _skip_if_spz_current(image_path, ply_path)
        if skipped is not None:
            return skipped

    if skip_up_to_date and ply_path.is_file():
        try:
            ply_stale = image_path.stat().st_mtime > ply_path.stat().st_mtime
        except OSError:
            ply_stale = True
        if not ply_stale:
            spz_target = ply_path.with_suffix(".spz")
            if export_spz and needs_spz_refresh(ply_path):
                if export_ply_to_spz(ply_path, spz_target):
                    try:
                        count = count_ply_vertices(ply_path)
                    except (OSError, ValueError):
                        count = None
                    msg = "SPZ exported (PLY current)"
                    if count is not None:
                        msg += f" — {count:,} splats"
                    ply_removed = False
                    if remove_ply_after_spz:
                        ply_removed, note = try_remove_ply_after_spz(
                            ply_path, spz_target, remove_requested=True
                        )
                        if note:
                            msg += f"; {note}"
                        elif ply_removed:
                            msg += "; PLY removed (only .spz kept)"
                    return PlySidecarResult(
                        ok=True,
                        image_path=image_path,
                        ply_path=ply_path,
                        message=msg,
                        splat_count=count,
                        splat_count_full=count,
                        spz_path=spz_target,
                        ply_removed=ply_removed,
                    )
                err = "SPZ export failed (gaussforge missing or conversion error)"
                return PlySidecarResult(
                    ok=False,
                    image_path=image_path,
                    ply_path=ply_path,
                    message=err,
                    spz_error=err,
                )
            return PlySidecarResult(
                ok=True,
                image_path=image_path,
                ply_path=ply_path,
                message="Skipped (PLY up to date)",
                skipped=True,
            )

    return process_image_to_sidecar_ply(
        image_path,
        limit_splats=limit_splats,
        max_splats=max_splats,
        ply_output_path=ply_path,
        export_spz=export_spz,
        remove_ply_after_spz=remove_ply_after_spz,
    )
