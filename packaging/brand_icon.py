"""Generate Sharp Local app icons from packaging/icon.svg.

Source of truth: packaging/icon.svg
Outputs: packaging/sharp-local.icns (macOS), packaging/sharp-local.ico (Windows),
and favicons under static/.

Edit packaging/icon.svg, then re-run this script (or rebuild) to refresh
.icns/.ico and favicons.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw

_PACKAGING = Path(__file__).resolve().parent
_REPO = _PACKAGING.parent
_SVG = _PACKAGING / "icon.svg"

# Matches packaging/icon.svg path (viewBox 0 0 16 16), fill #3399FF.
_ICON_PATH: tuple[tuple[float, float], ...] = (
    (13, 2),
    (3, 2),
    (3, 8),
    (10, 8),
    (10, 11),
    (3, 11),
    (3, 14),
    (13, 14),
    (13, 8),
    (6, 8),
    (6, 5),
    (13, 5),
)
_ICON_COLOR = (0x33, 0x99, 0xFF, 255)
_VIEWBOX = 16.0

_ICO_SIZES: tuple[int, ...] = (16, 24, 32, 48, 64, 128, 256)

_ICNS_LAYERS: tuple[tuple[str, int], ...] = (
    ("icon_16x16.png", 16),
    ("icon_16x16@2x.png", 32),
    ("icon_32x32.png", 32),
    ("icon_32x32@2x.png", 64),
    ("icon_128x128.png", 128),
    ("icon_128x128@2x.png", 256),
    ("icon_256x256.png", 256),
    ("icon_256x256@2x.png", 512),
    ("icon_512x512.png", 512),
    ("icon_512x512@2x.png", 1024),
)


def render_icon(size: int) -> Image.Image:
    """Rasterize the blocky S from packaging/icon.svg at ``size``×``size``.

    Drawn with Pillow (no Qt) so builds work headless. Keep ``_ICON_PATH`` in
    sync with the SVG path when the mark changes.
    """
    if not _SVG.is_file():
        raise FileNotFoundError(f"Missing icon source: {_SVG}")

    # Supersample then downscale for cleaner edges at large sizes.
    supersample = 4 if size >= 32 else 1
    render_size = size * supersample
    scale = render_size / _VIEWBOX
    img = Image.new("RGBA", (render_size, render_size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    points = [(x * scale, y * scale) for x, y in _ICON_PATH]
    draw.polygon(points, fill=_ICON_COLOR)
    if supersample > 1:
        img = img.resize((size, size), Image.Resampling.BOX)
    return img


def write_icns(path: Path) -> None:
    """Write a macOS .icns file (Pillow first, iconutil fallback)."""
    if sys.platform != "darwin":
        raise RuntimeError("write_icns requires macOS")

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        render_icon(1024).save(path, format="ICNS")
        return
    except Exception as exc:
        print(f"Pillow ICNS write failed ({exc}); falling back to iconutil", file=sys.stderr)

    iconset = path.with_suffix(".iconset")
    if iconset.exists():
        shutil.rmtree(iconset)
    iconset.mkdir()
    try:
        for name, layer_size in _ICNS_LAYERS:
            render_icon(layer_size).save(iconset / name, format="PNG")
        subprocess.run(
            ["iconutil", "-c", "icns", str(iconset), "-o", str(path)],
            check=True,
        )
    finally:
        shutil.rmtree(iconset, ignore_errors=True)


def write_ico(path: Path, sizes: Iterable[int] = _ICO_SIZES) -> None:
    """Write a multi-resolution Windows .ico file."""
    size_list = list(sizes)
    path.parent.mkdir(parents=True, exist_ok=True)
    largest = max(size_list)
    render_icon(largest).save(
        path,
        format="ICO",
        sizes=[(s, s) for s in size_list],
    )


def write_favicons(static_dir: Path) -> None:
    """Write small favicons for the Flask web UI."""
    static_dir.mkdir(parents=True, exist_ok=True)
    render_icon(32).save(static_dir / "favicon-32x32.png", format="PNG")
    render_icon(16).save(static_dir / "favicon-16x16.png", format="PNG")
    write_ico(static_dir / "favicon.ico", sizes=(16, 32, 48))


if __name__ == "__main__":
    if sys.platform == "win32":
        ico = _PACKAGING / "sharp-local.ico"
        write_ico(ico)
        print(f"Wrote {ico}")
    elif sys.platform == "darwin":
        icns = _PACKAGING / "sharp-local.icns"
        write_icns(icns)
        print(f"Wrote {icns}")
        ico = _PACKAGING / "sharp-local.ico"
        write_ico(ico)
        print(f"Wrote {ico}")
    else:
        ico = _PACKAGING / "sharp-local.ico"
        write_ico(ico)
        print(f"Wrote {ico}")

    write_favicons(_REPO / "static")
    print(f"Wrote favicons in {_REPO / 'static'}")
