"""Load/save batch GUI form state (Qt and Tk) under the user home directory."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


_SCHEMA_VERSION = 1


def settings_file_path() -> Path:
    return Path.home() / ".sharp_local_batch" / "gui_settings.json"


def default_batch_gui_settings() -> dict[str, Any]:
    return {
        "schema_version": _SCHEMA_VERSION,
        "folder": "",
        "recursive": True,
        "force_all": False,
        "limit_splats": False,
        "max_splats": "500000",
        "skip_up_to_date": True,
        "export_spz": True,
        "spz_only": False,
        "remove_ply_after_spz": True,
        "mirror": False,
        "output_mirror": "",
        "use_photos_library": False,
    }


_BOOL_KEYS = (
    "recursive",
    "force_all",
    "limit_splats",
    "skip_up_to_date",
    "export_spz",
    "spz_only",
    "remove_ply_after_spz",
    "mirror",
    "use_photos_library",
)


def _as_bool(value: object, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    return default


def _as_str(value: object, default: str) -> str:
    if isinstance(value, str):
        return value
    return default


def load_batch_gui_settings() -> dict[str, Any]:
    base = default_batch_gui_settings()
    path = settings_file_path()
    if not path.is_file():
        return base
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return base
    if not isinstance(raw, dict):
        return base
    for key in _BOOL_KEYS:
        if key in raw:
            base[key] = _as_bool(raw[key], base[key])
    for key in ("folder", "max_splats", "output_mirror"):
        if key in raw:
            base[key] = _as_str(raw[key], base[key])
    base["schema_version"] = _SCHEMA_VERSION
    return base


def _merge_batch_gui_settings(data: Mapping[str, Any]) -> dict[str, Any]:
    defaults = default_batch_gui_settings()
    merged: dict[str, Any] = {**defaults}
    for key in defaults:
        if key == "schema_version":
            continue
        if key in data:
            merged[key] = data[key]
    merged["schema_version"] = _SCHEMA_VERSION
    return merged


def _merged_equals_defaults(merged: Mapping[str, Any]) -> bool:
    """True if merged settings match defaults (same normalization as load)."""
    d = default_batch_gui_settings()
    for key in _BOOL_KEYS:
        if _as_bool(merged.get(key), d[key]) != d[key]:
            return False
    folder = _as_str(merged.get("folder"), "").strip()
    if folder != d["folder"]:
        return False
    max_splats = _as_str(merged.get("max_splats"), d["max_splats"]).strip()
    if not max_splats:
        max_splats = d["max_splats"]
    if max_splats != d["max_splats"]:
        return False
    output_mirror = _as_str(merged.get("output_mirror"), "").strip()
    if output_mirror != d["output_mirror"]:
        return False
    return True


def save_batch_gui_settings(data: Mapping[str, Any]) -> None:
    merged = _merge_batch_gui_settings(data)
    path = settings_file_path()
    if _merged_equals_defaults(merged):
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(merged, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def clear_saved_batch_gui_settings() -> None:
    path = settings_file_path()
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass
