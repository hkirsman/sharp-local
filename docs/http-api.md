# HTTP splat API

Sharp Local exposes a minimal HTTP API for on-demand splat generation. The
batch window app starts it via **Start server**; `python app.py` serves the
same endpoints (plus the browser UI at `/`).

Default base URL: **http://127.0.0.1:8765** (no trailing slash).

## GET `/health`

Readiness probe. Should be fast; do not load models here.

```json
{ "ok": true, "kind": "splat" }
```

| Field | Meaning |
|-------|---------|
| `ok` | `true` when the service can accept work |
| `kind` | Always `"splat"` for this server |

## POST `/transform`

Synchronous: hold the connection until the splat file is ready.

**Request:** multipart form

- `file` (required) - image bytes (JPEG, PNG, HEIC, etc.)
- `format` (optional) - `ply` (default) or `spz`. This is a preference.
  There is no dedicated "format" response header. If SPZ export fails,
  the server still returns PLY.

**Success (200):** raw bytes, not JSON

- `Content-Type: application/octet-stream`
- `Content-Disposition: attachment; filename="splat.ply"` or `splat.spz`

Use the filename extension to see what you actually got. Do not assume it
matches `format` (SPZ requests can fall back to PLY).

**Errors:** JSON `{ "error": "..." }` with 400, 413, 500, or 503.

## Examples

### curl

```bash
curl -s http://127.0.0.1:8765/health
curl -sS -o splat.ply \
  -F "file=@photo.jpg;type=image/jpeg" \
  http://127.0.0.1:8765/transform
```

### Python

Requires [requests](https://pypi.org/project/requests/) (`pip install requests`). Use a
long timeout on `/transform` - inference can take minutes. Success is **raw bytes**;
errors are JSON `{"error": "..."}`.

```python
import requests

base = "http://127.0.0.1:8765"
assert requests.get(f"{base}/health", timeout=3).json()["ok"]

with open("photo.jpg", "rb") as f:
    r = requests.post(
        f"{base}/transform",
        files={"file": ("photo.jpg", f, "image/jpeg")},
        data={"format": "ply"},  # or "spz"
        timeout=600,
    )
r.raise_for_status()
open("splat.ply", "wb").write(r.content)
```

On failure, skip `raise_for_status()` and read `r.json()["error"]` instead.

## Notes

- Bind `127.0.0.1` for same-machine clients; use `0.0.0.0` for LAN access
  (`python app.py --host 0.0.0.0 --port 8765`).
- Inference is serialized (one GPU job at a time).
- First request may download SHARP weights (~2.6 GB).

## Legacy browser API

The web UI still uses JSON endpoints under `/api/` (e.g. `POST /api/generate`).
New integrations should prefer `/health` and `/transform` above.
