# HTTP splat API

Sharp Local exposes a minimal HTTP API for on-demand splat generation. The
batch window app starts it via **Start server**; `python app.py` serves the
same endpoints (plus the browser UI at `/`).

Default base URL: **http://127.0.0.1:8765** (no trailing slash).

## GET `/api/logs`

Download the Sharp Local web log file (`sharp-web.log`) as an attachment.
Useful for packaged apps that have no console window.

**Success (200):** `text/plain` file download.

**Errors:** JSON `{ "error": "..." }` with 404 if the log is missing.

Failed `/api/generate` responses may include `log_path` and `log_url` pointing here.

## SHARP model checkpoint

The Apple SHARP weights (~2.6 GB) are **not** fetched during inference. Download
them explicitly (web UI banner, batch **SHARP model** group, or CLI
`--download-model`). Generation returns **503** until the checkpoint is ready.

### GET `/api/model/status`

```json
{
  "state": "idle",
  "percent": 0,
  "bytes_downloaded": 0,
  "bytes_total": 2791728742,
  "size_exact": true,
  "message": "",
  "error": null,
  "cache_path": null,
  "url": "https://ml-site.cdn-apple.com/models/sharp/…"
}
```

| Field | Meaning |
|-------|---------|
| `state` | `idle` \| `downloading` \| `ready` \| `error` |
| `percent` | 0-100 overall progress |
| `bytes_*` | Progress / expected size |
| `size_exact` | `true` when size came from disk or HTTP `Content-Length` |
| `cache_path` | On-disk path when the file is present |

### POST `/api/model/download`

Start a background download if not already ready. Returns the same status JSON.

### POST `/api/model/cancel`

Stop an in-progress download and discard the partial `.pt.tmp` file. Returns
the same status JSON. Idempotent if nothing is downloading.

`state` stays `downloading` (with `message` "Cancelling download…") until the
worker exits, then `idle`. Cancel is not a download `error`.

### POST `/api/model/delete`

Unload the in-memory predictor (if any) and delete the allowlisted checkpoint
file. Returns **409** while a download or inference is in progress.

Success includes `deleted_bytes` and `deleted_paths` plus the updated status fields.

## GET `/health`

Readiness probe. Should be fast; do not load models here.

```json
{
  "ok": true,
  "kind": "splat",
  "name": "sharp-local",
  "version": "…",
  "model_ready": false,
  "model_state": "idle"
}
```

| Field | Meaning |
|-------|---------|
| `ok` | `true` when the service process is up (ml-sharp present) |
| `kind` | Always `"splat"` for this server |
| `model_ready` | `true` when the SHARP checkpoint is on disk |
| `model_state` | Same states as `/api/model/status` |

`ok` stays true while weights are missing so Vaela's splat-service probe does
not treat the host as unreachable. Clients that need inference should check
`model_ready` or accept **503** from `POST /transform`.

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

When the checkpoint is missing or still downloading, the response is **503**
with `"error": "model_not_ready"` or `"model_downloading"` and a `model`
status object.

## Examples

### curl

```bash
curl -s http://127.0.0.1:8765/health
curl -s http://127.0.0.1:8765/api/model/status
curl -sS -X POST http://127.0.0.1:8765/api/model/download
curl -sS -X POST http://127.0.0.1:8765/api/model/cancel
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
health = requests.get(f"{base}/health", timeout=3).json()
assert health["ok"]
if not health.get("model_ready"):
    requests.post(f"{base}/api/model/download", timeout=10).raise_for_status()
    # poll GET /api/model/status until state == "ready"

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
- Download the SHARP checkpoint explicitly before the first generate / transform.

## Legacy browser API

The web UI still uses JSON endpoints under `/api/` (e.g. `POST /api/generate`).
New integrations should prefer `/health` and `/transform` above. `POST /api/generate`
also returns **503** until the model is ready.
