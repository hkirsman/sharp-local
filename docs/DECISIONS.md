# Sharp Local: Architecture Decision Log

This log tracks the major technical and packaging decisions made during the
development of Sharp Local.

---

### Decision 1: Skip Git LFS for vendored splat-transform helpers
*   **Date:** August 2026
*   **Context:** Packaged SharpBatch / SharpWeb apps need a standalone
    `splat-transform` CLI for optional splat count reduction. Bun
    `--compile` produces platform binaries of ~67 MB (macOS arm64) and
    ~100 MB (Windows x64). GitHub warns above 50 MB and hard-rejects above
    100 MB. Git LFS was considered for storage.
*   **Decision:** Commit the helpers as normal git blobs under
    `vendor/splat-transform/` (no `.gitattributes` LFS tracking). Keep each
    file under GitHub's 100 MB hard limit. Rebuild infrequently via
    `packaging/compile-splat-transform.sh`.
*   **Why:** Both current binaries fit under 100 MB (Windows is close at
    ~99.94 MB). Skipping LFS keeps clone/setup simple - no `git lfs install`
    requirement for contributors. Soft 50 MB warnings on push are acceptable.
    Revisit LFS (or release-asset hosting) if a future Bun / package bump
    pushes any helper over 100 MB.
