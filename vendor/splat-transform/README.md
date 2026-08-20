# Vendored splat-transform helpers

Pinned standalone CLIs for optional splat count reduction (`--limit-splats` /
"Limit splat count"). Built with Bun from `@playcanvas/splat-transform` using a
stub `webgpu` so Dawn native addons are not required. Sharp Local always runs
the helper with `-g cpu`.

- `VERSION` - npm package version pin
- `splat-transform-<ver>-darwin-arm64`
- `splat-transform-<ver>-windows-x64.exe`

Rebuild (infrequent):

```bash
./packaging/compile-splat-transform.sh
```

Committed as normal git binaries (keep each file under GitHub's 100 MB limit).
