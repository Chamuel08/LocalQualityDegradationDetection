# CLI Contract: detect.py

**Version**: 0.1.0  
**Feature**: 001-v0-fast-mvp  
**Entry point**: `detect.py` (repository root)

---

## Synopsis

```text
detect.py [--image PATH | --image-dir DIR] --mode fast [OPTIONS]
```

Analyze offline badcase frame(s) and emit a **QualityReport** as JSON (stdout or file) or HTML.

---

## Arguments

### Required (one input mode)

| Argument | Type | Description |
|----------|------|-------------|
| `--image` | path | Single image file (`.png`, `.jpg`, `.jpeg`, `.webp`) |
| `--image-dir` | path | Directory of images (non-recursive in v0.1) |

**Constraint**: Exactly one of `--image` or `--image-dir` MUST be provided.

### Mode

| Argument | Type | Default | Allowed (v0.1) |
|----------|------|---------|----------------|
| `--mode` | string | `fast` | `fast` only |

**Behavior**: If `--mode deep` or other value → exit code `2`, stderr: `unsupported mode for v0.1: {value}`.

### Output

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--output` | path | `-` (stdout) | Output file path. Format by extension: `.json` → JSON; `.html` → HTML |
| `--output-dir` | path | — | Batch only: write one report per input image into directory |

**Batch naming**: `{output-dir}/{input_stem}.json` (or `.html` if `--output` ends with `.html`).

**Stdout default**: Pretty-printed JSON (UTF-8, `ensure_ascii=False`).

### Optional sidecars

| Argument | Type | Description |
|----------|------|-------------|
| `--metadata` | path | JSON file → `BadcaseMetadata` / `SourceInfo` fields |
| `--ignore-regions` | path | JSON file: `{ "regions": [[x,y,w,h], ...] }` — excluded from detection |
| `--config` | path | YAML config (default: `./config.yaml` if exists, else `config.example.yaml`) |
| `--frame-id` | string | Override `frame_id` for single-image mode |

### Logging

| Argument | Type | Default |
|----------|------|---------|
| `--verbose` | flag | off |

---

## Exit Codes

| Code | Meaning |
|------|---------|
| `0` | Success (all batch items succeeded, or single image OK) |
| `1` | Runtime error (I/O, corrupt config, all batch items failed) |
| `2` | Invalid arguments / unsupported mode |

---

## Examples

### Single frame → stdout JSON

```bash
python detect.py --image data/sample/frames/edge/green_edge_01.png --mode fast
```

### Single frame → HTML report

```bash
python detect.py --image data/sample/frames/edge/green_edge_01.png --mode fast \
  --output reports/green_edge_01.html
```

### Batch directory

```bash
python detect.py --image-dir data/sample/frames/edge/ --mode fast \
  --output-dir reports/edge/
```

### With overlay ignore regions

```bash
python detect.py --image data/sample/frames/edge/overlay_01.png --mode fast \
  --ignore-regions data/sample/ignore_regions/overlay_01.json \
  --output report.json
```

---

## Metadata JSON Schema (sidecar)

```json
{
  "frame_id": "green_edge_01",
  "resolution": [1280, 720],
  "bitrate_kbps": 1500,
  "codec": "h264",
  "has_overlay": false
}
```

---

## Ignore Regions JSON Schema

```json
{
  "regions": [
    [0, 600, 1280, 120]
  ]
}
```

Each region: `[x, y, width, height]` in pixel coordinates.

---

## Output Contract

JSON output MUST validate against [`quality-report.schema.json`](./quality-report.schema.json).

Minimum fields for MVP acceptance tests:

- `overall_mos`, `severity`, `degradations`, `decision_trace`, `performance`
- Each degradation: `detector`, `bbox`, `evidence.method`, `evidence.metric`, `evidence.value`, `evidence.threshold`, `evidence.detail`

---

## Error Messages (stderr, human-readable)

| Condition | Message pattern |
|-----------|-----------------|
| Missing input | `error: specify --image or --image-dir` |
| File not found | `error: image not found: {path}` |
| Unsupported mode | `error: unsupported mode for v0.1: {mode}` |
| Invalid JSON sidecar | `error: failed to parse {path}: {reason}` |

---

## Non-goals (v0.1)

- No `--mode deep`
- No stdin image pipe (file paths only)
- No recursive `--image-dir`
- No API server / HTTP endpoint
