# Sample Data Manifest (v0.1)

Synthetic frames for pytest and quickstart validation. Regenerate with:

```bash
python scripts/generate_synthetic_samples.py
```

## Layout

| Directory | Count | Description | Expected detector |
|-----------|-------|-------------|-------------------|
| `frames/edge/` | 5 | Green spill on subject contour | `edge_bleed` |
| `frames/block/` | 5 | Nearest-neighbor upscale blockiness | `compression_artifact` |
| `frames/normal/` | 5 | Clean reference frames | none (severity `good`) |

## Golden reports (`expected/`)

| File | Source frame | Notes |
|------|--------------|-------|
| `edge_01.json` | `frames/edge/edge_01.png` | Must contain `edge_bleed` |
| `block_01.json` | `frames/block/block_01.png` | Must contain `compression_artifact` |
| `normal_01.json` | `frames/normal/normal_01.png` | Empty or low-severity degradations |

Regenerate golden JSON:

```bash
python detect.py --image data/sample/frames/edge/edge_01.png --mode fast --output data/sample/expected/edge_01.json
python detect.py --image data/sample/frames/block/block_01.png --mode fast --output data/sample/expected/block_01.json
python detect.py --image data/sample/frames/normal/normal_01.png --mode fast --output data/sample/expected/normal_01.json
```

Validate against schema:

```bash
pytest tests/contract/test_report_schema.py -q
```
