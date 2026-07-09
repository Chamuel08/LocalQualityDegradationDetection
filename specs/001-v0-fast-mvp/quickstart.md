# Quickstart: v0.1 Fast Mode MVP

**Feature**: 001-v0-fast-mvp  
**Goal**: Clone → install → run detection on sample badcase frames in **≤ 5 minutes** (SC-001)

---

## Prerequisites

- Python **3.10+**
- pip
- ~500 MB disk (MediaPipe + OpenCV wheels)

Optional: NVIDIA GPU (not required for MVP)

---

## 1. Clone & install

```bash
git clone <repo-url> LocalQualityDegradationDetection
cd LocalQualityDegradationDetection

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -e ".[dev]"
cp config.example.yaml config.yaml
```

If `pyproject.toml` not yet present during early implement, use:

```bash
pip install opencv-python-headless numpy mediapipe pyyaml jsonschema pytest
```

---

## 2. Verify sample data

After implementation, sample layout:

```text
data/sample/
├── frames/
│   ├── edge/          # ≥5 green-edge badcases
│   ├── block/         # ≥5 block-artifact badcases
│   └── normal/        # ≥5 clean reference frames
├── expected/          # ≥3 golden JSON reports
└── README.md
```

Check files exist:

```bash
ls data/sample/frames/edge/ | head
ls data/sample/frames/block/ | head
ls data/sample/frames/normal/ | head
```

---

## 3. Run single-frame detection (JSON)

```bash
python detect.py \
  --image data/sample/frames/edge/green_edge_01.png \
  --mode fast
```

### Expected outcome (edge badcase)

- Exit code: **0**
- stdout: valid JSON with:
  - `mode`: `"fast"`
  - `degradations[].detector`: `"edge_bleed"`
  - `degradations[].degradation_type`: contains `green_edge` or `color_spill`
  - `degradations[].evidence.detail`: Chinese description mentioning绿边/色差/ΔE
  - `decision_trace`: entries with stages `global_scan`, `routing`, `detection`, `aggregation`

Validate schema:

```bash
python detect.py --image data/sample/frames/edge/green_edge_01.png --mode fast \
  --output /tmp/report.json

python -c "
import json
from jsonschema import validate
from pathlib import Path
schema = json.loads(Path('specs/001-v0-fast-mvp/contracts/quality-report.schema.json').read_text())
report = json.loads(Path('/tmp/report.json').read_text())
validate(report, schema)
print('schema OK')
"
```

---

## 4. Run block-artifact detection

```bash
python detect.py \
  --image data/sample/frames/block/block_01.png \
  --mode fast \
  --output /tmp/block_report.json
```

### Expected outcome

- `degradations[].detector`: `"compression_artifact"`
- `degradations[].evidence.metric`: `"blockiness_score"`
- `degradations[].root_cause_hypothesis.cause`: `"encoding_loss"`

---

## 5. Normal frame (low false positive)

```bash
python detect.py \
  --image data/sample/frames/normal/clean_01.png \
  --mode fast
```

### Expected outcome

- `degradations`: `[]` **or** only minor items
- `overall_mos` ≥ **4.0**
- No `severity: "critical"` entries

---

## 6. HTML report

```bash
python detect.py \
  --image data/sample/frames/edge/green_edge_01.png \
  --mode fast \
  --output reports/demo.html

open reports/demo.html    # macOS
# xdg-open reports/demo.html   # Linux
```

### Expected outcome

- Browser opens static HTML
- Input image preview visible
- Bounding box(es) drawn on degradation regions
- Table/list shows `method`, `metric`, `value`, `threshold`, `detail` per item
- If no degradations: message like「未检出显著劣化」

---

## 7. Batch mode

```bash
mkdir -p reports/batch
python detect.py \
  --image-dir data/sample/frames/edge/ \
  --mode fast \
  --output-dir reports/batch/
```

### Expected outcome

- One `.json` per input PNG in `reports/batch/`
- Filename stem matches input stem

---

## 8. Ignore regions (overlay)

```bash
python detect.py \
  --image data/sample/frames/edge/overlay_01.png \
  --ignore-regions data/sample/ignore_regions/overlay_01.json \
  --mode fast \
  --output /tmp/overlay_report.json
```

Sidecar format: see [cli-contract.md](./contracts/cli-contract.md).

---

## 9. Run tests

```bash
pytest tests/ -q
```

### Expected

- Contract tests pass against golden samples in `data/sample/expected/`
- Unit tests for edge_bleed, compression, MOS aggregation
- Integration test runs CLI on ≥1 sample frame

---

## 10. Success criteria checklist

| ID | Check | Command / criterion |
|----|-------|---------------------|
| SC-001 | 5-min path | Steps 1–3 complete without manual debugging |
| SC-002 | Edge recall | ≥80% on `data/sample/frames/edge/` (≥5 images) |
| SC-003 | Block recall | ≥80% on `data/sample/frames/block/` |
| SC-004 | Normal FPR | ≤10% false critical on `data/sample/frames/normal/` |
| SC-006 | Schema | 100% golden samples pass JSON Schema |
| SC-007 | HTML | Step 6 renders bbox + evidence |

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `ModuleNotFoundError: mediapipe` | `pip install mediapipe` |
| Empty degradations on obvious green edge | Check `config.yaml` thresholds; verify foreground segmentation |
| Schema validation fails | Compare output to [data-model.md](./data-model.md) required fields |
| Slow on CPU | Expected for MVP; SC-005 allows ≤500ms on GPU, CPU uncapped |

---

## References

- Feature spec: [spec.md](../spec.md)
- Data model: [data-model.md](./data-model.md)
- CLI contract: [contracts/cli-contract.md](./contracts/cli-contract.md)
- JSON Schema: [contracts/quality-report.schema.json](./contracts/quality-report.schema.json)
- Public specs: [`specs/`](../)
