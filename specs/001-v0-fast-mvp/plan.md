# Implementation Plan: v0.1 Fast Mode MVP

**Branch**: `001-v0-fast-mvp` | **Date**: 2026-07-09 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `specs/001-v0-fast-mvp/spec.md`

## Summary

Deliver a **CLI-first, offline badcase single-frame** quality degradation detector with **Fast Mode only** and a **fixed linear pipeline**:

`GlobalScan (simplified) → EdgeBleed + CompressionArtifact (parallel) → ReportGenerator`

Output: JSON (default) or HTML report with **bbox + Evidence 四要素 + decision_trace + overall_mos**. No VLM, LLM Judge, Deep Mode, or other sub-detectors in v0.1.

Technical design aligns with this feature's `data-model.md`, `contracts/`, `research.md`, and [`method_selection.md`](../method_selection.md).

## Technical Context

**Language/Version**: Python 3.10+

**Primary Dependencies**:
- OpenCV (`opencv-python-headless`) — I/O, color space, morphology, contours
- NumPy — array ops
- MediaPipe — Selfie Segmentation (MVP GlobalScan foreground mask; CPU-friendly)
- `jsonschema` — contract validation in tests
- Optional: `opencv-contrib-python` if BRISQUE needed for compression confirmation; else pure NumPy blockiness for MVP

**Storage**: File-based only — input images, sidecar JSON (`--metadata`, `--ignore-regions`), output JSON/HTML under `--output` / `--output-dir`. Config: `config.yaml` at repo root (from `config.example.yaml`).

**Testing**: pytest — unit tests per detector, integration test on `data/sample/`, contract tests against `contracts/quality-report.schema.json`

**Target Platform**: macOS / Linux CLI; CPU required, GPU optional (MediaPipe/OpenCV may use GPU if available)

**Project Type**: CLI + Python library (`src/lqdd/`)

**Performance Goals**:
- P50 < 500ms per 720p frame on T4/3060-class GPU (MVP relaxed; long-term target 200ms)
- CPU-only: functional correctness over latency; no crash

**Constraints**:
- No external API calls (VLM/LLM)
- All thresholds in config with `[CONFIG]` keys
- Output MUST pass QualityReport JSON Schema (MVP subset)
- Evidence.detail MUST be Chinese

**Scale/Scope**:
- Single-frame + batch directory
- 2 sub-detectors + simplified GlobalScan
- `data/sample/` ≥ 15 public frames (5 edge, 5 block, 5 normal)

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | MVP Compliance | Notes |
|-----------|----------------|-------|
| I. Design-First | ✅ PASS | Pipeline, Schema, algorithms from feature contracts + research |
| II. CLI-First | ✅ PASS | `detect.py` entry; JSON stdout default |
| III. Explainability | ✅ PASS | Evidence 四要素 + decision_trace on every report |
| IV. Coarse-to-Fine & MVP Scope | ✅ PASS | Fixed Stage1→2→3; Out of Scope respected |
| V. Testability | ✅ PASS | pytest + JSON Schema validation + config file |
| VI. Badcase Frame Scene | ✅ PASS | Offline PNG/JPG; ignore_regions; 720p/1080p |

**Post-design re-check**: ✅ No violations. Fixed pipeline is an explicit MVP simplification (spec Out of Scope + Deferred to V1). V1 Agent/VLM/LLM 见 [`VERSION_ROADMAP.md`](../../VERSION_ROADMAP.md)、[`002-v1-agent-layer`](../002-v1-agent-layer/spec.md)。

## Project Structure

### Documentation (this feature)

```text
specs/001-v0-fast-mvp/
├── plan.md              # This file
├── research.md          # Phase 0 — algorithm & tooling decisions
├── data-model.md        # Phase 1 — canonical entities
├── quickstart.md        # Phase 1 — clone-to-run validation
├── contracts/           # Phase 1 — CLI + JSON Schema
│   ├── cli-contract.md
│   └── quality-report.schema.json
└── tasks.md             # Phase 2 (/speckit-tasks — not created here)
```

### Source Code (repository root)

```text
detect.py                      # CLI entry point (thin wrapper)
config.example.yaml            # [CONFIG] thresholds
pyproject.toml                 # package + pytest config

src/lqdd/
├── __init__.py
├── cli/
│   ├── __init__.py
│   └── main.py                # argparse, batch loop, exit codes
├── models/
│   ├── __init__.py
│   ├── enums.py               # RegionType, Severity, RootCauseCategory
│   ├── inputs.py              # SingleFrameInput, BadcaseMetadata
│   └── report.py              # DegradationItem, Evidence, QualityReport, TraceEntry
├── config/
│   ├── __init__.py
│   └── loader.py              # YAML → typed config
├── pipeline/
│   ├── __init__.py
│   └── fast_pipeline.py       # Fixed: GlobalScan → detectors → Report
├── global_scan/
│   ├── __init__.py
│   ├── scanner.py             # GlobalScan orchestration
│   ├── segmentation.py        # MediaPipe selfie + edge band
│   ├── nomination.py          # Edge-focused nomination (MVP)
│   └── text_ui.py             # Overlay band + ignore_regions merge
├── detectors/
│   ├── __init__.py
│   ├── base.py                # Detector protocol
│   ├── edge_bleed/
│   │   ├── __init__.py
│   │   └── detector.py        # Lab ΔE + green spill (MVP subset)
│   └── compression/
│       ├── __init__.py
│       └── detector.py        # DCT blockiness (MVP); optional local BRISQUE
└── report/
    ├── __init__.py
    ├── generator.py           # MOS aggregation, trace merge
    └── html_renderer.py       # Static HTML with bbox overlay

tests/
├── conftest.py
├── contract/
│   └── test_report_schema.py
├── unit/
│   ├── test_edge_bleed.py
│   ├── test_compression.py
│   ├── test_global_scan.py
│   └── test_mos.py
└── integration/
    └── test_cli_sample.py

data/sample/
├── frames/                    # PNG/JPG inputs
├── metadata/                  # optional sidecars
├── ignore_regions/            # optional overlay JSON
├── expected/                  # golden JSON for ≥3 frames
└── README.md                  # sample manifest
```

**Structure Decision**: Single Python package under `src/lqdd/` with root `detect.py` for discoverability. `pipeline/fast_pipeline.py` replaces AgentOrchestrator for v0.1 — hard-coded dispatch to `edge_bleed` + `compression_artifact` only.

## Implementation Phases

### Phase A — Scaffold & Models (foundation)

1. `pyproject.toml`, `config.example.yaml`, package layout
2. Implement `models/` from `data-model.md` and `contracts/quality-report.schema.json` (MVP fields only)
3. Config loader with defaults matching design `[CONFIG]` keys
4. Contract test skeleton validating empty/golden reports

### Phase B — GlobalScan (simplified)

MVP scope:

| Full product | MVP v0.1 |
|-----------|----------|
| Face parsing (CelebAMask) | MediaPipe Selfie Segmentation → foreground mask |
| face/hair/hand nominations | **Edge band only** + full-frame compression hint |
| text_ui OCR | Rectangular band heuristics + `--ignore-regions` union |
| fast_pass skip Stage 2 | **Disabled for MVP** — always run both detectors (simpler, avoids miss) |

Outputs: `segmentation_map`, `nominations` (edge + optional background), `decision_trace` entries for segmentation + nomination.

### Phase C — Sub-detectors

**EdgeBleed** (`research.md` R2):
- MVP: green spill + Lab ΔE color_spill on edge band from GlobalScan
- Map internal `EdgeDegradation` → canonical `DegradationItem` with `detector="edge_bleed"`, `root_cause_hypothesis.cause="matting_error"`

**CompressionArtifact** (`research.md` R3):
- MVP: DCT blockiness §4.1 on Y channel (full frame + ROI bboxes)
- Optional phase-2 within MVP: local BRISQUE confirm if opencv-contrib available
- `root_cause_hypothesis.cause="encoding_loss"`

Both detectors run **in parallel** (thread pool or sequential if simpler initially); results collected before Report.

### Phase D — Report Generator

Per `data-model.md` MOS rules:
- MOS: `base_mos=4.5`, decay_factor=0.7, caps for fast_reject/critical
- Merge detector outputs → `degradations[]` sorted by severity
- Build `decision_trace`: `mode_select` → `global_scan` → `routing` (fixed) → `detection` × N → `aggregation`
- HTML renderer: embed base64 image preview, draw bboxes, list evidence fields

### Phase E — CLI & Sample Data

- `detect.py` implementing [cli-contract.md](./contracts/cli-contract.md)
- `data/sample/` per [quickstart.md](./quickstart.md)
- Integration tests: SC-002~SC-004 thresholds on sample sets
- README quickstart link

## Complexity Tracking

> No constitution violations requiring justification.

| Simplification | Reason | Full design restored when |
|----------------|--------|----------------------------|
| Fixed pipeline vs AgentOrchestrator | MVP scope (spec Out of Scope) | [`002-v1-agent-layer`](../002-v1-agent-layer/spec.md) |
| MediaPipe vs face parsing | Faster bootstrap, CPU OK | GlobalScan spec §3.2 full path |
| Always run detectors vs fast_pass skip | Avoid false negatives in demo | After benchmark validates gates |
| EdgeBleed subset (green + ΔE only) | P1 user stories focus on green edge | face_artifact feature adds matting_trace |

## V1 Follow-up

v0.1 完成后接 [`002-v1-agent-layer`](../002-v1-agent-layer/spec.md)：

- `src/lqdd/pipeline/fast_pipeline.py` — v0.1 固定流水线
- `src/lqdd/pipeline/agent_pipeline.py` — V1 挂载 AgentOrchestrator（预留 Protocol / 工厂切换）
- VLMReasoner + LLMJudge 模块见 [`VERSION_ROADMAP.md`](../../VERSION_ROADMAP.md)

implement v0.1 时 `pipeline/` 包建议预留 `run_agent()` 接口签名，内部可先 `NotImplementedError`。

## Generated Artifacts

| Artifact | Path | Status |
|----------|------|--------|
| Research | [research.md](./research.md) | ✅ |
| Data model | [data-model.md](./data-model.md) | ✅ |
| CLI contract | [contracts/cli-contract.md](./contracts/cli-contract.md) | ✅ |
| JSON Schema | [contracts/quality-report.schema.json](./contracts/quality-report.schema.json) | ✅ |
| Quickstart | [quickstart.md](./quickstart.md) | ✅ |

## Next Step

Run `/speckit-tasks` to generate dependency-ordered `tasks.md`, then `/speckit-implement`.
