# Tasks: v0.1 Fast Mode MVP

**Input**: Design documents from `specs/001-v0-fast-mvp/`  
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/, quickstart.md

**Organization**: Tasks grouped by user story (US1 → US4 → US2 → US3) for independent delivery. Constitution V requires pytest + JSON Schema validation.

## Format: `[ID] [P?] [Story] Description`

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization and package layout

- [X] T001 Create `src/lqdd/` package tree per plan.md (cli, models, config, pipeline, global_scan, detectors, report)
- [X] T002 Create `pyproject.toml` with dependencies: opencv-python-headless, numpy, mediapipe, pyyaml, jsonschema; dev: pytest
- [X] T003 [P] Create `config.example.yaml` with global_scan, edge_bleed, compression, report thresholds from data-model.md
- [X] T004 [P] Create root `detect.py` thin entry calling `src/lqdd/cli/main.py`
- [X] T005 [P] Create `tests/conftest.py` with fixtures for sample image paths and schema path
- [X] T006 [P] Add `.gitignore` entries for `config.yaml`, `.venv`, `__pycache__`, `.cursor/`

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Models, config, GlobalScan, Report JSON core, pipeline skeleton — MUST complete before user stories

**⚠️ CRITICAL**: No user story work until this phase is complete

- [X] T007 [P] Implement enums in `src/lqdd/models/enums.py` (RegionType, Severity, RootCauseCategory)
- [X] T008 [P] Implement input types in `src/lqdd/models/inputs.py` (SingleFrameInput, BadcaseMetadata, BBox)
- [X] T009 [P] Implement report types in `src/lqdd/models/report.py` (Evidence, DegradationItem, TraceEntry, QualityReport, MOSBreakdown)
- [X] T010 Implement config loader in `src/lqdd/config/loader.py` loading YAML with design defaults
- [X] T011 [P] Implement detector protocol in `src/lqdd/detectors/base.py` (Detector protocol + shared helpers)
- [X] T012 Implement MediaPipe segmentation in `src/lqdd/global_scan/segmentation.py` (foreground mask + edge band)
- [X] T013 Implement text_ui + ignore_regions merge in `src/lqdd/global_scan/text_ui.py`
- [X] T014 Implement edge nomination in `src/lqdd/global_scan/nomination.py` (edge region + blockiness_hint)
- [X] T015 Implement GlobalScan orchestration in `src/lqdd/global_scan/scanner.py` (outputs nominations, trace entries)
- [X] T016 Implement MOS aggregation in `src/lqdd/report/generator.py` (base_mos, decay 0.7, caps, degradations merge)
- [X] T017 Implement JSON serialization in `src/lqdd/report/generator.py` (QualityReport → dict, UUID, ISO timestamp)
- [X] T018 Implement fixed routing trace in `src/lqdd/pipeline/fast_pipeline.py` (mode_select, global_scan, routing, detection stubs)
- [X] T019 [P] Add contract test skeleton in `tests/contract/test_report_schema.py` validating against `specs/001-v0-fast-mvp/contracts/quality-report.schema.json`
- [X] T020 [P] Add unit test for MOS formula in `tests/unit/test_mos.py`

**Checkpoint**: Foundation ready — models, config, GlobalScan, report JSON path exist

---

## Phase 3: User Story 1 — 单张 badcase JSON + 绿边检出 (Priority: P1) 🎯 MVP

**Goal**: `detect.py --image <path> --mode fast` 输出合法 JSON，检出 `edge_bleed` 绿边劣化

**Independent Test**: 对含绿边 PNG 运行 CLI，JSON 含 `degradations[].detector == "edge_bleed"` 且 `evidence.detail` 为中文

### Implementation for User Story 1

- [X] T021 [P] [US1] Implement green spill detection in `src/lqdd/detectors/edge_bleed/detector.py` (§4.1 green_channel)
- [X] T022 [P] [US1] Implement Lab ΔE color_spill in `src/lqdd/detectors/edge_bleed/detector.py` (§4.1b)
- [X] T023 [US1] Map EdgeBleed output to DegradationItem in `src/lqdd/detectors/edge_bleed/detector.py` (root_cause=matting_error, Chinese detail)
- [X] T024 [US1] Wire edge_bleed into `src/lqdd/pipeline/fast_pipeline.py` (edge nomination → detector → report)
- [X] T025 [US1] Implement CLI single-image mode in `src/lqdd/cli/main.py` (--image, --mode fast, --output, --metadata, --ignore-regions, --config)
- [X] T026 [US1] Add file-not-found and invalid-args exit codes per `contracts/cli-contract.md`
- [X] T027 [P] [US1] Add unit tests in `tests/unit/test_edge_bleed.py` (synthetic green border image)
- [X] T028 [US1] Add integration test in `tests/integration/test_cli_sample.py` for single edge frame JSON output

**Checkpoint**: US1 complete — one-command edge badcase JSON demo works

---

## Phase 4: User Story 4 — block 噪点检出 (Priority: P1)

**Goal**: 检出 `compression_artifact`，evidence 含 `blockiness_score`

**Independent Test**: block 样本 JSON 含 `degradations[].detector == "compression_artifact"`

### Implementation for User Story 4

- [X] T029 [P] [US4] Implement DCT blockiness in `src/lqdd/detectors/compression/detector.py` (Y channel, §4.1)
- [X] T030 [US4] Map block regions to bbox + DegradationItem in `src/lqdd/detectors/compression/detector.py` (root_cause=encoding_loss, Chinese detail)
- [X] T031 [US4] Wire compression_artifact into `src/lqdd/pipeline/fast_pipeline.py` (parallel or sequential with edge_bleed)
- [X] T032 [P] [US4] Add unit tests in `tests/unit/test_compression.py` (synthetic block pattern)
- [X] T033 [US4] Extend integration test in `tests/integration/test_cli_sample.py` for block frame detection

**Checkpoint**: US1 + US4 — full fixed pipeline (GlobalScan → edge + compression → Report)

---

## Phase 5: User Story 2 — 批量目录筛图 (Priority: P1)

**Goal**: `--image-dir` + `--output-dir` 批量产出 JSON

**Independent Test**: 3 张 PNG 目录生成 3 份对应 JSON；1 张损坏不影响其余

### Implementation for User Story 2

- [X] T034 [US2] Implement batch loop in `src/lqdd/cli/main.py` (--image-dir, --output-dir, stem-based filenames)
- [X] T035 [US2] Add per-file error handling in `src/lqdd/cli/main.py` (skip corrupt, log warning, continue batch)
- [X] T036 [US2] Extend integration test in `tests/integration/test_cli_sample.py` for batch 3-frame output

**Checkpoint**: US2 complete — batch badcase workflow matches USE_CASE §4 v0.1

---

## Phase 6: User Story 3 — HTML 报告 (Priority: P2)

**Goal**: `--output report.html` 可在浏览器查看 bbox + evidence

**Independent Test**: HTML 含图预览、劣化列表、evidence 四要素；无劣化显示「未检出显著劣化」

### Implementation for User Story 3

- [X] T037 [P] [US3] Implement HTML renderer in `src/lqdd/report/html_renderer.py` (base64 image, bbox overlay, evidence table)
- [X] T038 [US3] Wire HTML output path detection in `src/lqdd/cli/main.py` (.html extension → html_renderer)
- [X] T039 [US3] Add empty-degradation message in `src/lqdd/report/html_renderer.py` (未检出显著劣化)
- [X] T040 [US3] Add integration test in `tests/integration/test_cli_sample.py` for HTML output file exists and contains evidence fields

**Checkpoint**: US3 complete — interview demo with HTML report

---

## Phase 7: Polish & Cross-Cutting Concerns

**Purpose**: Sample data, golden JSON, README, V1 pipeline stub, quickstart validation

- [X] T041 [P] Create `data/sample/frames/{edge,block,normal}/` with ≥5 images each (synthetic script or placeholders in `scripts/generate_synthetic_samples.py`)
- [X] T042 [P] Create `data/sample/expected/` golden JSON for ≥3 frames matching schema
- [X] T043 [P] Add `data/sample/README.md` manifest listing frames and expected detectors
- [X] T044 Add `run_agent()` stub in `src/lqdd/pipeline/__init__.py` raising NotImplementedError with link to 002-v1-agent-layer
- [X] T045 Update README.md Implemented table and quickstart section per SC-001
- [X] T046 Run full `pytest tests/ -m "not vlm" -q` and fix failures
- [X] T047 Validate quickstart.md steps 1–3 manually (clone → install → detect sample)
- [X] T048 [P] Extend contract test in `tests/contract/test_report_schema.py` against all golden samples in `data/sample/expected/`

---

## Dependencies & Execution Order

### Phase Dependencies

- **Phase 1 Setup**: No dependencies — start immediately
- **Phase 2 Foundational**: Depends on Phase 1 — **BLOCKS all user stories**
- **Phase 3 US1**: Depends on Phase 2
- **Phase 4 US4**: Depends on Phase 3 (pipeline wiring pattern established)
- **Phase 5 US2**: Depends on Phase 3 (single-image CLI works)
- **Phase 6 US3**: Depends on Phase 3 (JSON report structure stable)
- **Phase 7 Polish**: Depends on Phases 3–6

### User Story Dependencies

- **US1 (P1)**: After Foundational — **MVP checkpoint** (edge JSON only)
- **US4 (P1)**: After US1 — adds compression to same pipeline
- **US2 (P1)**: After US1 — extends CLI to batch
- **US3 (P2)**: After US1 — adds HTML format (independent of US2/US4)

### Parallel Opportunities

- Phase 1: T003, T004, T005, T006 in parallel
- Phase 2: T007, T008, T009, T011, T019, T020 in parallel; then T012–T018 sequential
- Phase 3: T021, T022, T027 in parallel; T023–T026 sequential
- Phase 4: T029, T032 in parallel
- Phase 7: T041, T042, T043, T048 in parallel

---

## Parallel Example: User Story 1

```bash
# Detector algorithms in parallel:
T021: src/lqdd/detectors/edge_bleed/detector.py (green spill)
T022: src/lqdd/detectors/edge_bleed/detector.py (Lab ΔE)
T027: tests/unit/test_edge_bleed.py

# Then wire pipeline + CLI:
T023 → T024 → T025 → T028
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1 + Phase 2
2. Complete Phase 3 (US1)
3. **STOP and VALIDATE**: `python detect.py --image data/sample/frames/edge/green_edge_01.png --mode fast`
4. Demo-ready for edge badcase

### Incremental Delivery

1. Setup + Foundational → foundation ready
2. US1 → edge JSON demo (**MVP**)
3. US4 → add block detection
4. US2 → batch workflow
5. US3 → HTML for non-engineers
6. Polish → sample data + README + golden schema

### Suggested MVP Scope

**Minimum shippable**: Phase 1 + 2 + 3 (US1 only) — 28 tasks  
**Full v0.1 spec**: All phases — 48 tasks

---

## Notes

- No VLM/LLM tasks (deferred to `002-v1-agent-layer`)
- All thresholds from `config.example.yaml`; no magic numbers in detector code
- Evidence.detail MUST be Chinese (constitution III)
- `decision_trace` minimum stages: mode_select, global_scan, routing, detection, aggregation
