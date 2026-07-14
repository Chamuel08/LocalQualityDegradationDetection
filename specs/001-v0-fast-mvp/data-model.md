# Data Model: 001-v0-fast-mvp

**Date**: 2026-07-09  
**Canonical source**: [`contracts/quality-report.schema.json`](contracts/quality-report.schema.json)  
**Scope**: Entities required for v0.1 Fast Mode MVP (fixed pipeline, 2 detectors)

---

## Entity Relationship

```text
SingleFrameInput
    │
    ▼
GlobalScanOutput ──► RegionNomination[] ──► EdgeBleedInput
    │                                        │
    │                                        ▼
    │                                   EdgeBleedOutput ──► EdgeDegradation[]
    │
    └──► CompressionArtifactInput ──► CompressionArtifactOutput ──► CompressionDegradation[]
                │
                ▼
         ReportInput ──► QualityReport
                              ├── degradations: DegradationItem[]
                              ├── decision_trace: TraceEntry[]
                              └── mos_breakdown: MOSBreakdown
```

---

## Input Layer

### SingleFrameInput

CLI loads disk file into this structure.

| Field | Type | Required | Validation |
|-------|------|----------|------------|
| `frame` | `np.ndarray` | ✅ | H×W×3, BGR, uint8 |
| `frame_id` | `str` | ✅ | Unique per run; default = filename stem |
| `source_info` | `SourceInfo \| None` | ❌ | From `--metadata` |
| `ignore_regions` | `list[BBox] \| None` | ❌ | From `--ignore-regions`; each bbox x,y,w,h ≥ 0 |
| `mode` | `"fast"` | ✅ | MVP: only `"fast"` accepted |

```python
BBox = tuple[int, int, int, int]  # x, y, w, h
```

### BadcaseMetadata (sidecar)

Subset of [`USE_CASE_BADCASE.md`](../USE_CASE_BADCASE.md) §3 used in MVP:

| Field | Type | Used by |
|-------|------|---------|
| `frame_id` | `str` | Report `video_id` fallback |
| `resolution` | `tuple[int, int]` | Evidence detail text |
| `bitrate_kbps` | `int \| None` | compression root_cause tree |
| `codec` | `str \| None` | evidence detail |
| `has_overlay` | `bool` | text_ui hint |
| `overlay_types` | `list[str] \| None` | trace logging |

### SourceInfo

| Field | Type | Notes |
|-------|------|-------|
| `video_path` | `str \| None` | Original video path if known |
| `timestamp_ms` | `float \| None` | Frame timestamp |
| `resolution` | `tuple[int, int]` | (width, height) |
| `codec` | `str \| None` | e.g. h264 |
| `generator` | `str \| None` | Pipeline identifier |

---

## Stage 1: GlobalScan

### RegionType (enum)

| Value | Label | MVP used |
|-------|-------|----------|
| `face` | 1 | ❌ (not nominated in MVP) |
| `hair` | 2 | ❌ |
| `hand` | 3 | ❌ |
| `edge` | 4 | ✅ primary |
| `background` | 5 | ✅ optional |
| `body` | 6 | ❌ |
| `text_ui` | 7 | ✅ mask only |

### RegionNomination

| Field | Type | Validation |
|-------|------|------------|
| `region_type` | `RegionType` | Must be `edge` or `background` in MVP |
| `bbox` | `BBox` | Clipped to frame bounds |
| `mask` | `np.ndarray` | bool, H×W |
| `anomaly_score` | `float` | [0, 1] |
| `confidence` | `float` | [0, 1] |
| `suggested_detectors` | `list[str]` | MVP: `["edge_bleed"]` or `["compression_artifact"]` |
| `features` | `dict` | e.g. `green_channel_bias`, `blockiness_hint` |

### GlobalScanOutput

| Field | Type | Notes |
|-------|------|-------|
| `frame_index` | `int` | Default 0 for single frame |
| `segmentation_map` | `np.ndarray` | uint8, RegionType values |
| `global_quality_score` | `float` | [0, 1] coarse score |
| `is_fast_pass` | `bool` | MVP: always `False` (detectors always run) |
| `is_fast_reject` | `bool` | True if text_ui ratio ≥ 0.6 |
| `nominations` | `list[RegionNomination]` | ≥1 edge nomination if foreground detected |
| `scan_duration_ms` | `float` | Stage timing |

---

## Stage 2: Detector Outputs

### Evidence (canonical L2)

| Field | Type | Required | Rules |
|-------|------|----------|-------|
| `method` | `str` | ✅ | e.g. `lab_color_diff_alpha_edge`, `dct_blockiness` |
| `metric` | `str` | ✅ | e.g. `delta_e_mean`, `blockiness_score` |
| `value` | `float` | ✅ | Measured metric |
| `threshold` | `float` | ✅ | Config threshold used for decision |
| `detail` | `str` | ✅ | **Chinese**, human-readable |
| `attention_map` | `str \| None` | ❌ | MVP: null |

### RootCauseHypothesis

| Field | Type | MVP mapping |
|-------|------|-------------|
| `cause` | `RootCauseCategory` | edge → `matting_error`; block → `encoding_loss` |
| `confidence` | `float` | [0, 1] |

### DegradationItem (canonical report row)

| Field | Type | Required | Notes |
|-------|------|----------|-------|
| `degradation_id` | `str` | ✅ | UUID or `{detector}_{idx}` |
| `region_type` | `RegionType` | ✅ | |
| `degradation_type` | `str` | ✅ | e.g. `green_edge`, `encoding_block_artifact` |
| `severity` | `Severity` | ✅ | good/minor/moderate/severe/critical |
| `confidence` | `float` | ✅ | [0, 1] |
| `mos_impact` | `float` | ✅ | Negative float；仅用于 rule 后端 MOS 求和，非感知归因 |
| `bbox` | `BBox` | ✅ | Union of affected area |
| `frame_indices` | `list[int]` | ✅ | `[0]` for single frame |
| `description` | `str` | ✅ | Short Chinese summary |
| `detector` | `str` | ✅ | `edge_bleed` \| `compression_artifact` |
| `evidence` | `Evidence` | ✅ | Four elements + detail |
| `root_cause_hypothesis` | `RootCauseHypothesis` | ✅ | |
| `vlm_reasoning` | `VLMReasoning \| None` | ❌ | MVP: always null |

### Severity → mos_impact (MVP defaults from edge_bleed spec)

> **MOS 与归因解耦**：`mos_impact` 仅用于 rule 后端把各劣化项叠加成帧级 MOS 总分（`base_mos + Σ penalty × decay_factor^i`），是工程启发式求和，**非感知归因**。真正的归因（劣化是什么 / 在哪 / 为什么）看 `degradations[]`（detector / bbox / evidence / root_cause / vlm_reasoning），与 `mos_impact` 无关。`mos_model=clip_iqa` 时由 CLIP-IQA 直接预测总分，不使用本表。

| Severity | Typical mos_impact |
|----------|-------------------|
| minor | -0.2 |
| moderate | -0.3 |
| severe | -0.4 |
| critical | -0.8 |

---

## Stage 3: QualityReport

### QualityReport (top-level output)

| Field | Type | Required (MVP) |
|-------|------|----------------|
| `report_id` | `str` (UUID) | ✅ |
| `video_id` | `str` | ✅ (frame_id or session_id) |
| `mode` | `"fast"` | ✅ |
| `frame_index` | `int` | ✅ |
| `report_timestamp` | `str` (ISO 8601) | ✅ |
| `system_version` | `str` | ✅ e.g. `0.1.0` |
| `overall_mos` | `float` | ✅ [1.0, 5.0] |
| `severity` | `Severity` | ✅ aggregate |
| `mos_breakdown` | `MOSBreakdown` | ✅ |
| `degradations` | `list[DegradationItem]` | ✅ |
| `degradation_summary` | `DegradationSummary` | ✅ |
| `decision_trace` | `list[TraceEntry]` | ✅ L1 minimum |
| `vlm_reasoning_summary` | `list \| None` | ✅ null in MVP |
| `performance` | `PerformanceMetrics` | ✅ |

### MOSBreakdown

| Field | Type |
|-------|------|
| `base_mos` | `float` (default 4.5) |
| `total_penalty` | `float` (≤ 0) |
| `cap_applied` | `bool` |
| `cap_reason` | `str \| None` |
| `penalties` | `list[PenaltyItem]` |

### TraceEntry (L1)

| Field | Type | MVP stages used |
|-------|------|-----------------|
| `stage` | enum | `mode_select`, `global_scan`, `routing`, `detection`, `aggregation` |
| `module` | `str` | e.g. `fast_pipeline`, `edge_bleed` |
| `timestamp_ms` | `float` | Relative to pipeline start |
| `duration_ms` | `float` | |
| `input_summary` | `dict` | Serializable snapshot |
| `output_summary` | `dict` | |
| `decision` | `str` | Human-readable |
| `mode` | `"fast"` | |

### PerformanceMetrics

| Field | Type |
|-------|------|
| `total_ms` | `float` |
| `global_scan_ms` | `float` |
| `detection_ms` | `float` |
| `aggregation_ms` | `float` |

### DegradationSummary

| Field | Type |
|-------|------|
| `total_count` | `int` |
| `by_severity` | `dict[str, int]` |
| `by_detector` | `dict[str, int]` |
| `by_root_cause` | `dict[str, int]` |
| `top_issues` | `list[str]` (max 3) |

---

## Config Model (`config.yaml`)

Grouped `[CONFIG]` keys — single source for thresholds:

```yaml
global_scan:
  edge_expand_px: 10
  nomination_threshold: 0.3
  text_ui_ratio_critical: 0.6

edge_bleed:
  green_spill_minor: 0.05
  green_spill_moderate: 0.15
  green_spill_critical: 0.30
  delta_e_spill_threshold: 10.0

compression:
  blockiness_threshold: 1.8
  blockiness_coarse_threshold: 1.4

report:
  base_mos: 4.5
  decay_factor: 0.7
```

---

## State Transitions

### Pipeline state (in-memory)

```text
INIT → LOADED → SCANNED → DETECTED → REPORTED → EMITTED
```

| Transition | Trigger | Error path |
|------------|---------|------------|
| INIT→LOADED | Image read success | FileNotFound → CLI exit 1 |
| LOADED→SCANNED | GlobalScan complete | Segmentation fail → fallback full-frame edge mask + trace warning |
| SCANNED→DETECTED | Both detectors finish | Single detector fail → skip + trace `skipped` |
| DETECTED→REPORTED | ReportGenerator | Always succeeds with empty degradations allowed |
| REPORTED→EMITTED | JSON/HTML write | IO error → exit 1 |

### Batch mode

Each frame independent; corrupt frame → log + continue; summary exit 0 if ≥1 success.

---

## Validation Rules (implementer checklist)

1. Every `degradations[].evidence` has non-empty Chinese `detail`
2. `overall_mos = max(1.0, min(cap, base_mos + total_penalty))`
3. `bbox` width/height > 0 when degradation present
4. `detector` ∈ 9 类检测器集合（MVP 原始切片为 `{edge_bleed, compression_artifact}`，后续扩展至 9 类，见 README 检测器表）
5. JSON output validates against `contracts/quality-report.schema.json`
