# Research: 001-v0-fast-mvp

**Date**: 2026-07-09  
**Feature**: v0.1 Fast Mode MVP  
**Sources**: [`method_selection.md`](../method_selection.md), `data-model.md`, feature spec

---

## R1. GlobalScan segmentation (MVP simplification)

**Decision**: Use **MediaPipe Selfie Segmentation** (`model_selection=1`, landscape) as primary foreground mask; derive **edge band** via morphological gradient / contour ± `edge_expand_px` (default 10).

**Rationale**:
- Feature spec allows "简化版 face parsing + edge 提名"; full CelebAMask-HQ adds model download, GPU dependency, and hand ROI complexity not needed for edge_bleed + compression MVP.
- MediaPipe runs on CPU in ~20–40ms for 720p; meets GlobalScan < 100ms budget with nomination.
- Edge band from foreground mask matches edge_bleed input contract (`edge_mask`, `foreground_mask`).

**Alternatives considered**:
| Alternative | Rejected because |
|-------------|------------------|
| CelebAMask face parsing | Higher setup cost; hair/face nominations unused in v0.1 pipeline |
| Full-frame edge detection (Canny) | No foreground/background separation; high FP on texture |
| Skip segmentation; full-frame edge_bleed | Violates coarse-to-fine design; hurts explainability (no region_type) |

**MVP nomination rule**: Always nominate `edge` region if foreground area > 1% of frame; add `background` nomination only if blockiness_hint from coarse HF scan > 0.5 (optional optimization).

---

## R2. EdgeBleed algorithm

**Decision**: Implement **Lab ΔE + green channel excess** per [`method_selection.md`](../method_selection.md) §3.

**Rationale**:
- Canonical method already selected; ΔE > 10 maps to visible spill (MOS -0.2~0.4).
- Green spill is primary P1 acceptance scenario (User Story 1).
- Lightweight: OpenCV `cvtColor` BGR→Lab, vectorized pixel stats on edge band mask.

**MVP scope cut**: Defer §4.2 black edge, §4.3 matting_trace, §4.4 blending_artifact to post-MVP unless trivial. P1 acceptance only requires green_edge / color_spill.

**Key thresholds** (from design, via config):
- `green_spill_moderate` = 0.15, `delta_e_spill_threshold` = 10.0
- `edge_expand_px` = 10

**Alternatives considered**:
| Alternative | Rejected because |
|-------------|------------------|
| RGB G-channel ratio only | method_selection §3: luminance-chroma coupling, FP on green clothing |
| MODNet matting re-estimate | ~50ms extra; not needed for green edge demo |

---

## R3. CompressionArtifact algorithm

**Decision**: Implement **DCT block boundary energy ratio** on Y channel; map to `encoding_block_artifact`.

**Rationale**:
- Independent detector aligns with encoding_loss root_cause attribution.
- Pure NumPy/OpenCV — no ML model required for MVP blockiness signal.
- User Story 4 acceptance: `blockiness_score` in evidence.

**MVP scope**:
- Full-frame blockiness + bbox from connected components above threshold
- Defer §4.2 local BRISQUE to stretch goal (requires opencv-contrib `quality` module)
- Defer §4.3 mosquito, §4.4 chroma_block to post-MVP

**Key thresholds**:
- `blockiness_threshold` = 1.8 (detection)
- `blockiness_coarse_threshold` = 1.4 (candidate regions)

**Alternatives considered**:
| Alternative | Rejected because |
|-------------|------------------|
| Embed blockiness in HairTexture only | COMPRESSION doc 方案 B rejected: cross-ROI block missed |
| Whole-image BRISQUE | NSS prior invalid on synthetic/composited frames |

---

## R4. Pipeline orchestration

**Decision**: **Fixed linear pipeline** in `fast_pipeline.py` — no AgentOrchestrator, no VLM, no LLM Judge.

**Flow**:
```
load image → GlobalScan → parallel(EdgeBleed(edge nomination), CompressionArtifact(full frame))
         → ReportGenerator → JSON | HTML
```

**Rationale**: Feature spec Out of Scope explicitly allows fixed pipeline for v0.1. Avoids building routing/VLM infrastructure before core detectors validated.

**Routing trace (L1 minimum)**:
- `routing` stage: `decision="fixed_pipeline: edge_bleed + compression_artifact"`

**Alternatives considered**:
| Alternative | Rejected because |
|-------------|------------------|
| Minimal AgentOrchestrator stub | YAGNI — adds indirection without dynamic behavior in v0.1 |
| Sequential only | Parallel is trivial with 2 detectors; design allows parallel |

---

## R5. Report & MOS aggregation

**Decision**: Implement report_generator §9 formula: `base_mos=4.5`, penalty decay `0.7^i`, caps (fast_reject → 3.0, critical → 2.5).

**Rationale**: Constitution III requires explainability; MOS breakdown in trace supports demo narrative.

**MVP**: No `vlm_reasoning_summary`; `decision_trace` stages limited to: `mode_select`, `global_scan`, `routing`, `detection`, `aggregation`.

---

## R6. CLI & packaging

**Decision**: Root `detect.py` + `pyproject.toml` with `[project.scripts]` optional alias `lqdd= lqdd.cli.main:main`.

**Dependencies (pinned ranges in pyproject)**:
- `opencv-python-headless>=4.8`
- `numpy>=1.24`
- `mediapipe>=0.10`
- `pyyaml>=6.0`
- `jsonschema>=4.20`
- dev: `pytest>=7.4`

**Rationale**: Constitution II CLI-first; headless OpenCV for CI/server; no GUI.

---

## R7. HTML report

**Decision**: Single static HTML file — inline CSS, base64 PNG preview, SVG/canvas bbox overlay, degradation table with evidence columns.

**Rationale**: User Story 3; no web server. Jinja2 optional; stdlib string template acceptable for MVP.

**Alternatives considered**:
| Alternative | Rejected because |
|-------------|------------------|
| Separate JSON + viewer app | Over scope; spec requires `--output report.html` standalone |

---

## R8. Sample data layout

**Decision**: Public sample layout under `data/sample/`:

```
data/sample/frames/{edge,block,normal}/*.png
data/sample/expected/*.json
```

Minimum 15 frames (5+5+5) for SC-002~004; ≥3 golden JSON for SC-006 schema validation.

**Synthetic generation**: Script optional in `scripts/generate_synthetic_samples.py` — programmatic green border + JPEG Q=20 block patterns for CI reproducibility.

---

## Resolved Clarifications

All Technical Context items resolved; no remaining NEEDS CLARIFICATION blockers for `/speckit-tasks`.
