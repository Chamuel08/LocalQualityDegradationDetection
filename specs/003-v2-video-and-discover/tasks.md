# Tasks: V2 视频与主动发现（VideoClipRunner + TemporalFlicker + vlm_discover）

**Input**: Design documents from `specs/003-v2-video-and-discover/`
**Prerequisites**: spec.md, data-model.md
**Depends on**: [`002-v1-agent-layer`](../002-v1-agent-layer/) ReAct Agent 编排层已落地

**Organization**: Tasks grouped by user story (US1 → US2 → US3)。US1 扩展 ReAct Agent 工具集；US2 引入多帧聚合层；US3 在 US2 之上加帧间检测。

> **实现状态说明（2026-07-20）**：V2 三项能力代码已落地，单元测试 + live VLM 验证通过。下方任务勾选状态反映实际交付。

## Format: `[ID] [P?] [Story] Description`

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: V2 包结构、配置扩展、测试夹具

- [X] T201 Create `src/lqdd/temporal_flicker/` package tree（`__init__.py` + `detector.py`）
- [X] T202 Create `src/lqdd/pipeline/video_clip_runner.py` module
- [X] T203 [P] Extend `config.example.yaml` with `video` section（`max_frames` / `flicker_luma_threshold` / `flicker_hue_threshold` / `flicker_min_ratio`）— 若无新增则记为 N/A
- [X] T204 [P] Create `tests/unit/test_video_clip_runner.py` test skeleton（stub pipeline，不依赖 Ollama）

---

## Phase 2: User Story 1 — vlm_discover 工具 (Priority: P1) 🎯 MVP

**Goal**: ReAct Agent 白名单新增 `vlm_discover` action，VLM 对全帧主动扫描，findings 写入 `agent_meta.vlm_discover_findings`

**Independent Test**: 直接调 `_execute_vlm_discover` + live `OllamaVLMClient(qwen2.5vl:7b)`，对 `synthetic_portrait.png` 全帧扫描返回 `hand_extra_finger` finding

### Implementation for User Story 1

- [X] T205 [US1] Add `VLMDiscoverFinding` dataclass in `src/lqdd/models/agent.py`
- [X] T206 [US1] Add `vlm_discover` to `AgentAction` Literal whitelist in `src/lqdd/models/agent.py`
- [X] T207 [US1] Add `vlm_discover_findings: list[VLMDiscoverFinding]` to `AgentContext` and `AgentMeta` serialization
- [X] T208 [US1] Implement `VLM_DISCOVER_PROMPT` in `src/lqdd/agent/prompts.py`（要求严格 JSON：`findings[]` + `overall_assessment`）
- [X] T209 [US1] Implement `_execute_vlm_discover` in `src/lqdd/agent/orchestrator.py`：整帧 JPEG base64 → `VLMClient.confirm` → 解析 findings → 写入 `ctx.vlm_discover_findings`
- [X] T210 [US1] Enforce per-frame once constraint（`ctx.vlm_discover_findings` 非空时跳过，返回「本帧已执行过主动发现」）
- [X] T211 [US1] Emit `vlm_discover` / `vlm_discover_complete` / `vlm_discover_failed` trace entries
- [X] T212 [US1] VLM unavailable / JSON parse failure graceful degradation（不阻塞主链路）
- [X] T213 [P] [US1] Add unit test for `vlm_discover` mock flow in `tests/unit/test_video_clip_runner.py` or `tests/unit/test_orchestrator.py`

**Checkpoint**: US1 complete — `vlm_discover` 接入 ReAct Agent，mock + live VLM 验证通过

---

## Phase 3: User Story 2 — VideoClipRunner 多帧聚合 (Priority: P1)

**Goal**: 单帧 pipeline 的外层包装器，逐帧独立调用 `pipeline.run`，聚合 MOS + 跨帧 degradation 汇总

**Independent Test**: `VideoClipRunner` + stub pipeline（3 帧预设 MOS）；`aggregate_mos` = 均值，`worst_frame_mos`/`worst_frame_index` 正确

### Implementation for User Story 2

- [X] T214 [US2] Implement `VideoClipReport` dataclass in `src/lqdd/pipeline/video_clip_runner.py`
- [X] T215 [US2] Implement `VideoClipRunner.__init__(pipeline, flicker_luma_threshold, flicker_hue_threshold, flicker_min_ratio)`
- [X] T216 [US2] Implement `VideoClipRunner.run(frames, clip_id, frame_ids)`：逐帧 `pipeline.run(SingleFrameInput)` → 收集 `frame_reports`
- [X] T217 [US2] Implement `aggregate_mos` = 逐帧 `overall_mos` 均值；`worst_frame_mos` / `worst_frame_index`
- [X] T218 [US2] Implement `degradation_summary`：各 `degradation_type` 跨帧出现次数
- [X] T219 [US2] Auto-fill `frame_ids` = `{clip_id}_f{i:04d}` when not provided
- [X] T220 [US2] Empty frames → `ValueError`
- [X] T221 [US2] Implement `sample_frames_from_video(video_path, max_frames, start_sec, end_sec)`：均匀抽帧；`FileNotFoundError` / `RuntimeError`
- [X] T222 [P] [US2] Add unit tests in `tests/unit/test_video_clip_runner.py`：3 帧 stub / 空帧 ValueError / auto frame_ids / aggregate_mos / worst_frame

**Checkpoint**: US2 complete — `VideoClipRunner` 端到端跑通（stub pipeline，无 Ollama 依赖）

---

## Phase 4: User Story 3 — TemporalFlicker 帧间检测 (Priority: P2)

**Goal**: 帧间聚合层检测器，相邻帧亮度/色相跳变，仅在 `VideoClipRunner` 层调用

**Independent Test**: 5 帧（3 正常 + 1 亮闪 + 1 正常）→ `is_flickering=True`、≥ 2 段、`max_luma_delta > 8`；稳定 5 帧 → `is_flickering=False`

### Implementation for User Story 3

- [X] T223 [US3] Implement `FlickerSegment` / `TemporalFlickerResult` dataclasses in `src/lqdd/temporal_flicker/detector.py`
- [X] T224 [US3] Implement `detect_temporal_flicker(frames, luma_delta_threshold, hue_delta_threshold, min_flicker_ratio)`
- [X] T225 [US3] Algorithm: 灰度均值差 + HSV-H 通道均值差；超阈值记段；severity 分级
- [X] T226 [US3] `is_flickering` = `flicker_ratio >= min_flicker_ratio`；<2 帧返回空结果
- [X] T227 [US3] Wire `detect_temporal_flicker` into `VideoClipRunner.run`（帧间聚合层）
- [X] T228 [US3] Ensure `TemporalFlicker` NOT in `ALL_DETECTOR_NAMES` / `build_detector_registry`（单帧不可调用）
- [X] T229 [P] [US3] Add unit tests in `tests/unit/test_video_clip_runner.py`：稳定帧无闪烁 / 注入跳变检出 / 单帧兜底

**Checkpoint**: US3 complete — `TemporalFlicker` 帧间检测通过

---

## Phase 5: Schema & Documentation

**Purpose**: 扩展 v1 报告 schema，更新 README 与 VERSION_ROADMAP

- [X] T230 [P] Extend `specs/002-v1-agent-layer/contracts/quality-report.v1.schema.json`：`agent_steps[].action` enum 加 `vlm_discover`；新增 `vlm_discover_findings` 可选属性
- [X] T231 [P] Update `specs/002-v1-agent-layer/data-model.md`：`AgentAction` enum + `VLMDiscoverFinding` 实体 + `AgentMeta.vlm_discover_findings`
- [X] T232 [P] Update `specs/002-v1-agent-layer/spec.md`：Out of Scope 反映 V2 已实现
- [X] T233 [P] Update `specs/VERSION_ROADMAP.md`：V2 标记为 implemented
- [X] T234 Update `README.md`：特性 / ReAct Agent 工具表 / Python API 示例 / 项目状态表
- [X] T235 [P] Run `pytest tests/unit/test_video_clip_runner.py -q` 全绿
- [X] T236 Live VLM 验证 `vlm_discover` 对 `synthetic_portrait.png` 主动发现 `hand_extra_finger`

---

## Dependencies & Execution Order

### Phase Dependencies

- **Phase 1 Setup**: No dependencies
- **Phase 2 US1 (vlm_discover)**: Depends on V1 ReAct Agent 编排层
- **Phase 3 US2 (VideoClipRunner)**: Depends on Phase 1（可与 US1 并行）
- **Phase 4 US3 (TemporalFlicker)**: Depends on Phase 3（被 VideoClipRunner 调用）
- **Phase 5 Schema & Docs**: Depends on Phases 2–4

### User Story Dependencies

- **US1 (P1)**: 独立扩展 Agent 工具集 — **MVP part A**（VLM 主动发现）
- **US2 (P1)**: 独立多帧聚合层 — **MVP part B**（视频 clip 输入）
- **US3 (P2)**: 依赖 US2 — 帧间检测补充

### Parallel Opportunities

- Phase 1: T201, T202, T204 in parallel
- Phase 2 & Phase 3: US1 与 US2 可并行（US1 改 Agent，US2 加外层包装器，互不冲突）
- Phase 5: T230–T234 in parallel after Phases 2–4

---

## Implementation Strategy

### MVP First (US1 + US2)

1. Complete Phase 1 Setup
2. Complete Phase 2 (US1) + Phase 3 (US2) in parallel
3. **STOP and VALIDATE**: `pytest tests/unit/test_video_clip_runner.py -q`
4. Live VLM 验证 `vlm_discover`

### Incremental Delivery

1. Setup → 包结构 + 测试夹具
2. US1 → `vlm_discover` 接入 ReAct Agent
3. US2 → `VideoClipRunner` 多帧聚合
4. US3 → `TemporalFlicker` 帧间检测
5. Schema & Docs → v1 schema 扩展 + README + roadmap

### Suggested MVP Scope

**Minimum shippable**: Phase 1 + 2 + 3 + T227 + T235 — **vlm_discover + VideoClipRunner**
**Full 003 spec**: All phases — 含 TemporalFlicker + schema/docs

---

## Notes

- V2 不重构调度层为 Deep Mode（VLM 先行 + 子检测器量化）；`vlm_discover` 以低改动覆盖 CV 盲区主动发现核心需求
- PatchCore 等需正常样本 memory bank 的异常检测**不实现**（违反「无参考」核心原则）
- `TemporalFlicker` 仅实现亮度/色相跳变路径；光流时序方差路径未实现（Out of Scope）
- `vlm_discover_findings` **不**合并进 `degradations[]`（VLM 无像素级定位），仅写入 `agent_meta`
- `mos_impact_estimate` 仅参考记录，不参与最终 MOS 计算（MOS 由 `compute_mos()` 统一负责）
- CI default: `pytest -m "not vlm"`；live VLM 验证为手动 `@pytest.mark.vlm`
