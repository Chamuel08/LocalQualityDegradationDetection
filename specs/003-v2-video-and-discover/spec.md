# Feature Specification: V2 视频与主动发现（VideoClipRunner + TemporalFlicker + vlm_discover）

**Feature Branch**: `003-v2-video-and-discover`

**Created**: 2026-07-20

**Status**: Implemented（代码已落地，单元测试 + live VLM 验证通过）

**Depends on**: [`002-v1-agent-layer`](../002-v1-agent-layer/spec.md) — ReAct Agent 编排层与 `AgentContext` / `agent_meta` 必须先可用

**Input**: 在 V1 单帧 ReAct Agent 之上引入 **V2 三项能力**：

1. **`vlm_discover` 工具** — ReAct Agent 新增一个 action，让 VLM 对**全帧**主动扫描，发现 CV 规则检测器检不到的语义异常（AI 生成多指、面部生成错误、背景穿插等），结果写入 `agent_meta.vlm_discover_findings`。
2. **`VideoClipRunner`** — 单帧 pipeline 的外层包装器，支持多帧（视频 clip）输入：逐帧独立调用 `pipeline.run(SingleFrameInput)`，不改任何单帧内部接口。
3. **`TemporalFlicker`** — 帧间聚合层检测器，对相邻帧计算亮度/色相跳变，仅在 `VideoClipRunner` 层调用，不进入单帧 `ALL_DETECTOR_NAMES`。

> **设计取舍**：V2 不重构整个调度层为 Deep Mode（VLM 先行 + 子检测器量化），而是用 `vlm_discover` 这一低改动工具覆盖「CV 盲区主动发现」核心需求；Deep Mode 推迟。PatchCore 等需正常样本 memory bank 的异常检测**不实现**（违反系统「无参考」核心原则）。

---

## User Scenarios & Testing *(mandatory)*

### User Story 1 - VLM 主动发现 CV 盲区异常 (Priority: P1)

CV 规则检测器对 AI 生成内容特有的语义异常（多指、面部生成错误）无感知；Agent 在观察到「MOS 低且检出为 0」或「画面含 AI 生成信号」时，可自主选择 `vlm_discover` 让 VLM 对全帧主动扫描，发现这些异常并产出结构化 findings。

**Why this priority**: 补 CV 规则盲区是 V2 的核心增量价值；`vlm_discover` 是 ReAct Agent 工具集的自然扩展，改动面小。

**Independent Test**: 直接调 `_execute_vlm_discover` + live `OllamaVLMClient(qwen2.5vl:7b)`，对 `synthetic_portrait.png` 全帧扫描；返回 `hand_extra_finger` finding，写入 `ctx.vlm_discover_findings`，trace 含 `vlm_discover` stage。

**Acceptance Scenarios**:

1. **Given** MOS ≤ 3.5 且 detection_count == 0，**When** LLM 决策，**Then** 可选 `vlm_discover`，执行后 `agent_meta.vlm_discover_findings` 非空
2. **Given** 同一帧已执行过 `vlm_discover`，**When** 再次选择该 action，**Then** 跳过并返回「本帧已执行过主动发现」
3. **Given** VLM 不可用，**When** 执行 `vlm_discover`，**Then** trace 记 `vlm_discover_failed`，不阻塞主链路
4. **Given** VLM 返回非 JSON / 解析失败，**When** 解析，**Then** 记录 raw 截断，不崩溃

---

### User Story 2 - 视频 clip 多帧输入与逐帧检测 (Priority: P1)

用户输入一段视频 clip，系统均匀抽帧后逐帧跑单帧 pipeline，返回 `VideoClipReport`（逐帧报告 + 聚合 MOS + 最差帧 + 跨帧 degradation 汇总）。

**Why this priority**: V2 把输入从单帧扩展到多帧，是版本边界的核心扩展。

**Independent Test**: `VideoClipRunner` + stub pipeline（不依赖 Ollama），3 帧预设 MOS；`aggregate_mos` = 均值，`worst_frame_mos`/`worst_frame_index` 正确，`degradation_summary` 跨帧计数正确。

**Acceptance Scenarios**:

1. **Given** 3 帧输入，**When** `runner.run(frames)`，**Then** `frame_count=3`，`aggregate_mos` = 逐帧 MOS 均值
2. **Given** 空帧列表，**When** `runner.run([])`，**Then** 抛 `ValueError`
3. **Given** 未提供 `frame_ids`，**When** run，**Then** 自动补全为 `{clip_id}_f{i:04d}`
4. **Given** 视频文件路径，**When** `sample_frames_from_video(path, max_frames=8)`，**Then** 返回 ≤ 8 帧 BGR uint8 列表；文件不存在抛 `FileNotFoundError`
5. **Given** CLI `detect.py --video input.mp4 --max-frames 16 --output r.json`，**When** 执行，**Then** 输出 `VideoClipReport` JSON（`aggregate_mos` / `worst_frame_mos` / `worst_frame_index` / `flicker` / `degradation_summary` / `frame_reports[]`）；`--legacy-fixed` 走无 Agent 快速管线，不加则走 V1 Agent

---

### User Story 3 - 时域闪烁检测（帧间聚合） (Priority: P2)

`TemporalFlicker` 在帧间层检测亮度/色相跳变，单帧无法做；结果以 `TemporalFlickerResult` 返回，含闪烁区间、比例、最大跳变。

**Why this priority**: 时序劣化是视频特有问题，但优先级低于空间劣化；仅作聚合层补充。

**Independent Test**: 5 帧（3 正常 + 1 亮闪 + 1 正常），`detect_temporal_flicker` 返回 `is_flickering=True`、≥ 2 段、`max_luma_delta > 8`；稳定 5 帧返回 `is_flickering=False`。

**Acceptance Scenarios**:

1. **Given** 稳定帧序列（无亮度/色相跳变），**When** 检测，**Then** `is_flickering=False`，`flicker_segments=[]`
2. **Given** 含亮度跳变的帧序列，**When** 检测，**Then** `is_flickering=True`，`flicker_ratio > 0`，segments 的 `metric=luma_delta`
3. **Given** 单帧输入，**When** 检测，**Then** 返回空结果 `is_flickering=False`（至少 2 帧才能算帧间差）

---

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: `vlm_discover` MUST 作为 ReAct Agent 白名单 action 接入（`WHITELIST` 含 `vlm_discover`），由 LLM 自主决策触发
- **FR-002**: `vlm_discover` MUST 对整帧编码为 JPEG base64 后调 `VLMClient.confirm(VLM_DISCOVER_PROMPT, b64)`，解析返回的 `findings` 数组为 `VLMDiscoverFinding`
- **FR-003**: `vlm_discover` 每帧 MUST 最多执行 1 次（`ctx.vlm_discover_findings` 非空时跳过）；且受 `vlm.max_calls_per_frame` 总额限制
- **FR-004**: `vlm_discover` 的 findings MUST 写入 `agent_meta.vlm_discover_findings`，**不**合并进 `degradations[]`（无 bbox/mask）；`mos_impact_estimate` 仅参考，不参与 MOS 计算
- **FR-005**: `VideoClipRunner` MUST 对每帧独立调用 `pipeline.run(SingleFrameInput)`，不修改单帧 pipeline 内部接口
- **FR-006**: `VideoClipRunner` MUST 在帧间层调用 `detect_temporal_flicker`，返回 `VideoClipReport`（含 `aggregate_mos` / `worst_frame_mos` / `worst_frame_index` / `degradation_summary`）
- **FR-007**: `TemporalFlicker` MUST NOT 进入单帧 `ALL_DETECTOR_NAMES` / `build_detector_registry`（依赖多帧，单帧无法调用）
- **FR-008**: `TemporalFlicker` MUST 计算相邻帧灰度均值差（亮度跳变）与 HSV-H 通道均值差（色相漂移），任一超阈值即记一段 `FlickerSegment`
- **FR-009**: `is_flickering` MUST 在闪烁帧间比例 ≥ `min_flicker_ratio`（默认 0.15）时为真
- **FR-010**: `sample_frames_from_video` MUST 均匀抽取 ≤ `max_frames` 帧；文件不存在抛 `FileNotFoundError`，无法打开抛 `RuntimeError`
- **FR-011**: CLI `detect.py` MUST 暴露 `--video PATH` 与 `--max-frames N` 入口，复用 `VideoClipRunner` + `sample_frames_from_video`，输出 `VideoClipReport` JSON（与 GUI `run_video` 序列化结构一致）

### Out of Scope (003)

- Deep Mode（VLM 先行 + 子检测器量化，重构整个调度层）— 推迟；`vlm_discover` 已覆盖核心主动发现需求
- PatchCore / 预训练异常检测 — 不实现（违反「无参考」原则，需正常样本 memory bank）
- `TemporalFlicker` 的光流时序方差路径（`method_selection.md` 中备选）— 当前仅实现亮度/色相跳变，光流路径未实现

## Success Criteria *(mandatory)*

- **SC-001**: `vlm_discover` live VLM 验证通过（对 `synthetic_portrait.png` 主动发现 `hand_extra_finger`，写入 `vlm_discover_findings`）
- **SC-002**: `VideoClipRunner` 端到端跑通（5 帧 clip，`aggregate_mos` / flicker / 跨帧汇总正确）
- **SC-003**: `TemporalFlicker` 单元测试通过（稳定帧无闪烁 / 注入跳变检出 / 单帧兜底）
- **SC-004**: 新增单元测试 `tests/unit/test_video_clip_runner.py` 全部通过；不依赖 Ollama 的测试用 stub pipeline
- **SC-005**: v1 schema 扩展 `vlm_discover` action enum 与 `vlm_discover_findings` 属性，真实报告通过校验

## Assumptions

- V1 ReAct Agent 编排层（`AgentOrchestrator` / `AgentContext` / `agent_meta`）已可用
- `vlm_discover` 复用 V1 的 `OllamaVLMClient`（同一 `qwen2.5vl:7b` 模型），不引入新外部服务
- `VideoClipRunner` 可包装任意实现了 `.run(SingleFrameInput) -> QualityReport` 的 pipeline（`AgentPipeline` 或 `FastPipeline`）
- 视频抽帧依赖 OpenCV（`cv2.VideoCapture`），已是核心依赖

## Design References

| 文档 | 用途 |
|------|------|
| [`VERSION_ROADMAP.md`](../VERSION_ROADMAP.md) | V2 版本边界与实现决策记录（第 8 节） |
| [`data-model.md`](data-model.md) | `VLMDiscoverFinding` / `VideoClipReport` / `FlickerSegment` / `TemporalFlickerResult` 实体 |
| [`tasks.md`](tasks.md) | 任务清单 |
| [`../002-v1-agent-layer/contracts/quality-report.v1.schema.json`](../002-v1-agent-layer/contracts/quality-report.v1.schema.json) | 扩展报告 Schema（含 `vlm_discover` action 与 `vlm_discover_findings`） |
| [`../method_selection.md`](../method_selection.md) | `TemporalFlicker` 选型（光流 vs 亮度/色相跳变） |
