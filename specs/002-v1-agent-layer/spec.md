# Feature Specification: V1 Agent Layer（ReAct Agent 自主决策）

**Feature Branch**: `002-v1-agent-layer`

**Created**: 2026-07-09

**Last Updated**: 2026-07-14（编排层由早期「灰区 VLM + LLM Judge + Round 2」演进为 ReAct Agent 循环）

**Status**: Implemented（tasks.md 已完成；编排层已重构为 ReAct Agent，单元 + mock 集成测试通过）

**Depends on**: [`001-v0-fast-mvp`](../001-v0-fast-mvp/spec.md) — v0.1 fixed pipeline 与 Report Schema 必须先可用

**Input**: 为 badcase 批量筛图引入 **V1 Fast Mode ReAct Agent 编排层**：LLM（Qwen2.5-1.5B）在观察 CV 检测器全量结果后，自主推理并选择工具调用（VLM 视觉确认 / 重检 / 补检 / 接受），直到产出 `accept` 终止。技术 HOW 以本 feature 的 `plan.md`、`contracts/` 及 [`VERSION_ROADMAP.md`](../VERSION_ROADMAP.md) 为准。

> **架构演进说明（重要）**：初版 V1 设计为「confidence ∈ [0.4, 0.7] 灰区自动触发 VLM Confirm → LLM Judge 审查 Round 1 → Round 2 白名单补检，最多 2 轮」。实现过程中将其重构为 **ReAct Agent**：
> - VLM 是否调用、是否补检，**均由 LLM 在每一步自主决策**（`JudgeClient.decide()`），灰区阈值仅保留作**规则降级参考**（Ollama 不可用时由 `RuleBasedJudgeClient` 使用）。
> - 主链路**不再经过独立 Judge 阶段**；`agent_meta.judge_assessment` 在 ReAct 模式下为 `null`。
> - 旧的 `run_judge` / Round 2 执行器（`actions.py`）代码保留作**向后兼容降级路径**，非默认主链路。
>
> 本 spec 已按 ReAct 架构改写；早期灰区/Judge/Round-2 表述仅在本说明中保留作演进背景，不再作为有效需求。

---

## User Scenarios & Testing *(mandatory)*

### User Story 1 - LLM 自主决策是否调用 VLM (Priority: P1)

算法工程师批量筛图时，CV 检测器产出初步 degradations；LLM Agent 观察置信度与证据后，**自主判断**是否对某检测项调用 VLM 做 ROI 视觉确认，并在报告中给出 L3 语义解释。

**Why this priority**: badcase 兜底的核心价值——由 LLM 判断「视觉确认是否值得」，避免硬编码阈值在边界 case 上的误触发/漏触发。

**Independent Test**: `MockJudgeClient.decide` 返回 `vlm_analyze`；输入含低置信度项的检测结果；JSON 含 `vlm_reasoning` 与 `decision_trace` stage=`agent_step`（`agent_vlm_analyze_stepN`）。

**Acceptance Scenarios**:

1. **Given** 存在置信度偏低的检测项，**When** ReAct Agent 流程，**Then** LLM 可选择 `vlm_analyze`，触发 VLM Confirm，degradation 含 L3 字段，`agent_meta.agent_driven_vlm=true`
2. **Given** 全部高置信度且 LLM 判定可信，**When** 同上，**Then** LLM 可直接 `accept`，不调用 VLM，`agent_driven_vlm=false`
3. **Given** Ollama 不可用，**When** Agent 流程，**Then** 降级到 `RuleBasedJudgeClient` 规则决策，trace 记 `vlm_failed: service_unavailable`，流水线不阻塞

---

### User Story 2 - ReAct Agent 工具循环与补检 (Priority: P1)

LLM Agent 在每一步可从工具集 `{vlm_analyze, rerun_detector, dispatch_compression, accept}` 中选择一个执行；执行后将 observation 反馈给 LLM 进入下一步，直到 `accept` 或达最大步数。

**Why this priority**: Agent 自主决策能力；解决「MOS 低但无检出」等矛盾，并支持怀疑漏检时主动补检。

**Independent Test**: Mock/Rule `decide` 返回 `dispatch_compression`；MOS 偏低且无 compression 检出；Agent 执行后报告含 `compression_artifact`。

**Acceptance Scenarios**:

1. **Given** 所有结果可信，**When** Agent 决策，**Then** 第一步即 `accept`，`agent_steps` 长度 = 1
2. **Given** MOS 偏低且无 compression 检出，**When** Agent 决策，**Then** 可选 `dispatch_compression`，执行后 `merged_degradations` 含 `compression_artifact`
3. **Given** LLM 输出非法/不可解析 action，**When** 解析，**Then** 强制 `accept`，trace 记录，不阻塞
4. **Given** 同一 action 在历史中连续出现 ≥ 2 次，**When** 防护触发，**Then** 强制 `accept` 终止循环（防小模型陷入死循环）

---

### User Story 3 - decision_trace 完整可回溯 (Priority: P2)

复核人员需从 JSON 看到 Agent 每一环：路由、检测、**每个 agent_step（thought / action / observation）**、聚合。

**Independent Test**: 含 `vlm_analyze` 的样本；trace 含 `routing`、`detection`、`agent_step × N`、`aggregation`，且 `agent_meta.agent_steps` 完整可读。

---

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: V1 Fast Mode MUST 使用 `AgentOrchestrator`（ReAct Agent）替代 v0.1 fixed pipeline
- **FR-002**: VLM 是否调用 MUST 由 LLM Agent 自主决策（`decide()` 接口），**而非硬编码灰区阈值**；灰区阈值仅作规则降级参考
- **FR-003**: Agent 每步 MUST 调用 LLM `decide()`（Qwen2.5-1.5B 或配置等价物），输出 `{thought, action, ...}` 严格 JSON
- **FR-004**: Agent 可选工具 MUST 来自白名单 `{vlm_analyze, rerun_detector, dispatch_compression, accept}`；非法/不可解析 action 强制 `accept`
- **FR-005**: Agent 最大步数 = `max_rounds × 3`（默认 `max_rounds=2` → 6 步）；超限或 `accept` 后终止并出报告
- **FR-006**: 单帧 VLM 调用 MUST 受 `vlm.max_calls_per_frame` 限制（默认 3），超出后该工具跳过并 trace 记 `vlm_skipped: quota_exceeded`
- **FR-007**: VLM/LLM 不可用 MUST 降级且不阻塞主链路（trace 记原因）；LLM 不可用时降级到 `RuleBasedJudgeClient`，再不可用则直接 `accept`
- **FR-008**: Report MUST 填充 `vlm_reasoning`（Agent 触发 VLM 的 case）、`agent_meta.agent_steps`（完整 thought/action/observation/latency）与完整 `decision_trace`
- **FR-009**: LLM Agent MUST NOT 替代 `ReportGenerator` 的 MOS 聚合公式（MOS 仅由 `compute_mos()` 计算，支持 `rule` / `clip_iqa` / `internal` 三种后端）

### Out of Scope (002)

- Deep Mode 全量 VLM 粗分（V2，已推迟；`vlm_discover` 已覆盖核心主动发现需求）
- LLM Judge 超过 2 轮迭代（ReAct 已替代该机制）
- `hand_anomaly` CV 检测器的多指计数 / 粘连 / 手部模糊（实验性 MVP，CV 路径未实现；但 `vlm_discover` 工具可由 VLM 主动发现手部语义异常如多指，见 V2）
- ~~V2 视频 / TemporalFlicker~~ → **已实现**（V2：`VideoClipRunner` + `TemporalFlicker`，见 [`VERSION_ROADMAP.md`](../VERSION_ROADMAP.md) 第 8 节）

## Success Criteria *(mandatory)*

- **SC-001**: `agent_driven_vlm` 可统计（Agent 自主触发 VLM 的帧比例可从 `agent_meta` 聚合）
- **SC-002**: Agent actions 100% 落在白名单内或被强制 `accept`（不崩溃；fuzz/异常输入验证）
- **SC-003**: 「MOS 低无检出」矛盾 case 中 Agent 可触发 `dispatch_compression`（mock 验证）
- **SC-004**: VLM/LLM mock 集成测试通过；`pytest -m vlm` 可选运行
- **SC-005**: `decision_trace` 含 `agent_step` 条目且 `agent_meta.agent_steps` 可人工读懂

## Assumptions

- Ollama 本地可跑 `qwen2.5vl:7b` 与 `qwen2.5:1.5b`，或云端 OpenAI 兼容 API
- v0.1 的 QualityReport Schema 扩展 `vlm_reasoning` / `agent_meta` 字段，不破坏现有 golden sample
- CI 默认 `pytest -m "not vlm"`；VLM 测试 mock

## Design References

| 文档 | 用途 |
|------|------|
| [`VERSION_ROADMAP.md`](../VERSION_ROADMAP.md) | 版本边界 |
| [`plan.md`](plan.md) | Agent 编排实现计划 |
| [`contracts/llm-judge.schema.json`](contracts/llm-judge.schema.json) | 旧 Judge 输出 Schema（向后兼容降级路径） |
| [`contracts/vlm-confirm.schema.json`](contracts/vlm-confirm.schema.json) | VLM Confirm Schema |
| [`contracts/quality-report.v1.schema.json`](contracts/quality-report.v1.schema.json) | 扩展报告 Schema（含 `agent_step` / `agent_steps` / `agent_driven_vlm`） |
