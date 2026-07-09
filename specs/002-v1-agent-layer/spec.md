# Feature Specification: V1 Agent Layer（VLM 兜底 + LLM Judge）

**Feature Branch**: `002-v1-agent-layer`

**Created**: 2026-07-09

**Status**: Implemented（tasks.md 已完成，28 tests passed）

**Depends on**: [`001-v0-fast-mvp`](../001-v0-fast-mvp/spec.md) — v0.1 fixed pipeline 与 Report Schema 必须先可用

**Input**: 为 badcase 批量筛图引入 **V1 Fast Mode Agent 编排层**：VLM 灰区视觉兜底 + 小 LLM 对 pipeline 结果的整合与 Round 2 自我决策。技术 HOW 以本 feature 的 `plan.md`、`contracts/` 及 [`VERSION_ROADMAP.md`](../VERSION_ROADMAP.md) 为准。

---

## User Scenarios & Testing *(mandatory)*

### User Story 1 - 灰区 badcase 经 VLM 确认 (Priority: P1)

算法工程师批量筛图时，某 ROI 子检测器 confidence 落在 0.4–0.7 灰区，系统应调 VLM 对该 ROI 做二次视觉确认，并在报告中给出 L3 语义解释。

**Why this priority**: badcase 兜底的核心价值；避免纯规则在边界 case 上误报/漏报。

**Independent Test**: Mock VLM 返回 `vlm_judgment=true`；输入灰区 edge_bleed 结果；JSON 含 `vlm_reasoning` 与 `decision_trace` stage=`vlm_confirm`。

**Acceptance Scenarios**:

1. **Given** edge_bleed confidence=0.55，**When** Fast Mode Agent 流程，**Then** 触发 VLM Confirm，degradation 含 L3 字段
2. **Given** confidence=0.85，**When** 同上，**Then** 不调用 VLM
3. **Given** VLM 服务不可用，**When** 灰区 case，**Then** 降级硬阈值，trace 含 `vlm_skipped`

---

### User Story 2 - LLM Judge 整合 Round 1 并决定是否补检 (Priority: P1)

Round 1 全部子检测器完成后，小 LLM 审查 degradations + MOS + trace，判断 consistent / uncertain / inconsistent，并输出**结构化**白名单 actions（非直接改报告文本）。

**Why this priority**: Agent 自我决策能力；解决「MOS 低但无检出」等矛盾。

**Independent Test**: Mock Judge 返回 `dispatch_compression`；MOS=3.2 且 degradations=[]；Round 2 执行 compression 后报告更新。

**Acceptance Scenarios**:

1. **Given** 所有 confidence > 0.7 且 MOS 与检出一致，**When** Judge 审查，**Then** `needs_round2=false`，直接出报告
2. **Given** MOS 偏低且无 degradation，**When** Judge 审查，**Then** actions 含 `dispatch_compression`，Round 2 最多 1 轮
3. **Given** Judge 输出非法 action，**When** 解析，**Then** 忽略并 trace `judge_action_rejected`

---

### User Story 3 - decision_trace 完整可回溯 (Priority: P2)

复核人员需从 JSON 看到 Agent 每一环：路由、检测、VLM、Judge、Round 2。

**Independent Test**: 含灰区 + Round 2 的样本；trace 含 `routing`、`detection`、`vlm_confirm`、`aggregation` 及 Judge 决策摘要。

---

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: V1 Fast Mode MUST 使用 AgentOrchestrator 替代 v0.1 fixed pipeline
- **FR-002**: confidence ∈ [0.4, 0.7] 的 ROI MUST 触发 VLM Confirm（配额与降级见 design spec §8）
- **FR-003**: Round 1 结束后 MUST 调用 LLM Judge（Qwen2.5-1.5B 或配置等价物）
- **FR-004**: Judge 输出 MUST 为结构化 JSON（assessment, actions[], needs_round2）；禁止 open-ended 重跑全部检测器
- **FR-005**: Round 2 actions MUST 仅来自白名单（§13）：vlm_analyze, rerun_detector, dispatch_compression, accept
- **FR-006**: 最大迭代 **2 轮**（Round 1 + Round 2）；Round 2 后强制出报告
- **FR-007**: VLM / LLM 不可用 MUST 降级且不阻塞主链路（trace 记录 skip 原因）
- **FR-008**: Report MUST 填充 `vlm_reasoning`（灰区 case）与完整 `decision_trace`
- **FR-009**: LLM Judge MUST NOT 替代 ReportGenerator 的 MOS 聚合公式

### Out of Scope (002)

- Deep Mode 全量 VLM 粗分（P2，可后续 feature）
- LLM Judge 超过 2 轮迭代
- 新增子检测器算法（face/hair 等属其他 feature）
- V2 视频 / TemporalFlicker

## Success Criteria *(mandatory)*

- **SC-001**: 灰区样本 VLM 调用率可统计（10%–30% 量级，见 `config.example.yaml`）
- **SC-002**: Judge actions 100% 落在白名单内（fuzz 测试）
- **SC-003**: 「MOS 低无检出」矛盾 case 中 ≥80% 触发合理 action（benchmark 子集）
- **SC-004**: VLM/LLM mock 集成测试通过；`pytest -m vlm` 可选运行
- **SC-005**: decision_trace 含 Judge 条目且可人工读懂

## Assumptions

- Ollama 本地可跑 `qwen2.5-vl:7b` 与 `qwen2.5:1.5b`，或云端 OpenAI 兼容 API
- v0.1 的 QualityReport Schema 扩展 `vlm_reasoning` 字段，不破坏现有 golden sample
- CI 默认 `pytest -m "not vlm"`；VLM 测试 mock

## Design References

| 文档 | 用途 |
|------|------|
| [`VERSION_ROADMAP.md`](../VERSION_ROADMAP.md) | 版本边界 |
| [`plan.md`](plan.md) | Agent 编排实现计划 |
| [`contracts/llm-judge.schema.json`](contracts/llm-judge.schema.json) | Judge 输出 Schema |
| [`contracts/vlm-confirm.schema.json`](contracts/vlm-confirm.schema.json) | VLM Confirm Schema |
| [`contracts/quality-report.v1.schema.json`](contracts/quality-report.v1.schema.json) | 扩展报告 Schema |
