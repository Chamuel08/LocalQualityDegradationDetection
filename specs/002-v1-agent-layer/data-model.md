# Data Model: 002-v1-agent-layer

**Date**: 2026-07-09  
**Canonical source**: [`plan.md`](plan.md)、[`contracts/`](contracts/)  
**Extends**: [`001-v0-fast-mvp/data-model.md`](../001-v0-fast-mvp/data-model.md)  
**Scope**: Agent 编排层实体 — ReAct Agent 循环、VLM Confirm、向后兼容的 LLM Judge / Round 2

> **演进说明**：编排层已重构为 ReAct Agent。下方 `LLMJudge` / `Round2Executor` / `JudgeOutput` 实体保留描述作**向后兼容降级路径**（`run_judge` / `actions.py` 仍在仓库中，但主链路 `AgentOrchestrator.run` 不再经过独立 Judge 阶段）。主链路实体为 `AgentStep` / `AgentAction` / `AgentContext.agent_steps`。

---

## Entity Relationship

```text
SingleFrameInput
    │
    ▼
GlobalScanOutput
    │
    ▼
AgentOrchestrator ──► AgentContext（agent_steps, merged_degradations, vlm_results）
    │
    ├── Router ──► RoutingDecision[]（dispatch / skip，灰区也 dispatch）
    │
    ├── SubDetectors ──► DegradationItem[]（preliminary）
    │
    ├── ReAct Agent Loop（LLM decide 每步选工具）
    │     ├── vlm_analyze   ──► VLMConfirm ──► VLMResult[] ──► fuse ──► DegradationItem（+ vlm_reasoning）
    │     ├── rerun_detector──► SubDetector 重检
    │     ├── dispatch_compression ──► compression_artifact 检测器
    │     └── accept        ──► 终止
    │
    └── ReportGenerator ──► QualityReport（v1，含 agent_meta.agent_steps）
```

---

## Agent Layer

### AgentContext

跨步可变状态（内存，不持久化）。

| Field | Type | Notes |
|-------|------|-------|
| `round_index` | `int` | 保留字段（ReAct 模式下主要用 `agent_steps` 计数） |
| `max_rounds` | `int` | 默认 2；Agent 最大步数 = `max_rounds × 3` [CONFIG] |
| `mode` | `"fast"` \| `"deep"` | 当前仅 fast 可用 |
| `frame_input` | `SingleFrameInput` | ReAct 工具执行（VLM crop / 重检）需要 |
| `scan_output` | `GlobalScanOutput` | 重检 / 补检需要 |
| `routing_decisions` | `list[RoutingDecision]` | |
| `dispatched_detectors` | `list[str]` | |
| `preliminary_degradations` | `list[DegradationItem]` | CV 检测器输出 |
| `merged_degradations` | `list[DegradationItem]` | VLM 融合 / 补检后，Agent 持续更新 |
| `pending_vlm` | `list[VlmPendingItem]` | 旧灰区机制保留（主链路不再使用） |
| `vlm_results` | `list[VLMResult]` | |
| `judge_output` | `JudgeOutput \| None` | 仅向后兼容降级路径使用 |
| `round2_actions_executed` | `list[AgentAction]` | 仅向后兼容降级路径使用 |
| `vlm_calls_count` | `int` | 配额计数（`vlm.max_calls_per_frame`） |
| `vlm_ms` / `judge_ms` | `float` | 性能统计（`judge_ms` 在 ReAct 下累计 LLM decide 耗时） |
| `agent_steps` | `list[AgentStep]` | **ReAct 主链路**：每步 thought/action/observation |
| `max_rounds_reached` | `bool` | 达步数上限时置真 |

### RoutingDecision

| Field | Type | Validation |
|-------|------|------------|
| `source` | `"nomination"` | Fast Mode |
| `nomination` | `RegionNomination \| None` | |
| `decision` | `"dispatch"` \| `"skip"` | ReAct 下灰区也 `dispatch`；`vlm_pending` 已废弃 |
| `target_detector` | `str \| None` | 9 个检测器之一 |
| `reason` | `str` | 中文或英文技术摘要 |
| `confidence_band` | `"high"` \| `"grey"` \| `"low"` \| `None` | 仅供规则降级参考 | |

### VlmPendingItem

| Field | Type | Notes |
|-------|------|-------|
| `degradation_id` | `str` | 关联 preliminary |
| `detector` | `str` | |
| `bbox` | `BBox` | ROI 裁剪 |
| `preliminary_confidence` | `float` | |
| `region_type` | `str` | |

### VLMResult

| Field | Type | Required | Notes |
|-------|------|----------|-------|
| `region_type` | `str` | ✅ | |
| `degradation_type` | `str` | ✅ | |
| `is_degraded` | `bool` | ✅ | |
| `vlm_confidence` | `float` | ✅ | [0, 1] |
| `vlm_severity` | `str` | ✅ | Severity enum |
| `reasoning` | `str` | ✅ | L3 中文 |
| `mos_impact_estimate` | `float` | ✅ | ≤ 0 |
| `root_cause` | `str` | ✅ | RootCauseCategory |
| `ux_impact` | `str` | ✅ | 用户体验描述 |
| `vlm_latency_ms` | `float` | ✅ | |

### VLMReasoning（写入 DegradationItem）

| Field | Type | Notes |
|-------|------|-------|
| `reasoning` | `str` | L3 核心 |
| `vlm_confidence` | `float` | |
| `ux_impact` | `str` | |
| `fusion_decision` | `str` | e.g. `agree`, `vlm_override`, `detector_override`, `uncertain` |

### JudgeOutput（向后兼容降级路径）

> ReAct 主链路**不使用** `JudgeOutput`。仅 `run_judge` / `RuleBasedJudgeClient.review` 等降级路径产出。主链路的 LLM 决策由 `decide()` 返回的 `{thought, action, ...}` 直接解析为 `AgentStep`。

| Field | Type | Required | Validation |
|-------|------|----------|------------|
| `assessment` | `str` | ✅ | `consistent` \| `uncertain` \| `inconsistent` |
| `reasoning` | `str` | ✅ | Judge 中文摘要 |
| `actions` | `list[AgentAction]` | ✅ | 白名单过滤后 |
| `needs_round2` | `bool` | ✅ | |
| `judge_latency_ms` | `float` | ❌ | 性能 |
| `raw_rejected_actions` | `list[str]` | ❌ | 非法 action 记录 |

### AgentStep（ReAct 主链路）

每一步 Thought–Action–Observation 的完整记录，写入 `AgentContext.agent_steps` 与 `agent_meta.agent_steps`。

| Field | Type | Notes |
|-------|------|-------|
| `step_index` | `int` | 从 1 起 |
| `thought` | `str` | LLM 推理过程（中文） |
| `action` | `AgentAction` | 解析后的动作 |
| `observation` | `str` | 工具执行结果描述（执行后回填） |
| `latency_ms` | `float` | 该步 LLM decide 耗时 |

### AgentAction（白名单）

| action | Parameters | Constraints |
|--------|------------|-------------|
| `vlm_analyze` | `degradation_id?`, `reason?` | 对指定项调用 VLM；未指定则对全部未确认项 |
| `rerun_detector` | `detector`, `nomination_threshold_delta` | delta ∈ [-0.15, -0.05] |
| `dispatch_compression` | `reason?` | 无参 |
| `accept` | `reason?` | 终止循环 |

```python
class AgentAction(TypedDict, total=False):
    action: Literal["vlm_analyze", "rerun_detector", "dispatch_compression", "accept"]
    degradation_id: str
    detector: str
    nomination_threshold_delta: float
    reason: str
```

### JudgeInput（序列化摘要，向后兼容降级路径用）

> ReAct 主链路使用 `build_agent_observe_prompt()` 生成 system+user prompt（`AGENT_SYSTEM_PROMPT` + `AGENT_OBSERVE_TEMPLATE`），含 `global_mos`、`detection_count`、`detections_json`、`skipped_detectors`、`history`。下方 `JudgeInput` 仅旧 `build_judge_prompt` 使用。

| Field | Type | Notes |
|-------|------|-------|
| `global_mos` | `float` | Round 1 预聚合或 base_mos 估计 |
| `detections` | `list[dict]` | detector, confidence, degradation_type, severity |
| `skipped_detectors` | `list[str]` | |
| `vlm_skipped_count` | `int` | |
| `mode` | `str` | |

---

## TraceEntry 扩展（V1）

在 001 `TraceEntry` 基础上新增 stage：

| stage | module 示例 | 何时写入 |
|-------|-------------|----------|
| `routing` | `FastRouter` | 每个 nomination 决策后 |
| `vlm_confirm` | `VLMConfirm` | 每次 VLM 调用 / 失败 / 配额跳过 |
| `agent_step` | `ReactAgent` | ReAct 每一步（`agent_vlm_analyze_stepN` / `agent_accept_stepN` 等） |
| `judge` | `LLMJudge` | 仅向后兼容降级路径 |

**Schema note**: v1 schema 的 `TraceEntry.stage` enum 已包含 `agent_step`（与 `vlm_confirm`、`judge` 并列）。`decision` 字段承载 `agent_<action>_stepN`。

---

## Config Model 扩展

```yaml
agent:
  enabled: true                    # false → fallback fast_pipeline
  legacy_fixed: false              # CLI --legacy-fixed 覆盖
  max_rounds: 2
  high_confidence_threshold: 0.7
  grey_zone_lower: 0.4
  grey_zone_upper: 0.7
  max_detectors_per_frame: 5
  hard_decision_threshold: 0.55    # VLM 不可用

vlm:
  provider: ollama                 # ollama | openai
  model: qwen2.5vl:7b
  host: http://localhost:11434
  timeout_ms: 2000
  max_calls_per_frame: 3

judge:
  provider: ollama
  model: qwen2.5:1.5b
  host: http://localhost:11434
  timeout_ms: 1500
```

环境变量覆盖：`OLLAMA_HOST`, `LQDD_VLM_MODEL`, `LQDD_JUDGE_MODEL`, `OPENAI_API_KEY`, `OPENAI_API_BASE`.

---

## QualityReport v1 Delta

| Field | v0.1 | v1 |
|-------|------|-----|
| `system_version` | `0.1.0` | `1.0.0` |
| `mode` | `fast` | `fast` \| `deep` |
| `vlm_reasoning_summary` | `null` | Agent 触发 VLM 的 case 为 `list[VLMReasoningSummary]` |
| `DegradationItem.vlm_reasoning` | 可选 null | Agent 触发 VLM 的项为对象 |
| `agent_meta.judge_assessment` | — | ReAct 模式下为 `null`（无独立 Judge 阶段） |
| `agent_meta.agent_driven_vlm` | — | `bool`，VLM 是否由 Agent 自主触发 |
| `agent_meta.agent_steps` | — | `list[{step, thought, action, reason, observation, latency_ms}]` |
| `decision_trace` | 5 stages | + `agent_step × N`（`vlm_confirm` 在 VLM 调用时出现） |
| `performance` | 4 字段 | + `vlm_ms`, `judge_ms` |

**Backward compatibility**: v0.1 golden 无 `vlm_reasoning` / `agent_meta` — 仍通过 v0.1 schema。Agent 关闭时输出等同 v0.1。

---

## State Transitions

```text
SCAN → ROUTE → DETECT
                     │
                     ▼
              ┌── ReAct Agent Loop ──┐
              │  Observe(CV + history)│
              │  Think + Act (LLM)    │
              │  Execute tool         │
              │   ├ vlm_analyze       │
              │   ├ rerun_detector    │
              │   ├ dispatch_compression
              │   └ accept → break    │
              └──────────┬────────────┘
                         ▼
                     REPORT
```

**Invariants**:
- Agent 步数不得超过 `max_rounds × 3`
- 同一 action 连续出现 ≥ 2 次时强制 `accept`（防死循环）
- LLM `decide()` 返回 `None` / 不可解析时强制 `accept`
- `vlm_analyze` 受 `vlm.max_calls_per_frame` 配额限制
- MOS 仅由 `ReportGenerator.compute_mos()` 计算（支持 `rule` / `clip_iqa` / `internal`）
