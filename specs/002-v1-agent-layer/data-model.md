# Data Model: 002-v1-agent-layer

**Date**: 2026-07-09  
**Canonical source**: [`plan.md`](plan.md)、[`contracts/`](contracts/)  
**Extends**: [`001-v0-fast-mvp/data-model.md`](../001-v0-fast-mvp/data-model.md)  
**Scope**: Agent 编排层实体 — VLM Confirm、LLM Judge、Round 2 状态

---

## Entity Relationship

```text
SingleFrameInput
    │
    ▼
GlobalScanOutput
    │
    ▼
AgentOrchestrator ──► AgentContext（round, trace, pending_vlm）
    │
    ├── Router ──► RoutingDecision[]（high/grey/low）
    │
    ├── SubDetectors ──► DegradationItem[]（preliminary）
    │
    ├── VLMConfirm ──► VLMResult[] ──► fuse ──► DegradationItem（+ vlm_reasoning）
    │
    ├── LLMJudge ──► JudgeOutput（assessment, actions[], needs_round2）
    │
    ├── Round2Executor ──► ActionResult[]
    │
    └── ReportGenerator ──► QualityReport（v1）
```

---

## Agent Layer

### AgentContext

跨 Round 可变状态（内存，不持久化）。

| Field | Type | Notes |
|-------|------|-------|
| `round_index` | `int` | 1 或 2 |
| `max_rounds` | `int` | 固定 2 [CONFIG] |
| `mode` | `"fast"` \| `"deep"` | 002 仅 fast |
| `frame_input` | `SingleFrameInput` | |
| `scan_output` | `GlobalScanOutput` | |
| `routing_decisions` | `list[RoutingDecision]` | |
| `preliminary_degradations` | `list[DegradationItem]` | Round 1 检测器输出 |
| `merged_degradations` | `list[DegradationItem]` | VLM 融合后 |
| `pending_vlm` | `list[VlmPendingItem]` | 灰区待确认 |
| `vlm_results` | `list[VLMResult]` | |
| `judge_output` | `JudgeOutput \| None` | Round 1 结束 |
| `round2_actions_executed` | `list[AgentAction]` | |
| `vlm_calls_count` | `int` | 配额计数 |
| `max_rounds_reached` | `bool` | Round 2 后强制 true |

### RoutingDecision

| Field | Type | Validation |
|-------|------|------------|
| `source` | `"nomination"` | Fast Mode |
| `nomination` | `RegionNomination \| None` | |
| `decision` | `"dispatch"` \| `"skip"` \| `"vlm_pending"` | |
| `target_detector` | `str \| None` | `edge_bleed` \| `compression_artifact` |
| `reason` | `str` | 中文或英文技术摘要 |
| `confidence_band` | `"high"` \| `"grey"` \| `"low"` \| `None` | |

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

### JudgeOutput

| Field | Type | Required | Validation |
|-------|------|----------|------------|
| `assessment` | `str` | ✅ | `consistent` \| `uncertain` \| `inconsistent` |
| `reasoning` | `str` | ✅ | Judge 中文摘要 |
| `actions` | `list[AgentAction]` | ✅ | 白名单过滤后 |
| `needs_round2` | `bool` | ✅ | |
| `judge_latency_ms` | `float` | ❌ | 性能 |
| `raw_rejected_actions` | `list[str]` | ❌ | 非法 action 记录 |

### AgentAction（白名单）

| action | Parameters | Constraints |
|--------|------------|-------------|
| `vlm_analyze` | `target_region`, `detector`, `reason?` | 仅 §8 未触发时 |
| `rerun_detector` | `detector`, `nomination_threshold_delta` | delta ∈ [-0.15, -0.05] |
| `dispatch_compression` | `reason?` | 无参 |
| `accept` | `target_region`, `reason?` | 确认不补检 |

```python
class AgentAction(TypedDict, total=False):
    action: Literal["vlm_analyze", "rerun_detector", "dispatch_compression", "accept"]
    target_region: str
    detector: str
    nomination_threshold_delta: float
    reason: str
```

### JudgeInput（序列化摘要）

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
| `vlm_confirm` | `VLMConfirm` | 每次灰区 VLM 调用或 skip |
| `judge` | *新增 enum 值* 或复用 `aggregation` + `decision=judge_*` | Round 1 Judge 完成 |

**Schema note**: v1 schema 在 `TraceEntry.stage` enum 增加 `judge`（或通过 `decision` 字段承载 `judge_consistent` 等，实现时二选一并在 contract 固定）。

推荐：**扩展 enum 增加 `judge`**，与 `vlm_confirm` 并列。

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
  model: qwen2.5-vl:7b
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
| `vlm_reasoning_summary` | `null` | 灰区 case 为 `list[VLMReasoningSummary]` |
| `DegradationItem.vlm_reasoning` | 可选 null | 灰区必填对象 |
| `decision_trace` | 5 stages | + `vlm_confirm`, `judge` |
| `performance` | 4 字段 | + `vlm_ms`, `judge_ms`（可选扩展） |

**Backward compatibility**: v0.1 golden 无 `vlm_reasoning`、trace 无 judge — 仍通过 v0.1 schema。Agent 关闭时输出等同 v0.1。

---

## State Transitions

```text
[Round 1]
  SCAN → ROUTE → DETECT → VLM_CONFIRM? → JUDGE
                                              │
                    needs_round2=false ───────┼──► REPORT
                    needs_round2=true ────────┘
                              │
[Round 2]                     ▼
  EXECUTE_ACTIONS → MERGE → REPORT (max_rounds_reached=true)
```

**Invariants**:
- `round_index` 不得超过 `max_rounds`
- `judge_output.actions` 经白名单过滤后才执行
- MOS 仅由 `ReportGenerator.compute_mos()` 计算
