# Research: 002-v1-agent-layer

**Date**: 2026-07-09  
**Feature**: V1 Agent Layer（VLM 兜底 + LLM Judge）  
**Sources**: [`plan.md`](plan.md), [`contracts/`](contracts/), [`VERSION_ROADMAP.md`](../VERSION_ROADMAP.md), 001 research

---

## R1. VLM 服务接入（灰区 Confirm）

**Decision**: **Ollama 优先** + **OpenAI 兼容 API 回退**；统一 `VLMClient` 接口，模型默认 `qwen2.5-vl:7b`（`LQDD_VLM_MODEL` 可覆盖）。

**Rationale**:
- spec Assumptions 明确 Ollama 本地可跑 Qwen2.5-VL-7B
- Constitution：API Key 仅环境变量
- 灰区 Confirm 为 ROI 裁剪图 + 文本 prompt（§8.2），单帧 1–3 次调用，HTTP 足够
- `httpx` 同步客户端简单可靠；超时 `vlm_timeout_ms=2000` [CONFIG]

**Request shape (Ollama)**:
```http
POST /api/chat
{"model":"qwen2.5-vl:7b","messages":[{"role":"user","content":"...","images":["<base64>"]}],"format":"json"}
```

**Alternatives considered**:
| Alternative | Rejected because |
|-------------|------------------|
| 仅 OpenAI 官方 SDK | 绑定厂商；Ollama 本地开发是主路径 |
| 内嵌 transformers 推理 | 依赖重、显存高、与「Ollama 可跑」假设不一致 |
| 跳过 VLM、纯硬阈值 | 违反 V1 产品定义与 FR-002 |

**Degradation path**: `vlm_available=False` → 硬阈值 `hard_decision_threshold=0.55`；trace `vlm_skipped: {reason}`

---

## R2. LLM Judge 接入

**Decision**: **Qwen2.5-1.5B** via Ollama（`qwen2.5:1.5b`）或同等 API；**强制 JSON 输出** + jsonschema 校验 + 白名单过滤。

**Rationale**:
- Judge 输入为结构化摘要（degradations、MOS、skipped_detectors），1.5B 足够
- §13 要求 `assessment`, `actions[]`, `needs_round2` — 非 open-ended 文本
- 解析失败 → `assessment=uncertain`, `needs_round2=false`, trace `judge_parse_failed`

**Alternatives considered**:
| Alternative | Rejected because |
|-------------|------------------|
| 规则引擎替代 Judge | 无法覆盖「MOS 低无检出」等语义矛盾（US2） |
| 7B 作 Judge | 延迟高；设计已指定 1.5B |
| Judge 直接改 MOS | 违反 FR-009 与 report_generator 职责分离 |

---

## R3. Orchestrator 状态机

**Decision**: `AgentOrchestrator.run()` 实现 **2-round 有限状态机**：

```text
INIT → GLOBAL_SCAN → ROUTE → DETECT (parallel) → VLM_CONFIRM (grey ROI)
     → JUDGE_R1 → [ROUND2_ACTIONS] → AGGREGATE → REPORT
```

`AgentContext` 持有：`round_index`, `pending_vlm`, `judge_output`, `merged_degradations`, `routing_trace`.

**Rationale**:
- 对齐 002 feature contracts 与 max_rounds=2 约束
- Round 2 仅执行白名单 actions，由 `actions.py` 分发
- `fast_pipeline.py` 保留，`--legacy-fixed` 绕过 Agent（调试 / 回归 v0.1 golden）

**Alternatives considered**:
| Alternative | Rejected because |
|-------------|------------------|
| 扩展现有 fast_pipeline 内嵌 if | 职责混乱；违反「AgentOrchestrator 替代 fixed pipeline」 |
| 无界 while 循环 | 违反 FR-006 与 §13 终止条件 |

---

## R4. Router 置信度分级（Fast Mode）

**Decision**: 对每个 `RegionNomination` / 检测器输出应用三级带：

| Band | Condition | Action |
|------|-----------|--------|
| HIGH | confidence ≥ 0.7 | dispatch，无 VLM |
| GREY | 0.4 ≤ confidence < 0.7 | dispatch + `vlm_pending` |
| LOW | confidence < 0.4 | skip，记录 reason |

配额：`max_detectors_per_frame=5`, `max_vlm_calls_per_frame=3` [CONFIG]。

**Rationale**: 直接映射 FR-002 灰区 VLM 与 Router 三级带设计。

**002 scope**: 检测器仍仅 `edge_bleed`、`compression_artifact`；Router 映射表预留 `face→face_artifact` 等，未实现检测器返回 `detector_not_available`。

---

## R5. VLM ↔ Detector 融合

**Decision**: 按 §8.4 规则实现 `fuse.py`：

1. 一致 → `final_confidence = 0.4×det + 0.6×vlm`
2. 不一致 + vlm_conf > 0.8 → 以 VLM 为准
3. 不一致 + det_conf > 0.8 → 以检测器为准
4. 其他 → `uncertain`，MOS 扣分取较小 |mos_impact|

`vlm_reasoning` 写入 `DegradationItem.vlm_reasoning`（L3 字段）。

---

## R6. 测试与 CI 策略

**Decision**:
- **默认 CI**：`pytest -m "not vlm"` — `VLMClient` / `JudgeClient` 注入 `MockClient`（fixtures JSON）
- **可选**：`pytest -m vlm` — `@pytest.mark.vlm` 需 `OLLAMA_HOST` 可达
- **Fuzz**：随机非法 action 字符串 → 100% 被白名单拒绝（SC-002）

**Rationale**: Constitution V + spec SC-004；v0.1 CI 不受影响。

---

## R7. Schema 演进

**Decision**: 新增 `quality-report.v1.schema.json`（`system_version` ≥ 1.0.0）：

- `mode`: `fast` | `deep`（002 仅实现 fast Agent）
- `decision_trace.stage` 增加 `vlm_confirm`, `judge` 使用现有 enum 或扩展
- `DegradationItem.vlm_reasoning` 必填于灰区 case（对象，非 null）
- v0.1 `quality-report.schema.json` **不变**；v1 为超集

**Rationale**: spec Assumption「不破坏现有 golden sample」。

---

## R8. Deep Mode

**Decision**: **002 Out of Scope** — CLI `--mode deep` 返回 exit 2 或 stub「not implemented」；Orchestrator 预留 `mode=deep` 分支与 fallback 钩子，实现延后。

**Rationale**: 002 spec Out of Scope 明确 Deep Mode 可后续 feature；优先交付 Fast Agent + VLM + Judge。

---

## Resolved Clarifications

| Item | Resolution |
|------|------------|
| VLM 提供商 | Ollama 默认 + OpenAI 兼容 env |
| Judge 模型 | Qwen2.5-1.5B |
| 子检测器范围 | 002 仅 edge_bleed + compression；Router 可扩展 |
| Deep Mode | 延后 |
| v0.1 兼容 | `--legacy-fixed` + 独立 v1 schema |
