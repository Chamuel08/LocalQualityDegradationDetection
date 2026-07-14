# CLI Contract Delta: detect.py (V1)

**Version**: 1.0.0  
**Feature**: 002-v1-agent-layer  
**Base**: [`001-v0-fast-mvp/contracts/cli-contract.md`](../001-v0-fast-mvp/contracts/cli-contract.md)

---

## Synopsis (V1)

```text
detect.py [--image PATH | --image-dir DIR] --mode {fast,deep} [OPTIONS]
```

**V1 变更**：`--mode fast` 默认走 **AgentOrchestrator（ReAct Agent）**（非 v0.1 fixed pipeline）；LLM 自主决策是否调用 VLM、是否补检。

---

## Mode

| Argument | Default | Allowed (V1) | Behavior |
|----------|---------|--------------|----------|
| `--mode` | `fast` | `fast`, `deep` | `fast` → Agent pipeline；`deep` → **002 未实现**，exit 2 |

**v0.1 回退**：

| Flag | Description |
|------|-------------|
| `--legacy-fixed` | 强制使用 v0.1 `fast_pipeline.py`（无 VLM/Judge）；用于 golden 回归 |

---

## Environment (V1)

| Variable | Default | Purpose |
|----------|---------|---------|
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama API |
| `LQDD_VLM_MODEL` | `qwen2.5vl:7b` | VLM Confirm |
| `LQDD_JUDGE_MODEL` | `qwen2.5:1.5b` | Agent LLM（ReAct 决策） |
| `OPENAI_API_BASE` | — | OpenAI 兼容 endpoint |
| `OPENAI_API_KEY` | — | API 密钥 |
| `LQDD_AGENT_ENABLED` | `true` | `false` 等同 `--legacy-fixed` |

VLM/LLM 不可用时：**不失败**，Agent 降级到 `RuleBasedJudgeClient` 规则决策出报告，`decision_trace` 含 `vlm_failed: service_unavailable`。

---

## Examples (V1)

### Fast Mode with Agent（默认）

```bash
python detect.py --image data/sample/frames/edge/grey_edge_01.png --mode fast
```

### Legacy v0.1 fixed pipeline

```bash
python detect.py --image data/sample/frames/edge/edge_01.png --mode fast --legacy-fixed
```

### Mock CI（无 Ollama）

```bash
LQDD_AGENT_ENABLED=false pytest tests/ -m "not vlm" -q
```

---

## Output Contract (V1)

- Agent 开启：JSON SHOULD validate against [`quality-report.v1.schema.json`](./quality-report.v1.schema.json)
- Legacy fixed：JSON MUST validate against v0.1 schema
- Agent 自主触发 VLM 的 case SHOULD 含 `degradations[].vlm_reasoning` 与 `decision_trace` stage `vlm_confirm`
- ReAct 每一步 MUST 含 `decision_trace` stage `agent_step`（`decision=agent_<action>_stepN`），且 `agent_meta.agent_steps` 完整
- `agent_meta.judge_assessment` 在 ReAct 模式下为 `null`（无独立 Judge 阶段）

---

## New Error Messages

| Condition | Message pattern |
|-----------|-----------------|
| Deep not implemented | `error: deep mode not implemented in 002; use --mode fast` |
| Agent action parse failure | trace only: 强制 `accept` 终止（不 exit 非 0） |
| VLM service unavailable | trace only: `vlm_failed: service_unavailable`（不 exit 非 0） |

---

## Non-goals (002)

- `--mode deep` 完整实现
- Agent HTTP 服务端
- 递归 `--image-dir`
