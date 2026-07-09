# CLI Contract Delta: detect.py (V1)

**Version**: 1.0.0  
**Feature**: 002-v1-agent-layer  
**Base**: [`001-v0-fast-mvp/contracts/cli-contract.md`](../001-v0-fast-mvp/contracts/cli-contract.md)

---

## Synopsis (V1)

```text
detect.py [--image PATH | --image-dir DIR] --mode {fast,deep} [OPTIONS]
```

**V1 变更**：`--mode fast` 默认走 **AgentOrchestrator**（非 v0.1 fixed pipeline），含 VLM 灰区 + LLM Judge。

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
| `LQDD_VLM_MODEL` | `qwen2.5-vl:7b` | VLM Confirm |
| `LQDD_JUDGE_MODEL` | `qwen2.5:1.5b` | LLM Judge |
| `OPENAI_API_BASE` | — | OpenAI 兼容 endpoint |
| `OPENAI_API_KEY` | — | API 密钥 |
| `LQDD_AGENT_ENABLED` | `true` | `false` 等同 `--legacy-fixed` |

VLM/Judge 不可用时：**不失败**，降级出报告，`decision_trace` 含 `vlm_skipped` / `judge_skipped`。

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
- 灰区 case MUST 含 `degradations[].vlm_reasoning` 与 `decision_trace` stage `vlm_confirm`
- Judge 完成 MUST 含 trace entry `stage=judge` 或 `decision` 前缀 `judge_`

---

## New Error Messages

| Condition | Message pattern |
|-----------|-----------------|
| Deep not implemented | `error: deep mode not implemented in 002; use --mode fast` |
| Judge parse failure | trace only: `judge_parse_failed`（不 exit 非 0） |

---

## Non-goals (002)

- `--mode deep` 完整实现
- Agent HTTP 服务端
- 递归 `--image-dir`
