# Quickstart: V1 Agent Layer

**Feature**: 002-v1-agent-layer  
**Goal**: 在 001 基础上拉起 Ollama（或 mock）→ 验证 Agent Fast Mode **ReAct Agent 自主决策（VLM 视觉确认 + 补检）**  
**Prerequisite**: [`001-v0-fast-mvp/quickstart.md`](../001-v0-fast-mvp/quickstart.md) 步骤 1–3 已通过

---

## Prerequisites

- 001 v0.1 代码已安装：`pip install -e ".[dev]"`
- **可选实机 VLM/Judge**：
  - [Ollama](https://ollama.com/) 已安装并运行
  - 模型已拉取：`ollama pull qwen2.5vl:7b` 和 `ollama pull qwen2.5:1.5b`
- **CI / 无 GPU**：使用 mock（默认 `pytest -m "not vlm"`）

---

## 1. 配置 Agent

```bash
cp config.example.yaml config.yaml
```

在 `config.yaml` 追加（实现阶段由 002 tasks 写入 `config.example.yaml`）：

```yaml
agent:
  enabled: true
  max_rounds: 2
  high_confidence_threshold: 0.7
  grey_zone_lower: 0.4
  grey_zone_upper: 0.7

vlm:
  provider: ollama
  model: qwen2.5vl:7b
  host: http://localhost:11434
  timeout_ms: 2000

judge:
  provider: ollama
  model: qwen2.5:1.5b
  timeout_ms: 1500
```

环境变量（覆盖 config）：

```bash
export OLLAMA_HOST=http://localhost:11434
export LQDD_VLM_MODEL=qwen2.5vl:7b
export LQDD_JUDGE_MODEL=qwen2.5:1.5b
```

---

## 2. 验证 Ollama（可选）

```bash
curl -s "$OLLAMA_HOST/api/tags" | head
ollama run qwen2.5:1.5b "reply ok" 
```

不可达时 Agent 应**降级**到 `RuleBasedJudgeClient` 规则决策出报告，trace 含 `vlm_failed: service_unavailable`。

---

## 3. 运行 Agent Fast Mode

```bash
# V1 默认（ReAct Agent）
python detect.py --image data/sample/frames/edge/edge_01.png --mode fast

# v0.1 回归
python detect.py --image data/sample/frames/edge/edge_01.png --mode fast --legacy-fixed
```

**期望（Agent 开启 + Ollama 可用）**：

- JSON `system_version` ≥ `1.0.0`
- `decision_trace` 含 `routing`、`detection`、`agent_step × N`（若 Agent 触发 VLM 则含 `vlm_confirm`）
- `agent_meta.agent_steps` 含每步 `thought` / `action` / `observation`
- 若 Agent 自主触发 VLM：`agent_meta.agent_driven_vlm=true`，对应 degradation 含 `vlm_reasoning.reasoning`（中文）
- 通过 [`contracts/quality-report.v1.schema.json`](./contracts/quality-report.v1.schema.json)

---

## 4. Mock 集成测试（无 Ollama）

```bash
LQDD_AGENT_ENABLED=false pytest tests/ -m "not vlm" -q
```

实现后新增：

```bash
pytest tests/integration/test_agent_pipeline_mock.py -q
pytest tests/contract/test_judge_schema.py tests/contract/test_vlm_schema.py -q
```

---

## 5. 实机 VLM 测试（可选）

```bash
pytest tests/ -m vlm -q
```

需 Ollama 运行且模型已拉取；标记为 `@pytest.mark.vlm`。

---

## 6. 验收场景对照

| 场景 | 命令 / 条件 | 期望 |
|------|-------------|------|
| US1 Agent 触发 VLM | 含低置信度项的样本，LLM 选 `vlm_analyze` | `agent_step(agent_vlm_analyze_stepN)` trace + `vlm_reasoning`，`agent_driven_vlm=true` |
| US1 高置信跳过 VLM | 全高置信，LLM 直接 `accept` | `agent_driven_vlm=false`，无 `vlm_confirm` |
| US1 VLM 不可用 | `OLLAMA_HOST` 无效 | 降级 + `vlm_failed: service_unavailable` |
| US2 Agent 补检 | MOS 低无检出，LLM 选 `dispatch_compression` | `agent_step(agent_dispatch_compression_stepN)`，merged 含 `compression_artifact` |
| US2 非法 action | fuzz 非法/不可解析 | 强制 `accept`，trace 记录，不崩溃 |
| US3 完整 trace | 含 `vlm_analyze` 的样本 | stages ≥ routing, detection, agent_step×N, aggregation |

---

## 7. 相关契约

| 文档 | 用途 |
|------|------|
| [cli-contract.md](./contracts/cli-contract.md) | CLI `--legacy-fixed`、环境变量 |
| [llm-judge.schema.json](./contracts/llm-judge.schema.json) | Judge 输出 |
| [vlm-confirm.schema.json](./contracts/vlm-confirm.schema.json) | VLM Confirm 输出 |
| [quality-report.v1.schema.json](./contracts/quality-report.v1.schema.json) | 报告 v1 |

---

## 8. 下一步

执行 `/speckit-tasks` 生成 `tasks.md`，再 `/speckit-implement` 实现 Agent 层。
