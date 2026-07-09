# Quickstart: V1 Agent Layer

**Feature**: 002-v1-agent-layer  
**Goal**: 在 001 基础上拉起 Ollama（或 mock）→ 验证 Agent Fast Mode 灰区 VLM + LLM Judge  
**Prerequisite**: [`001-v0-fast-mvp/quickstart.md`](../001-v0-fast-mvp/quickstart.md) 步骤 1–3 已通过

---

## Prerequisites

- 001 v0.1 代码已安装：`pip install -e ".[dev]"`
- **可选实机 VLM/Judge**：
  - [Ollama](https://ollama.com/) 已安装并运行
  - 模型已拉取：`ollama pull qwen2.5-vl:7b` 和 `ollama pull qwen2.5:1.5b`
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
  model: qwen2.5-vl:7b
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
export LQDD_VLM_MODEL=qwen2.5-vl:7b
export LQDD_JUDGE_MODEL=qwen2.5:1.5b
```

---

## 2. 验证 Ollama（可选）

```bash
curl -s "$OLLAMA_HOST/api/tags" | head
ollama run qwen2.5:1.5b "reply ok" 
```

不可达时 Agent 应**降级**出报告，trace 含 `vlm_skipped` / `judge_skipped`。

---

## 3. 运行 Agent Fast Mode

```bash
# V1 默认（实现后）
python detect.py --image data/sample/frames/edge/edge_01.png --mode fast

# v0.1 回归
python detect.py --image data/sample/frames/edge/edge_01.png --mode fast --legacy-fixed
```

**期望（Agent 开启 + Ollama 可用）**：

- JSON `system_version` ≥ `1.0.0`
- `decision_trace` 含 `routing`、`vlm_confirm`（灰区时）、`judge`
- 灰区 degradations 含 `vlm_reasoning.reasoning`（中文）
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
| US1 灰区 VLM | edge 样本 confidence ∈ [0.4,0.7] | `vlm_confirm` trace + `vlm_reasoning` |
| US1 高置信跳过 VLM | confidence > 0.7 | 无 `vlm_confirm` 或 `vlm_skipped` |
| US1 VLM 不可用 | `OLLAMA_HOST` 无效 | 降级 + `vlm_skipped` |
| US2 Judge dispatch | MOS 低无检出 mock | Round 2 `dispatch_compression` |
| US2 白名单拒绝 | fuzz 非法 action | trace `judge_action_rejected` |
| US3 完整 trace | 灰区 + Round 2 样本 | stages ≥ routing, detection, vlm_confirm, judge, aggregation |

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
