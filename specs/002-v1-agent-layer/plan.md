# Implementation Plan: V1 Agent Layer（VLM 兜底 + LLM Judge）

**Branch**: `002-v1-agent-layer` | **Date**: 2026-07-09 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `specs/002-v1-agent-layer/spec.md`

**Depends on**: [`001-v0-fast-mvp`](../001-v0-fast-mvp/spec.md) — v0.1 `lqdd` package、QualityReport Schema、GlobalScan + 2 检测器必须先可用

## Summary

在 v0.1 固定 pipeline 之上引入 **AgentOrchestrator**，使 `--mode fast` 走 **Agent 编排**（非 `fast_pipeline.py` 直连）：

```text
GlobalScan → Router（置信度分级）→ SubDetectors（edge_bleed, compression_artifact）
    → VLM Confirm（灰区 ROI）→ LLM Judge（Round 1 审查）
    → Round 2 白名单 actions（可选）→ ReportGenerator
```

**VLM**：Qwen2.5-VL-7B（Ollama `qwen2.5-vl:7b` 或 OpenAI 兼容 API）对 confidence ∈ [0.4, 0.7] 的 ROI 做二次确认，输出 L3 `vlm_reasoning`。

**LLM Judge**：Qwen2.5-1.5B 审查全帧 degradations + MOS + trace，输出结构化 `assessment` / `actions[]` / `needs_round2`；**不**改写 MOS 公式。

**范围**：本 feature 仅编排层 + VLM/Judge 集成；**不**新增 face/hair 等子检测器算法（属后续 feature）。Round 2 白名单动作在现有 2 检测器上验证。

技术依据：本 feature 的 `plan.md`、`research.md`、`contracts/`；[`VERSION_ROADMAP.md`](../VERSION_ROADMAP.md)。

## Technical Context

**Language/Version**: Python 3.10+（MediaPipe 推荐 3.10–3.12；3.14 沿用 v0.1 GrabCut 回退）

**Primary Dependencies**（在 v0.1 基础上新增）:
- `httpx` 或 `requests` — Ollama / OpenAI 兼容 HTTP 客户端
- `pydantic`（可选）— Judge/VLM JSON 响应校验
- 保留：OpenCV, NumPy, PyYAML, jsonschema, pytest

**External Services**（V1 必需，可 mock）:
- **VLM**：`qwen2.5-vl:7b` @ Ollama（默认 `http://localhost:11434`）或 `OPENAI_API_BASE` + `OPENAI_API_KEY`
- **LLM Judge**：`qwen2.5:1.5b` @ Ollama 或同等小模型 API
- 密钥仅环境变量：`OLLAMA_HOST`, `OPENAI_API_KEY`, `LQDD_VLM_MODEL`, `LQDD_JUDGE_MODEL`

**Storage**: 文件型 — 输入帧、sidecar JSON、输出 JSON/HTML；新增 `config.agent` / `config.vlm` / `config.judge` 配置块

**Testing**:
- 默认 CI：`pytest -m "not vlm"`（mock VLM/Judge）
- 可选本地：`pytest -m vlm`（需 Ollama 运行）
- Contract：Judge 输出 Schema、VLM Confirm Schema、扩展 QualityReport v1 Schema
- Fuzz：Judge actions 白名单校验

**Target Platform**: macOS / Linux CLI；CPU 可跑检测器；VLM/Judge 建议 GPU 或本地 Ollama

**Project Type**: CLI + Python library（`src/lqdd/` 扩展）

**Performance Goals**:
- 检测器段：继承 v0.1（P50 < 500ms / 720p CPU）
- VLM Confirm：单次 < 2000ms（`vlm_timeout_ms` [CONFIG]）
- Judge：单次 < 1500ms（`judge_timeout_ms` [CONFIG]）
- 灰区 VLM 调用率：10%–30%（SC-001，可统计）

**Constraints**:
- 最大 **2 轮**（Round 1 + Round 2）；Round 2 后强制出报告
- Judge actions **仅白名单**：`vlm_analyze`, `rerun_detector`, `dispatch_compression`, `accept`
- VLM/Judge 不可用 → 降级 + `decision_trace` 记录 skip（不阻塞主链路）
- Judge **不得**替代 ReportGenerator MOS 聚合（FR-009）
- Evidence.detail 保持中文（Constitution III）
- v0.1 golden JSON **仍须**通过 v0.1 schema（无 Agent 时）或 v1 schema（Agent 开启时向后兼容）

**Scale/Scope**:
- 单帧 + 批量目录（继承 001 CLI）
- 子检测器：002 阶段仍 **edge_bleed + compression_artifact**（Router 预留 6 类映射）
- Deep Mode：**本 feature Out of Scope**（仅 stub + fallback 设计，实现可延后到 002b 或 003）
- Sample：灰区帧 + mock 集成测试夹具

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | V1 Compliance | Notes |
|-----------|-----------------|-------|
| I. Design-First | ✅ PASS | 对齐 VERSION_ROADMAP、002 contracts |
| II. CLI-First | ✅ PASS | `detect.py --mode fast` 默认 Agent；JSON/HTML 不变 |
| III. Explainability | ✅ PASS | Evidence 四要素 + decision_trace 含 routing/vlm_confirm/judge |
| IV. Coarse-to-Fine & Scope | ✅ PASS | Agent 层为 V1 产品必需；002 不扩 face/hair 检测器 |
| V. Testability | ✅ PASS | mock VLM/Judge + `pytest -m vlm` 可选 + Schema 契约 |
| VI. Badcase Frame Scene | ✅ PASS | 离线单帧；灰区 badcase 为核心场景 |

**Post-design re-check**: ✅ PASS。引入外部 VLM/LLM 为 Constitution 明示的 V1 约束（§Technology Constraints），非 v0.1 豁免范围的违规。`fast_pipeline.py` 保留作 `--legacy-fixed` 调试回退，非默认路径。

## Project Structure

### Documentation (this feature)

```text
specs/002-v1-agent-layer/
├── plan.md              # This file
├── research.md          # Phase 0 — VLM/Judge client & orchestration decisions
├── data-model.md        # Phase 1 — Agent entities & state machine
├── quickstart.md        # Phase 1 — Ollama setup + agent validation
├── contracts/
│   ├── cli-contract.md              # CLI delta（mode fast = Agent）
│   ├── quality-report.v1.schema.json
│   ├── llm-judge.schema.json
│   └── vlm-confirm.schema.json
└── tasks.md             # Phase 2 (/speckit-tasks — not created here)
```

### Source Code (repository root — V1 additions)

```text
config.example.yaml            # + agent, vlm, judge sections

src/lqdd/
├── agent/
│   ├── __init__.py
│   ├── orchestrator.py        # AgentOrchestrator: Round 1/2 loop
│   ├── router.py              # Confidence bands → dispatch/skip/vlm_pending
│   ├── context.py             # AgentContext（round, pending_vlm, judge_result）
│   ├── judge.py               # LLM Judge client + JSON parse + whitelist filter
│   ├── actions.py             # Round 2 executor（dispatch_compression, rerun_detector, …）
│   └── prompts.py             # VLM Confirm + Judge prompt templates
├── vlm/
│   ├── __init__.py
│   ├── client.py              # Ollama / OpenAI-compatible HTTP
│   ├── confirm.py             # Gray-zone ROI confirm + crop
│   └── fuse.py                # detector × VLM fusion rules (§8.4)
├── pipeline/
│   ├── fast_pipeline.py       # v0.1 fixed（--legacy-fixed 回退）
│   └── agent_pipeline.py      # V1 默认：Orchestrator → Report
├── cli/
│   └── main.py                # --mode fast → agent_pipeline；env 检测
├── config/
│   └── loader.py              # + AgentConfig, VLMConfig, JudgeConfig
└── models/
    └── agent.py               # VLMResult, JudgeOutput, AgentAction, RoutingDecision

tests/
├── unit/
│   ├── test_router.py
│   ├── test_judge_parser.py
│   ├── test_vlm_fuse.py
│   └── test_actions_whitelist.py
├── integration/
│   └── test_agent_pipeline_mock.py
├── contract/
│   ├── test_judge_schema.py
│   ├── test_vlm_schema.py
│   └── test_report_v1_schema.py
└── fixtures/
    ├── mock_judge_responses/
    └── mock_vlm_responses/
```

**Structure Decision**: 单项目扩展 — 在 `src/lqdd/` 新增 `agent/`、`vlm/` 包；`pipeline/agent_pipeline.py` 为 V1 默认入口；检测器与 Report 复用 001 实现。

## Complexity Tracking

> 无 Constitution 违规需豁免。下表记录相对 v0.1 的复杂度增量理由。

| Addition | Why Needed | Simpler Alternative Rejected Because |
|----------|------------|-------------------------------------|
| `agent/` + `vlm/` 包 | V1 产品必需 Agent 层（Constitution IV） | 继续 fixed pipeline 违反 VERSION_ROADMAP V1 定义 |
| HTTP 客户端 + mock 层 | Ollama/API 集成与 CI 隔离 | 硬编码仅 Ollama 无法覆盖云端 API 与无网 CI |
| Round 2 状态机 | LLM Judge 自我决策（FR-004–006） | 单次 pipeline 无法处理 MOS/检出矛盾 |

## Phase 0 Output

See [research.md](./research.md) — VLM/Judge 客户端、降级策略、mock 方案、Router 置信度分级均已决议。

## Phase 1 Output

See:
- [data-model.md](./data-model.md) — AgentContext, VLMResult, JudgeOutput, RoutingDecision
- [contracts/](./contracts/) — CLI delta、JSON Schemas
- [quickstart.md](./quickstart.md) — Ollama 拉起 → 灰区样本 → mock/实机验证

## Implementation Phases (for /speckit-tasks)

| Phase | Focus | Key deliverables |
|-------|-------|------------------|
| P1 Setup | Config + models + HTTP client skeleton | `config.agent`, `models/agent.py`, `vlm/client.py` mock |
| P2 Router | Fast mode routing + trace | `agent/router.py`, nomination → detector map |
| P3 VLM Confirm | Gray zone ROI + fusion | `vlm/confirm.py`, `vlm/fuse.py`, `vlm_reasoning` on DegradationItem |
| P4 LLM Judge | Round 1 review + action parse | `agent/judge.py`, whitelist filter |
| P5 Round 2 | Action executor + 2-round cap | `agent/actions.py`, `agent/orchestrator.py` |
| P6 CLI | Wire `--mode fast` → agent | `agent_pipeline.py`, `--legacy-fixed` flag |
| P7 Polish | Tests, quickstart, README | `pytest -m vlm`, golden 灰区样本 |

**MVP checkpoint（002）**: mock 集成测试通过 — 灰区 edge 触发 VLM trace + Judge dispatch_compression 场景。

## Deferred (Out of 002 Scope)

- Deep Mode VLM 粗分（`--mode deep` 完整实现）
- face_artifact / hair_texture / hand_anomaly 检测器
- TemporalFlicker（V2）
- 视频 clip 输入
