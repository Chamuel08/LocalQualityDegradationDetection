# Implementation Plan: V1 Agent Layer（VLM 兜底 + LLM Judge）

**Branch**: `002-v1-agent-layer` | **Date**: 2026-07-09 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `specs/002-v1-agent-layer/spec.md`

**Depends on**: [`001-v0-fast-mvp`](../001-v0-fast-mvp/spec.md) — v0.1 `lqdd` package、QualityReport Schema、GlobalScan + 子检测器必须先可用（001 原始切片 2 检测器，后续扩展至 9 类）

## Summary

在 v0.1 固定 pipeline 之上引入 **AgentOrchestrator（ReAct Agent）**，使 `--mode fast` 走 **Agent 编排**（非 `fast_pipeline.py` 直连）：

```text
GlobalScan → Router（置信度分级，灰区也 dispatch）→ 9 SubDetectors
    → ReAct Agent Loop（LLM 每步自主决策）
        ├ vlm_analyze   → VLM Confirm（ROI）→ fuse
        ├ rerun_detector → 重检某检测器
        ├ dispatch_compression → 补检压缩伪影
        └ accept         → 终止
    → ReportGenerator
```

**ReAct Agent**：Qwen2.5-1.5B（Ollama `qwen2.5:1.5b` 或 OpenAI 兼容 API）每步观察 CV 全量结果 + 历史步骤，输出严格 JSON `{thought, action, ...}`，从白名单工具中选择一个执行；执行后将 observation 反馈进入下一步，直到 `accept` 或达 `max_rounds × 3` 步。

**VLM**：Qwen2.5-VL-7B 对 Agent 指定的 ROI 做视觉确认，输出 L3 `vlm_reasoning`。VLM 是否调用由 LLM 自主决定，**不再由灰区阈值硬编码触发**。

**范围**：本 feature 为编排层 + VLM/Agent LLM 集成；9 个子检测器算法复用 v0.1 实现。旧 `run_judge` / Round 2 执行器保留作向后兼容降级路径，主链路无独立 Judge 阶段。

技术依据：本 feature 的 `plan.md`、`research.md`、`contracts/`；[`VERSION_ROADMAP.md`](../VERSION_ROADMAP.md)。

> **架构演进**：初版设计为「灰区 VLM + LLM Judge + Round 2 两轮」，实现时重构为 ReAct Agent 循环。下方 Project Structure / Phases 描述已按实际代码结构更新。

## Technical Context

**Language/Version**: Python 3.10+（MediaPipe 推荐 3.10–3.12；3.14 沿用 v0.1 GrabCut 回退）

**Primary Dependencies**（在 v0.1 基础上新增）:
- `httpx` 或 `requests` — Ollama / OpenAI 兼容 HTTP 客户端
- `pydantic`（可选）— Judge/VLM JSON 响应校验
- 保留：OpenCV, NumPy, PyYAML, jsonschema, pytest

**External Services**（V1 必需，可 mock）:
- **VLM**：`qwen2.5vl:7b` @ Ollama（默认 `http://localhost:11434`）或 `OPENAI_API_BASE` + `OPENAI_API_KEY`
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
- Agent 最大步数 = `max_rounds × 3`（默认 6 步）；超限或 `accept` 后强制出报告
- Agent actions **仅白名单**：`vlm_analyze`, `rerun_detector`, `dispatch_compression`, `accept`
- 同一 action 连续出现 ≥ 2 次 → 强制 `accept`（防小模型死循环）
- VLM/Judge 不可用 → 降级 + `decision_trace` 记录原因（不阻塞主链路）
- Agent **不得**替代 ReportGenerator MOS 聚合（FR-009）
- Evidence.detail 保持中文（Constitution III）
- v0.1 golden JSON **仍须**通过 v0.1 schema（无 Agent 时）或 v1 schema（Agent 开启时向后兼容）

**Scale/Scope**:
- 单帧 + 批量目录（继承 001 CLI）
- 子检测器：复用 v0.1 的 9 个检测器（demo 上 6/8 可用；`banding` 检出率偏低、`hand_anomaly` 实验性）
- Deep Mode：**未实现**（`--mode deep` 返回 exit 2，V2 计划）
- Sample：含低置信度项的帧 + mock 集成测试夹具

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
│   ├── orchestrator.py        # AgentOrchestrator + run_react_agent 循环 + 工具执行器
│   ├── router.py              # Confidence bands → dispatch/skip（灰区也 dispatch）
│   ├── context.py             # AgentContext（agent_steps, merged_degradations, vlm_results）
│   ├── judge_client.py        # JudgeClient ABC + OllamaJudgeClient + MockJudgeClient + RuleBasedJudgeClient
│   ├── actions.py             # 旧 Round 2 执行器（向后兼容保留）
│   └── prompts.py             # AGENT_SYSTEM_PROMPT + AGENT_OBSERVE_TEMPLATE（ReAct）+ 旧 Judge/VLM prompts
├── vlm/
│   ├── __init__.py
│   ├── client.py              # Ollama / OpenAI-compatible HTTP
│   ├── confirm.py             # run_vlm_confirm_for_item（单项，供 Agent 调用）+ 旧 run_vlm_confirm
│   └── fuse.py                # detector × VLM fusion rules
├── pipeline/
│   ├── fast_pipeline.py       # v0.1 fixed（--legacy-fixed 回退）
│   └── agent_pipeline.py      # V1 默认：Orchestrator → Report
├── cli/
│   └── main.py                # --mode fast → agent_pipeline；env 检测
├── config/
│   └── loader.py              # + AgentConfig, VLMConfig, JudgeConfig, ReportConfig.mos_model
└── models/
    └── agent.py               # AgentStep, AgentAction, VLMResult, JudgeOutput, RoutingDecision, AgentContext

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
| ReAct Agent 循环 | LLM 自主决策是否调 VLM / 补检（FR-002~005） | 旧灰区硬编码 + 两轮 Judge 无法处理「MOS 低无检出」与「高置信无需 VLM」等差异化场景 |

## Phase 0 Output

See [research.md](./research.md) — VLM/Judge 客户端、降级策略、mock 方案、Router 置信度分级均已决议。

## Phase 1 Output

See:
- [data-model.md](./data-model.md) — AgentContext, VLMResult, JudgeOutput, RoutingDecision
- [contracts/](./contracts/) — CLI delta、JSON Schemas
- [quickstart.md](./quickstart.md) — Ollama 拉起 → 灰区样本 → mock/实机验证

## Implementation Phases (for /speckit-tasks)

> 实际实现时 P3–P5 由「灰区 VLM + Judge + Round 2」演进为 ReAct Agent 循环。下表保留原阶段划分作历史记录，最终交付以 ReAct 架构为准（见 tasks.md 演进说明）。

| Phase | Focus | Key deliverables |
|-------|-------|------------------|
| P1 Setup | Config + models + HTTP client skeleton | `config.agent`, `models/agent.py`, `vlm/client.py` mock |
| P2 Router | Fast mode routing + trace | `agent/router.py`（灰区也 dispatch） |
| P3 VLM Confirm | ROI confirm + fusion（供 Agent 调用） | `vlm/confirm.py`（`run_vlm_confirm_for_item`）, `vlm/fuse.py` |
| P4 ReAct Agent | LLM decide 循环 + 工具执行器 | `agent/judge_client.py`（decide）, `agent/orchestrator.py`（`run_react_agent`） |
| P5 工具执行器 | vlm_analyze / rerun_detector / dispatch_compression / accept | `agent/orchestrator.py` 内工具函数 |
| P6 CLI | Wire `--mode fast` → agent | `agent_pipeline.py`, `--legacy-fixed` flag |
| P7 Polish | Tests, quickstart, README | `pytest -m vlm`, 样本, `agent_meta.agent_steps` 验证 |

**MVP checkpoint（002）**: mock 集成测试通过 — Agent 自主触发 `vlm_analyze` + `accept`，trace 含 `agent_step`，`agent_driven_vlm=true`。

## Deferred (Out of 002 Scope)

- Deep Mode VLM 粗分（`--mode deep` 完整实现，V2，已推迟；`vlm_discover` 已覆盖核心主动发现需求）
- `hand_anomaly` CV 检测器多指 / 粘连 / 手部模糊（实验性 MVP；`vlm_discover` 工具可由 VLM 主动发现手部语义异常）
- ~~TemporalFlicker（V2）~~ → **已实现**（V2）
- ~~视频 clip 输入~~ → **已实现**（V2：`VideoClipRunner`）
