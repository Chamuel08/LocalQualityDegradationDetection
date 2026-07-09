# Tasks: V1 Agent Layer（VLM 兜底 + LLM Judge）

**Input**: Design documents from `specs/002-v1-agent-layer/`  
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/, quickstart.md  
**Depends on**: [`001-v0-fast-mvp`](../001-v0-fast-mvp/) 代码与测试已通过

**Organization**: Tasks grouped by user story (US1 → US2 → US3). Constitution V + spec SC-004 要求 mock 集成测试与 contract 校验。

## Format: `[ID] [P?] [Story] Description`

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: V1 包结构、依赖、配置、测试夹具

- [X] T001 Create `src/lqdd/agent/` and `src/lqdd/vlm/` package trees per plan.md
- [X] T002 Add `httpx` dependency in `pyproject.toml` for Ollama/OpenAI HTTP client
- [X] T003 [P] Extend `config.example.yaml` with `agent`, `vlm`, `judge` sections from data-model.md
- [X] T004 [P] Create `tests/fixtures/mock_vlm_responses/` and `tests/fixtures/mock_judge_responses/` with sample JSON
- [X] T005 [P] Extend `tests/conftest.py` with v1 schema path and mock client fixtures

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Agent 模型、配置加载、HTTP 客户端骨架、Context — MUST complete before user stories

**⚠️ CRITICAL**: No user story work until this phase is complete

- [X] T006 [P] Implement agent dataclasses in `src/lqdd/models/agent.py` (RoutingDecision, VLMResult, JudgeOutput, AgentAction, AgentContext)
- [X] T007 Extend `src/lqdd/config/loader.py` with AgentConfig, VLMConfig, JudgeConfig and env overrides
- [X] T008 [P] Implement prompt templates in `src/lqdd/agent/prompts.py` (VLM Confirm + Judge per §8.2/§13)
- [X] T009 [P] Implement `src/lqdd/agent/context.py` AgentContext factory and state helpers
- [X] T010 Implement `src/lqdd/vlm/client.py` with VLMClient protocol, OllamaVLMClient, MockVLMClient
- [X] T011 Implement `src/lqdd/agent/judge_client.py` with JudgeClient protocol, OllamaJudgeClient, MockJudgeClient
- [X] T012 [P] Add contract test skeleton in `tests/contract/test_vlm_schema.py` against `specs/002-v1-agent-layer/contracts/vlm-confirm.schema.json`
- [X] T013 [P] Add contract test skeleton in `tests/contract/test_judge_schema.py` against `specs/002-v1-agent-layer/contracts/llm-judge.schema.json`

**Checkpoint**: Foundation ready — config, models, mock clients exist

---

## Phase 3: User Story 1 — 灰区 VLM 确认 (Priority: P1) 🎯 MVP

**Goal**: confidence ∈ [0.4, 0.7] 触发 VLM Confirm，degradation 含 `vlm_reasoning` 与 `vlm_confirm` trace

**Independent Test**: Mock VLM 返回 `is_degraded=true`；灰区 edge_bleed JSON 含 `vlm_reasoning.reasoning` 与 trace stage `vlm_confirm`

### Implementation for User Story 1

- [X] T014 [P] [US1] Implement confidence band router in `src/lqdd/agent/router.py` (high/grey/low per §3.1)
- [X] T015 [US1] Implement ROI crop + VLM Confirm in `src/lqdd/vlm/confirm.py` using `vlm/client.py`
- [X] T016 [US1] Implement detector×VLM fusion in `src/lqdd/vlm/fuse.py` per §8.4 rules
- [X] T017 [US1] Map VLMResult to `DegradationItem.vlm_reasoning` in `src/lqdd/vlm/fuse.py`
- [X] T018 [US1] Implement VLM unavailable degradation in `src/lqdd/vlm/confirm.py` (hard_decision_threshold + trace `vlm_skipped`)
- [X] T019 [P] [US1] Add unit tests in `tests/unit/test_router.py` for confidence bands
- [X] T020 [P] [US1] Add unit tests in `tests/unit/test_vlm_fuse.py` for fusion rules
- [X] T021 [US1] Add unit test in `tests/unit/test_vlm_confirm.py` for mock VLM gray-zone flow

**Checkpoint**: US1 complete — VLM confirm + fusion unit tests green

---

## Phase 4: User Story 2 — LLM Judge + Round 2 (Priority: P1)

**Goal**: Round 1 后 Judge 审查；`dispatch_compression` 等白名单 action 触发 Round 2

**Independent Test**: Mock Judge `needs_round2=true` + `dispatch_compression`；MOS 低无检出样本 Round 2 后含 `compression_artifact`

### Implementation for User Story 2

- [X] T022 [US2] Implement Judge JSON parse + whitelist filter in `src/lqdd/agent/judge.py`
- [X] T023 [US2] Implement Round 2 action executor in `src/lqdd/agent/actions.py` (vlm_analyze, rerun_detector, dispatch_compression, accept)
- [X] T024 [US2] Implement `AgentOrchestrator.run()` 2-round state machine in `src/lqdd/agent/orchestrator.py`
- [X] T025 [US2] Wire detectors + router + VLM + judge into `src/lqdd/agent/orchestrator.py` Round 1 flow
- [X] T026 [US2] Wire Round 2 actions + max_rounds cap into `src/lqdd/agent/orchestrator.py`
- [X] T027 [US2] Ensure Judge does NOT alter MOS formula — delegate to `src/lqdd/report/generator.py` only
- [X] T028 [P] [US2] Add unit tests in `tests/unit/test_judge_parser.py` for parse + whitelist rejection
- [X] T029 [P] [US2] Add unit tests in `tests/unit/test_actions_whitelist.py` for dispatch_compression and fuzz reject
- [X] T030 [US2] Add integration test in `tests/integration/test_agent_pipeline_mock.py` for Judge Round 2 dispatch_compression

**Checkpoint**: US1 + US2 — full Agent loop with mock VLM/Judge

---

## Phase 5: User Story 3 — decision_trace 完整可回溯 (Priority: P2)

**Goal**: trace 含 routing、detection、vlm_confirm、judge、aggregation 及 Round 2 摘要

**Independent Test**: 灰区 + Round 2 mock 样本；`decision_trace` 含全部 stage 且 `decision` 字段可人工读懂

### Implementation for User Story 3

- [X] T031 [US3] Emit `routing` trace entries from `src/lqdd/agent/router.py` into AgentContext
- [X] T032 [US3] Emit `vlm_confirm` and `judge` trace entries from `src/lqdd/agent/orchestrator.py`
- [X] T033 [US3] Add `agent_meta` and `vlm_reasoning_summary` to report in `src/lqdd/report/generator.py` / `src/lqdd/models/report.py`
- [X] T034 [US3] Extend `report_to_dict()` in `src/lqdd/models/report.py` for v1 fields (`system_version` 1.0.0, performance.vlm_ms/judge_ms)
- [X] T035 [US3] Add integration assertion in `tests/integration/test_agent_pipeline_mock.py` for full trace stages

**Checkpoint**: US3 complete — auditable decision_trace

---

## Phase 6: CLI & Pipeline Integration

**Purpose**: `--mode fast` 默认 Agent；`--legacy-fixed` 保留 v0.1

- [X] T036 Implement `src/lqdd/pipeline/agent_pipeline.py` wrapping AgentOrchestrator
- [X] T037 Wire `--mode fast` to `agent_pipeline.py` in `src/lqdd/cli/main.py` when `agent.enabled=true`
- [X] T038 Add `--legacy-fixed` flag and `LQDD_AGENT_ENABLED=false` fallback in `src/lqdd/cli/main.py`
- [X] T039 Return exit 2 for `--mode deep` with message per `contracts/cli-contract.md`
- [X] T040 [P] Add contract test in `tests/contract/test_report_v1_schema.py` against `quality-report.v1.schema.json`

---

## Phase 7: Polish & Cross-Cutting Concerns

**Purpose**: 灰区样本、README、回归、quickstart 验证

- [X] T041 [P] Add grey-zone sample frame in `scripts/generate_synthetic_samples.py` → `data/sample/frames/edge/grey_edge_01.png`
- [X] T042 [P] Add mock fixture for MOS-low-no-detection case in `tests/fixtures/mock_judge_responses/dispatch_compression.json`
- [X] T043 Update README.md Implemented table for V1 Agent layer
- [X] T044 Run `pytest tests/ -m "not vlm" -q` and fix failures; ensure v0.1 `--legacy-fixed` tests still pass
- [X] T045 Validate quickstart.md steps with mock clients (no Ollama required)
- [X] T046 Bump default `system_version` to `1.0.0` in config when agent enabled

---

## Dependencies & Execution Order

### Phase Dependencies

- **Phase 1 Setup**: No dependencies
- **Phase 2 Foundational**: Depends on Phase 1 — **BLOCKS all user stories**
- **Phase 3 US1**: Depends on Phase 2
- **Phase 4 US2**: Depends on Phase 3 (VLM fusion pattern)
- **Phase 5 US3**: Depends on Phase 4 (orchestrator emits traces)
- **Phase 6 CLI**: Depends on Phase 4 minimum (orchestrator runnable)
- **Phase 7 Polish**: Depends on Phases 3–6

### User Story Dependencies

- **US1 (P1)**: After Foundational — **MVP part A** (VLM gray zone)
- **US2 (P1)**: After US1 — **MVP part B** (Judge + Round 2)
- **US3 (P2)**: After US2 — trace completeness

### Parallel Opportunities

- Phase 1: T003, T004, T005 in parallel
- Phase 2: T006, T008, T009, T012, T013 in parallel; then T007, T010, T011 sequential
- Phase 3: T014, T019, T020 in parallel; T015–T018 sequential
- Phase 4: T028, T029 in parallel
- Phase 7: T041, T042 in parallel

---

## Parallel Example: User Story 1

```bash
# Router + unit tests in parallel:
T014: src/lqdd/agent/router.py
T019: tests/unit/test_router.py
T020: tests/unit/test_vlm_fuse.py

# Then VLM pipeline:
T015 → T016 → T017 → T021
```

---

## Implementation Strategy

### MVP First (US1 + US2 mock)

1. Complete Phase 1 + Phase 2
2. Complete Phase 3 (US1) + Phase 4 (US2)
3. **STOP and VALIDATE**: `pytest tests/integration/test_agent_pipeline_mock.py -q`
4. Wire CLI (Phase 6) for demo

### Incremental Delivery

1. Setup + Foundational → mock clients ready
2. US1 → VLM gray zone with mock
3. US2 → Judge + Round 2 with mock (**full Agent MVP**)
4. US3 → trace polish
5. CLI + Polish → production default path

### Suggested MVP Scope

**Minimum shippable**: Phase 1 + 2 + 3 + 4 + T036–T038 — **30 tasks**  
**Full 002 spec**: All phases — **46 tasks**

---

## Notes

- No new sub-detectors (face/hair) — use edge_bleed + compression_artifact only
- Deep Mode stub only (`--mode deep` exit 2)
- CI default: `pytest -m "not vlm"`; Ollama tests optional `@pytest.mark.vlm`
- Judge MUST NOT compute MOS — ReportGenerator only
- v0.1 golden samples validated via `--legacy-fixed`
