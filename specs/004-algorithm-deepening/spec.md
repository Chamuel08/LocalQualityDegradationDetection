# Feature Specification: 004 算法深化（时序建模升级 + 多模态画质归因）

**Feature Branch**: `004-algorithm-deepening`

**Created**: 2026-07-23

**Status**: Implemented（代码已落地，单元测试通过）

**Depends on**:
- [`002-v1-agent-layer`](../002-v1-agent-layer/spec.md) — ReAct Agent 编排层
- [`003-v2-video-and-discover`](../003-v2-video-and-discover/spec.md) — `TemporalFlicker` / `vlm_discover` / `VideoClipRunner`

**Motivation**: 在 V2 基础上深化两条算法线，把项目从「工程偏重」往「算法深度」拨：
- **时序建模**：V2 的 `TemporalFlicker` 仅做亮度/色相跳变，无法区分真闪烁 vs 镜头运动，也无局部定位
- **可解释质量分析 / 局部画质劣化检测**：缺自然语言画质描述与业务场景归因
- **结合业务场景**：检出劣化后未映射到点播、直播、推荐、转码增强等业务场景与修复建议
- **客观评测 / 效果归因**：缺 prompt 对检测率/FPR 影响的消融实验

本 spec 在 V2 之上深化算法能力，**不重构**调度层，**不引入**新外部服务。

---

## User Scenarios & Testing *(mandatory)*

### User Story 1 - 时序建模升级：运动补偿 + 时序 SSIM + 局部闪烁 (Priority: P1)

V2 的 `TemporalFlicker` 把镜头运动误判为闪烁；本升级用光流对齐后残差能量（C1）区分真闪烁 vs 运动，用相邻补偿帧 SSIM（C2）衡量时序一致性，用分块热力图 + 连通域（C3）产出局部闪烁段（带 bbox）。

**Why this priority**: 时序建模是视频画质的核心维度；运动补偿直接解决 V2 把镜头运动误判为闪烁的痛点。

**Independent Test**: `tests/unit/test_temporal_flicker.py` — 稳定帧无闪烁 / 亮度跳变检出 / 运动补偿区分运动 / 局部闪烁热力图 + bbox / 可关闭。

**Acceptance Scenarios**:

1. **Given** 稳定帧序列，**When** 检测，**Then** `is_flickering=False`，`mean_motion_compensated_delta < 1.0`，`temporal_ssim > 0.95`
2. **Given** 含亮度跳变的帧序列，**When** 检测，**Then** `is_flickering=True`，至少一段 `metric=luma_delta`
3. **Given** 纹理图平移（无闪烁），**When** `enable_motion_compensation=True`，**Then** `mean_luma_delta < 5.0`，`mean_motion_compensated_delta < 15.0`，`temporal_ssim > 0.8`
4. **Given** `enable_motion_compensation=False`，**When** 检测，**Then** `mean_motion_compensated_delta=0.0`，`temporal_ssim=1.0`，`flicker_heatmap=None`
5. **Given** 局部区域闪烁的帧序列，**When** `block_size=16`，**Then** `flicker_heatmap` 非空，`localized_segments` 含带 bbox 的段且 bbox 落在闪烁区域

### User Story 2 - VLM 画质自然语言描述 (Priority: P1)

新增 `vlm_caption` Agent 工具：VLM 对整帧生成自然语言画质描述（overall_quality / caption / primary_degradations / affected_regions / ux_impact），写入 `agent_meta.quality_caption` 与 `QualityReport.quality_caption`。

**Why this priority**: 多模态画质归因让检测结果从数值变成可读的自然语言总结，是可解释质量分析的关键一环。

**Independent Test**: `tests/unit/test_vlm_caption.py` — 成功 / 已存在跳过 / 调用次数耗尽 / VLM 不可用 / 字段缺失 / 非 dict 响应。

**Acceptance Scenarios**:

1. **Given** VLM 返回合法 caption JSON，**When** `vlm_caption`，**Then** `ctx.quality_caption` 填充，`vlm_calls_count += 1`，trace 含 `vlm_caption` stage
2. **Given** `ctx.quality_caption` 已存在，**When** 再次 `vlm_caption`，**Then** 跳过，不重复调用
3. **Given** VLM 不可用（返回 None），**When** `vlm_caption`，**Then** trace 记 `vlm_caption_failed`，`quality_caption` 保持 None
4. **Given** VLM 返回非 dict / 解析失败，**When** 解析，**Then** 不崩溃，`quality_caption` 保持 None

### User Story 3 - 业务场景归因 (Priority: P1)

`attribute_scenarios(degradations)` 把检出劣化映射到业务场景（转码增强 / 直播 / 推荐 / AIGC 审核）并给出修复建议，写入 `QualityReport.scenario_attribution`。

**Why this priority**: 检测出劣化只是第一步；归因让检测从「检出」升级到「可执行的业务建议」，打通检测与下游业务。

**Independent Test**: `tests/unit/test_attribution.py` — 压缩伪影→转码增强 / 模糊→推荐 / 手部异常→AIGC 审核 / 无劣化→空。

**Acceptance Scenarios**:

1. **Given** 含 `compression_artifact` 的劣化列表，**When** 归因，**Then** 产出 `scenario="转码增强"`，含修复建议
2. **Given** 含 `hand_anomaly` 的劣化列表，**When** 归因，**Then** 产出 `scenario="AIGC 审核"`
3. **Given** 空劣化列表，**When** 归因，**Then** 返回空列表，`scenario_attribution=None`

### User Story 4 - VLM prompt 消融实验 (Priority: P2)

提供 `baseline` / `strict` / `loose` 三种 `vlm_discover` prompt 变体，`scripts/vlm_prompt_ablation.py` 在带 GT 的基准集上跑出 TPR / FPR / 平均 findings / 平均延迟表。

**Why this priority**: 客观测评需要量化指标；消融实验量化 prompt 对检测率/误检率的影响，是效果归因的可落地形式。

**Independent Test**: `python scripts/vlm_prompt_ablation.py --mock --samples 40` — 产出 markdown 表，loose TPR > strict TPR，strict FPR < loose FPR。

**Acceptance Scenarios**:

1. **Given** `--mock` 模式，**When** 跑 40 样本，**Then** 产出三种变体的 TPR/FPR 表
2. **Given** 真实 manifest + ollama，**When** 跑，**Then** 调真实 VLM，产出真实指标
3. **Given** manifest 不存在且非 mock，**When** 跑，**Then** 退出码 1 + 提示

---

## Requirements *(mandatory)*

### Functional Requirements

**C 时序建模升级**:
- **FR-C1**: `detect_temporal_flicker` MUST 支持光流运动补偿（Farneback），产出 `mean_motion_compensated_delta` / `max_motion_compensated_delta`
- **FR-C2**: MUST 计算相邻补偿帧 SSIM，产出 `temporal_ssim`
- **FR-C3**: MUST 支持分块热力图 + 连通域，产出 `flicker_heatmap`（HxW float）与 `localized_segments`（带 bbox）
- **FR-C4**: `enable_motion_compensation=False` 时 MUST 跳过光流，`block_size<=0` 时 MUST 不生成热力图
- **FR-C5**: 触发判定取相对阈值超得最多的 metric，同分时优先 `luma_delta`（可解释性）

**D1 VLM 画质描述**:
- **FR-D1-1**: `vlm_caption` MUST 作为 ReAct Agent 白名单 action（`WHITELIST` 含 `vlm_caption`）
- **FR-D1-2**: MUST 对整帧编码 JPEG base64 后调 `VLMClient.confirm(VLM_CAPTION_PROMPT, b64)`，解析为 `quality_caption` dict
- **FR-D1-3**: 每帧 MUST 最多执行 1 次；受 `vlm.max_calls_per_frame` 总额限制
- **FR-D1-4**: 结果 MUST 写入 `agent_meta.quality_caption` 与 `QualityReport.quality_caption`，不进 `degradations[]`

**D3 业务场景归因**:
- **FR-D3-1**: `attribute_scenarios(degradations)` MUST 把劣化类型映射到业务场景 + 修复建议
- **FR-D3-2**: 结果 MUST 写入 `QualityReport.scenario_attribution`，序列化到报告 JSON

**D2 prompt 消融**:
- **FR-D2-1**: `VLM_DISCOVER_PROMPT_VARIANTS` MUST 含 `baseline` / `strict` / `loose` 三种 prompt
- **FR-D2-2**: `scripts/vlm_prompt_ablation.py` MUST 支持 `--manifest`（真实）/ `--mock`（演示）两种模式
- **FR-D2-3**: MUST 输出 TPR / FPR / 平均 findings / 平均延迟的 markdown 表

### Out of Scope (004)

- Deep Mode（VLM 先行 + 子检测器量化，重构整个调度层）— 推迟
- 训练自有 NR-IQA 模型 / PLCC/SRCC 拟合 — 不实现（客观评测用 CLIP-IQA + 消融表覆盖，不做模型训练）
- 端到端视频体验优化 — 仅做检测+归因，不做编码策略联动

## Success Criteria *(mandatory)*

- **SC-001**: `tests/unit/test_temporal_flicker.py` 全部通过（8 用例）
- **SC-002**: `tests/unit/test_vlm_caption.py` 全部通过（6 用例）
- **SC-003**: `tests/unit/test_attribution.py` 全部通过
- **SC-004**: `tests/unit/test_ui_app.py` 全部通过（含 quality_caption / scenario_attribution / flicker_heatmap 展示）
- **SC-005**: `python scripts/vlm_prompt_ablation.py --mock --samples 40` 产出合法 markdown 表
- **SC-006**: 全量 `tests/unit/` 通过（63 passed）
- **SC-007**: v1 schema 扩展 `vlm_caption` action enum、`quality_caption` / `scenario_attribution` 属性

## Assumptions

- V1/V2 编排层与 `TemporalFlicker` 已可用
- `vlm_caption` 复用 V1 的 `OllamaVLMClient`（同一 `qwen2.5vl:7b`），不引入新外部服务
- 光流用 OpenCV `cv2.calcOpticalFlowFarneback`（已是核心依赖）
- SSIM 用 `cv2.blur` 简化实现（不引入新依赖）

## Design References

| 文档 | 用途 |
|------|------|
| [`data-model.md`](data-model.md) | `TemporalFlickerResult` 扩展 / `ScenarioAttribution` / `quality_caption` 实体 |
| [`tasks.md`](tasks.md) | 任务清单 |
| [`../003-v2-video-and-discover/spec.md`](../003-v2-video-and-discover/spec.md) | V2 基线 |
| [`../002-v1-agent-layer/contracts/quality-report.v1.schema.json`](../002-v1-agent-layer/contracts/quality-report.v1.schema.json) | 扩展报告 Schema |
