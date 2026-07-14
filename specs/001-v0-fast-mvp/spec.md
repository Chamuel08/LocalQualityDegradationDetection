# Feature Specification: v0.1 Fast Mode MVP

**Feature Branch**: `001-v0-fast-mvp`

**Created**: 2026-07-09

**Status**: Draft

**Input**: 实现局部画质 badcase 单帧检测最小可运行版本（Fast Mode MVP）。技术设计以本 feature 目录下的 `plan.md`、`data-model.md`、`contracts/` 及 [`specs/`](../) 公开文档为准。

## User Scenarios & Testing *(mandatory)*

### User Story 1 - 单张 badcase 检测并出 JSON 报告 (Priority: P1)

算法或质检工程师拿到一张问题截图，希望在本地一条命令得到结构化 JSON：是否劣化、劣化类型、区域 bbox、可读 evidence、整体 MOS。

**Why this priority**: 最小可演示、可复现；所有后续能力的基础。

**Independent Test**: 对一张已知含绿边的 PNG 运行 CLI，输出 JSON 含 `edge_bleed` 劣化项且 `evidence.detail` 为中文描述。

**Acceptance Scenarios**:

1. **Given** 一张含明显绿边的 720p badcase PNG，**When** 用户执行 `detect.py --image <path> --mode fast`，**Then** stdout 或 `--output` 文件为合法 JSON，含至少 1 条 `degradations[]`，`detector` 为 `edge_bleed`
2. **Given** 一张无明显劣化的正常参考帧，**When** 同上命令，**Then** `degradations` 为空或 `overall_mos` ≥ 4.0，且无 critical 级别误报
3. **Given** 输入文件不存在，**When** 执行 CLI，**Then** 非零退出码且 stderr 含可读错误信息

---

### User Story 2 - 批量 badcase 目录筛图 (Priority: P1)

用户有一个目录存放多张 badcase 截图，需要批量生成报告文件供复核。

**Why this priority**: 对应真实 badcase 批量工作流（见 [`USE_CASE_BADCASE.md`](../USE_CASE_BADCASE.md)）。

**Independent Test**: 对含 3 张图的目录运行 `--image-dir`，产出 3 份 JSON（或指定 `--output-dir`）。

**Acceptance Scenarios**:

1. **Given** 目录含 3 张 PNG，**When** `detect.py --image-dir ./badcase/ --mode fast --output-dir ./reports/`，**Then** `./reports/` 下生成 3 个 JSON，文件名与输入可对应
2. **Given** 目录中 1 张损坏图片，**When** 批量处理，**Then** 其余图片仍成功；损坏项在报告或日志中标注 skip

---

### User Story 3 - HTML 报告供人工复核 (Priority: P2)

用户需要将结果给非工程同事查看，需要可视化 bbox 与 evidence 文本。

**Why this priority**: HTML 报告便于非工程同事复核；依赖 P1 JSON 结构。

**Independent Test**: `--output report.html` 可在浏览器打开，可见劣化框与 evidence 摘要。

**Acceptance Scenarios**:

1. **Given** 含劣化的 badcase 图，**When** `--output report.html`，**Then** HTML 含输入图预览、劣化列表、每条 evidence 的 method/metric/value/threshold/detail
2. **Given** 无劣化帧，**When** 生成 HTML，**Then** 显示「未检出显著劣化」类提示，不报错

---

### User Story 4 - 低码率 block 噪点检出 (Priority: P1)

编码链路常见 H.264 块效应 badcase，需在 MVP 中覆盖（CompressionArtifact）。

**Why this priority**: 编码链路常见劣化（H.264 block 效应）；见 `research.md` CompressionArtifact 决策。

**Independent Test**: 对合成或真实 block 噪点样本，JSON 含 `compression_artifact` 或合并后的 encoding 类劣化项。

**Acceptance Scenarios**:

1. **Given** 含明显 block 噪点的 720p 帧，**When** Fast Mode 检测，**Then** 检出 encoding/block 类劣化，`evidence` 含 blockiness 相关 metric
2. **Given** 高质量无 block 帧，**When** 检测，**Then** 不因 compression 误报

---

### Edge Cases

- 输入含大面积 UI overlay（字幕、水印、贴纸等）：应支持 `--ignore-regions` JSON，overlay 区域不参与检测（见 [`USE_CASE_BADCASE.md`](../USE_CASE_BADCASE.md) §6）
- 非 720p 分辨率（540p/1080p）：应能完成检测，evidence 可记录原始分辨率
- 无 GPU 环境：允许 CPU 降级，延迟可放宽，功能不崩溃
- 单帧无 VLM/LLM：Fast MVP 不依赖外部 API 即可完成主流程

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: 系统 MUST 提供 CLI `detect.py`，接受 `--image` 单帧或 `--image-dir` 批量输入
- **FR-002**: 系统 MUST 支持 `--mode fast`（本 feature 唯一必需模式）
- **FR-003**: 系统 MUST 输出符合 `contracts/quality-report.schema.json` 的 JSON（`degradations[]`、`overall_mos`、`decision_trace`）
- **FR-004**: 每条 `degradations[]` MUST 含 Evidence 四要素：`method`、`metric`、`value`、`threshold`、`detail`（中文）
- **FR-005**: 系统 MUST 实现 Stage 1 Global Scan（分割/提名，可简化版 face parsing + edge 提名）
- **FR-006**: 系统 MUST 实现子检测器 `edge_bleed`（Lab ΔE + green channel；见 `research.md`）
- **FR-007**: 系统 MUST 实现子检测器 `compression_artifact`（DCT blockiness；见 `research.md`）
- **FR-008**: 系统 MUST 实现 Stage 3 Report Generator，聚合 MOS 与 degradations（见 `data-model.md`）
- **FR-009**: 系统 MUST 支持 `--output` 写入 JSON 或 HTML 文件；默认 stdout JSON
- **FR-010**: 系统 MUST 支持可选 `--metadata` / `--ignore-regions`（sidecar JSON）
- **FR-011**: 系统 MUST 在 `decision_trace` 中记录各阶段调度决策（L1 可解释最小集）

### Out of Scope (v0.1 MVP)

- Deep Mode、VLM 视觉确认、LLM Agent 编排（均属 V1，见 002）
- 实时流接入
- 完整 100 张数据集与 benchmark CI（可留 stub + sample 数据）

> **演进说明**：v0.1 MVP 原始切片仅含 `edge_bleed` + `compression_artifact` 两个检测器；后续提交将子检测器扩展至 9 类（blur / mosaic / banding / background / hair_texture / face_artifact / hand_anomaly，其中 `hand_anomaly` 为实验性 MVP）。这些扩展检测器在 `--legacy-fixed` v0.1 路径下同样确定性运行，详见 README「检测器」表。

### Deferred to V1（明确非放弃）

以下能力在 v0.1 **刻意延后**，V1 产品 **必须** 交付：

- **VLM 视觉确认** — badcase ROI 兜底（由 LLM Agent 自主触发，非硬编码灰区）
- **LLM Agent 编排** — ReAct 循环：LLM 自主决策是否调 VLM / 是否补检（取代早期设计的 Judge + Round 2）
- **AgentOrchestrator** — 动态编排替代 fixed pipeline
- **Deep Mode** — 单条 badcase 深度归因（V2 计划，002 未实现）

详见 [`specs/002-v1-agent-layer/spec.md`](../002-v1-agent-layer/spec.md)、[`specs/VERSION_ROADMAP.md`](../VERSION_ROADMAP.md)。

### Key Entities

- **SingleFrameInput**: 帧图像路径、frame_id、可选 metadata、ignore_regions
- **DegradationItem**: 劣化类型、severity、confidence、mos_impact、bbox、evidence、detector
- **Evidence**: method、metric、value、threshold、detail
- **TraceEntry**: stage、module、decision、duration_ms
- **QualityReport**: overall_mos、degradations、decision_trace、processing_time_ms

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: 用户可在 5 分钟内从 clone 到对 sample 图片跑出 JSON 结果（含 README quickstart）
- **SC-002**: 绿边样本检出 recall ≥ 80%（sample set ≥ 5 张）
- **SC-003**: block 噪点样本 recall ≥ 80%（sample set ≥ 5 张）
- **SC-004**: 正常帧误报率 ≤ 10%（sample set ≥ 5 张）
- **SC-005**: 单帧 Fast Mode 在 GPU（T4/3060 级）P50 延迟 < 500ms（MVP 放宽；长期目标 200ms）
- **SC-006**: 100% 输出 JSON 通过项目 Schema 校验（golden sample ≥ 3 张）
- **SC-007**: HTML 报告可在浏览器打开并展示 bbox + evidence，无需额外服务

## Assumptions

- 输入为已解码 BGR 帧（PNG/JPG），上游负责录屏抽帧
- Python 3.10+，OpenCV、NumPy 可用；GPU 可选
- 首版可使用轻量 face parsing / 前景分割，不强制训练新模型
- 设计细节、算法参数、目录结构以本 feature 的 `plan.md` / `research.md` / `contracts/` 为准
- Sample 数据位于 `data/sample/`（plan 阶段定义最小集）

## Design References

| 文档 | 用途 |
|------|------|
| [`data-model.md`](data-model.md) | 实体与 Schema |
| [`contracts/quality-report.schema.json`](contracts/quality-report.schema.json) | JSON 输出校验 |
| [`contracts/cli-contract.md`](contracts/cli-contract.md) | CLI 契约 |
| [`research.md`](research.md) | 算法决策（EdgeBleed、Compression） |
| [`../USE_CASE_BADCASE.md`](../USE_CASE_BADCASE.md) | 业务场景 |
| [`../method_selection.md`](../method_selection.md) | 算法选型 |
