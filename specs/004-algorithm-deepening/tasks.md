# Tasks: 004 算法深化

**Feature**: `004-algorithm-deepening`

**Status**: Implemented

---

## C. 时序建模升级（TemporalFlicker）

- [x] **C1** 运动补偿：`temporal_flicker/detector.py` 加光流对齐（Farneback）+ 残差能量 → `mean/max_motion_compensated_delta`
- [x] **C2** 时序 SSIM：相邻补偿帧 SSIM → `temporal_ssim`
- [x] **C3** 局部闪烁热力图：分块时序变化 → `flicker_heatmap` + 带 bbox 的 `localized_segments`
- [x] **C-test** `tests/unit/test_temporal_flicker.py`：稳定帧 / 亮度跳变 / 运动补偿区分 / 局部闪烁 / 可关闭 / 单帧兜底

## D1. VLM 画质自然语言描述（vlm_caption）

- [x] `prompts.py` 加 `VLM_CAPTION_PROMPT`
- [x] `AgentAction` 加 `vlm_caption`；`WHITELIST` 加 `vlm_caption`
- [x] `AgentContext` / `AgentMeta` 加 `quality_caption`
- [x] `orchestrator._execute_vlm_caption` 实现
- [x] `run_react_agent` 派发 `vlm_caption`；`AGENT_SYSTEM_PROMPT` / `AGENT_OBSERVE_TEMPLATE` 更新
- [x] `QualityReport` 加 `quality_caption`；`report_to_dict` 序列化
- [x] `ReportGenerator.generate` 接收并写入 `quality_caption`
- [x] `tests/unit/test_vlm_caption.py`：成功 / 跳过 / 次数耗尽 / VLM 不可用 / 字段缺失 / 非 dict

## D2. VLM prompt 消融实验

- [x] `prompts.py` 加 `VLM_DISCOVER_PROMPT_VARIANTS`（baseline / strict / loose）
- [x] `scripts/vlm_prompt_ablation.py`：`--manifest` 真实模式 / `--mock` 演示模式 / markdown 表输出
- [x] mock 模式验证：loose TPR > strict TPR，strict FPR < loose FPR

## D3. 业务场景归因（scenario_attribution）

- [x] 新建 `src/lqdd/attribution/scenario.py`：`ScenarioAttribution` + `attribute_scenarios`
- [x] `ReportGenerator.generate` 调用 `attribute_scenarios` 写入 `QualityReport.scenario_attribution`
- [x] `report_to_dict` 序列化 `scenario_attribution`
- [x] `tests/unit/test_attribution.py`：压缩→转码 / 模糊→推荐 / 手部→AIGC / 空→空

## GUI 展示

- [x] `ui/app.py` 加 `_quality_caption_md` / `_scenario_attribution_md` / `_flicker_heatmap_rgb`
- [x] `run_single` 返回 `caption_md` / `scenario_md`
- [x] `run_video` 返回 `heatmap_rgb` + 展示 C1/C2/C3 指标
- [x] `build_ui` 加 VLM 画质描述 / 业务场景归因 / 局部闪烁热力图面板
- [x] `tests/unit/test_ui_app.py` 扩展（12 用例通过）

## Specs & Schema

- [x] 新建 `specs/004-algorithm-deepening/`（spec.md / data-model.md / tasks.md）
- [x] 扩展 `specs/002-v1-agent-layer/contracts/quality-report.v1.schema.json`（vlm_caption enum / quality_caption / scenario_attribution）

## README

- [ ] README 特性 + 结果展示加新算法能力（VLM 画质描述 / 场景归因 / 时序建模升级 / 消融表）
