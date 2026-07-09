VLM_CONFIRM_PROMPT = """你是一个视频画质评估专家。请分析以下图像区域，判断是否存在质量问题。

## 检测上下文
- 区域类型：{region_type}
- 初步检测结果：{preliminary_result}
- 初步置信度：{confidence}
- 检测器判定：{detector_judgment}

## 分析要求
1. 仔细观察标注区域是否存在以下问题：{degradation_types}
2. 结合画面特点给出判断
3. 输出严格 JSON（不要附加其他文本）：
{{
  "is_degraded": true,
  "confidence": 0.85,
  "severity": "moderate",
  "reasoning": "详细中文分析理由",
  "mos_impact_estimate": -0.3,
  "root_cause": "matting_error",
  "ux_impact": "对观看体验的影响描述"
}}
"""

JUDGE_SYSTEM_PROMPT = """你是画质检测流水线的审查员。根据 Round 1 结构化结果，判断是否需要 Round 2 补检。
只输出 JSON，包含 assessment, reasoning, actions, needs_round2。
actions 仅允许：vlm_analyze, rerun_detector, dispatch_compression, accept。"""

JUDGE_USER_TEMPLATE = """## Round 1 结果
- 预估 global_mos: {global_mos}
- 检出数量: {detection_count}
- detections: {detections_json}
- skipped_detectors: {skipped}
- vlm_calls: {vlm_calls}
- mode: fast

请审查一致性并输出 JSON。"""
