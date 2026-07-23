# Data Model: 004 算法深化

**Feature**: `004-algorithm-deepening`

**Depends on**: [`003-v2-video-and-discover/data-model.md`](../003-v2-video-and-discover/data-model.md) — `TemporalFlickerResult` / `FlickerSegment` / `AgentMeta`

> 本文档定义 004 新增/扩展实体。所有变更**向后兼容**（新字段可选）。

---

## 1. C 时序建模升级：TemporalFlickerResult 扩展

### FlickerSegment 扩展

| Field | Type | Required | Notes |
|-------|------|----------|-------|
| `bbox` | `list[int] \| None` | ❌ | C3 局部闪烁段填充 `[x, y, w, h]`；全局段为 None |

### TemporalFlickerResult 扩展

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `mean_motion_compensated_delta` | `float` | `0.0` | C1 运动补偿后残差能量均值 |
| `max_motion_compensated_delta` | `float` | `0.0` | C1 运动补偿后残差能量最大值 |
| `temporal_ssim` | `float` | `1.0` | C2 相邻补偿帧 SSIM 均值 [0,1]，越高越一致 |
| `flicker_heatmap` | `np.ndarray \| None` | `None` | C3 HxW float，每像素跨帧平均时序变化 |
| `localized_segments` | `list[FlickerSegment]` | `[]` | C3 带 bbox 的局部闪烁段 |
| `method` | `str` | `temporal_luma_hue_motion_ssim` | 升级后方法名 |

### detect_temporal_flicker 签名扩展

```python
def detect_temporal_flicker(
    frames: list[np.ndarray],
    luma_delta_threshold: float = 8.0,
    hue_delta_threshold: float = 6.0,
    min_flicker_ratio: float = 0.15,
    motion_delta_threshold: float | None = None,   # None → 取 luma_delta_threshold
    enable_motion_compensation: bool = True,        # C1 开关
    block_size: int = 16,                            # C3 分块尺寸；<=0 不生成热力图
) -> TemporalFlickerResult
```

### 算法

**C1 运动补偿时序变化**：
1. 对相邻帧 `prev_gray` / `next_gray` 用 `cv2.calcOpticalFlowFarneback` 估光流
2. 用 `cv2.remap` 把 `next` warp 回 `prev` 坐标（`BORDER_REPLICATE`）
3. 残差 = `cv2.absdiff(prev_gray, warped)`；`mc_delta = residual.mean()`

**C2 时序 SSIM**：
1. 对 `prev_gray` 与 `warped` 算 SSIM（`cv2.blur` 简化实现，11x11 窗口，C1=6.5025, C2=58.5225）
2. `temporal_ssim` = 所有相邻帧对 SSIM 的均值

**C3 局部闪烁热力图**：
1. 累加每对相邻帧的（运动补偿后）残差 → `heatmap_accum`
2. `heatmap = heatmap_accum / (n-1)`
3. 分块均值 → `block_mask = block_avg >= motion_delta_threshold`
4. `cv2.connectedComponentsWithStats` → 每个连通域产出一个带 bbox 的 `FlickerSegment`

**触发判定**：`luma_delta` / `hue_delta` / `motion_compensated_delta` 任一超阈值，取相对阈值超得最多的 metric；同分时优先 `luma_delta`（用 `>` 而非 `>=` 实现 tie-breaking）。

---

## 2. D1 VLM 画质描述：quality_caption

### VLM_CAPTION_PROMPT（提示词契约）

`vlm_caption` 调用 VLM 时使用的 prompt（定义于 `src/lqdd/agent/prompts.py`）。要求 VLM 输出严格 JSON：

```json
{
  "overall_quality": "good",
  "caption": "整体画质良好，左上角存在轻度压缩块效应，对观看体验影响较小。",
  "primary_degradations": ["compression_artifact"],
  "affected_regions": ["左上角"],
  "ux_impact": "对观看体验影响较小"
}
```

- `overall_quality` 仅允许 `excellent` / `good` / `fair` / `poor` / `bad`
- `primary_degradations` 为数组（可为空）
- `caption` 为 1-2 句中文描述

### AgentAction 白名单扩展

```python
action: Literal[
    "vlm_analyze", "rerun_detector", "dispatch_compression",
    "vlm_discover", "vlm_caption", "accept"
]
```

| action | Parameters | Constraints |
|--------|------------|-------------|
| `vlm_caption` | `reason?` | 对全帧生成画质描述；每帧最多 1 次；与 `vlm_discover` 互斥；结果写入 `agent_meta.quality_caption`（不进 `degradations[]`） |

### AgentContext / AgentMeta 扩展

| Field | Type | Notes |
|-------|------|-------|
| `AgentContext.quality_caption` | `dict[str, Any] \| None` | VLM 画质描述（不进 degradations） |
| `AgentMeta.quality_caption` | `dict[str, Any] \| None` | 序列化到报告；未触发时为 None |

### quality_caption 结构

| Field | Type | Notes |
|-------|------|-------|
| `overall_quality` | `str` | `excellent` / `good` / `fair` / `poor` / `bad` / `unknown` |
| `caption` | `str` | 1-2 句中文描述 |
| `primary_degradations` | `list[str]` | 主要劣化类型（可为空） |
| `affected_regions` | `list[str]` | 影响区域（可为空） |
| `ux_impact` | `str` | 体验影响描述 |

### TraceEntry 扩展

| stage | module | decision |
|-------|--------|----------|
| `vlm_caption` | `VLMCaption` | `vlm_caption_complete` / `vlm_caption_failed` |

---

## 3. D3 业务场景归因：ScenarioAttribution

### ScenarioAttribution

| Field | Type | Required | Notes |
|-------|------|----------|-------|
| `scenario` | `str` | ✅ | 业务场景名（如「转码增强」/「直播」/「推荐」/「AIGC 审核」） |
| `confidence` | `float` | ✅ | 归因置信度 [0, 1] |
| `degradation_types` | `list[str]` | ✅ | 触发该场景的劣化类型列表 |
| `evidence_refs` | `list[str]` | ✅ | 证据引用（degradation_id 列表） |
| `recommendation` | `str` | ✅ | 修复建议（中文） |

```python
@dataclass
class ScenarioAttribution:
    scenario: str
    confidence: float
    degradation_types: list[str]
    evidence_refs: list[str]
    recommendation: str
```

### attribute_scenarios

| Signature | Notes |
|-----------|-------|
| `(degradations: list[DegradationItem]) -> list[ScenarioAttribution]` | 按 detector/degradation_type 映射到业务场景；空列表返回空 |

### 场景映射规则

| detector / degradation_type | scenario | recommendation |
|------------------------------|----------|----------------|
| `compression_artifact` | 转码增强 | 提升码率 / 换用 HEVC / AV1 |
| `blur_artifact` | 推荐 | 上游素材质量复核 / 推荐链路降级 |
| `mosaic_artifact` / `banding_artifact` | 直播 | 编码器参数调优 / 色深提升 |
| `hand_anomaly` / `face_artifact` | AIGC 审核 | 生成模型迭代 / 人工复核 |
| `edge_bleed` | 后期合成 | 抠像算法升级 / 边缘溢色修复 |

### QualityReport 扩展

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `scenario_attribution` | `list[dict] \| None` | `None` | 业务场景归因列表 |
| `quality_caption` | `dict \| None` | `None` | VLM 画质描述 |

---

## 4. D2 prompt 消融：VLM_DISCOVER_PROMPT_VARIANTS

| Variant | 特征 | 适用场景 |
|---------|------|---------|
| `baseline` | 当前生产 prompt（5 类异常 + JSON schema） | 生产默认，平衡召回与精度 |
| `strict` | 仅置信度 ≥ 0.8 且 severity ≥ moderate | 误报敏感场景 |
| `loose` | 置信度 ≥ 0.3 即输出 | 漏检敏感场景（AIGC 审核） |

### scripts/vlm_prompt_ablation.py

| Mode | 说明 |
|------|------|
| `--manifest <path>` | 真实模式：加载基准集，调真实 VLM |
| `--mock --samples N` | 演示模式：合成样本 + Mock VLM，无需 ollama |
| `--out <path>` | 将 markdown 表写入文件 |

输出指标：TPR（检测率）/ FPR（误检率）/ Avg Findings / Avg Latency / N (pos/neg)。

---

## 5. QualityReport v1 Schema 扩展

| 字段 | 变更 |
|------|------|
| `agent_meta.agent_steps[].action` enum | 新增 `vlm_caption` |
| `agent_meta.quality_caption` | 新增可选属性，`dict \| null` |
| `decision_trace[].stage` | 新增 `vlm_caption` |
| `scenario_attribution` | 新增可选属性，`list[ScenarioAttribution] \| null` |
| `quality_caption` | 新增可选属性，`dict \| null` |

**向后兼容**：新字段均为可选；V1/V2 报告（无新字段）仍通过 schema。
