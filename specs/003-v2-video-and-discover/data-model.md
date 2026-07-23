# Data Model: V2 视频与主动发现

**Feature**: `003-v2-video-and-discover`

**Depends on**: [`002-v1-agent-layer/data-model.md`](../002-v1-agent-layer/data-model.md) — `AgentContext` / `AgentAction` / `AgentMeta` / `QualityReport`

> 本文档定义 V2 新增实体。所有实体均**不破坏** V1 报告 schema（新字段为可选）。

---

## 1. vlm_discover 工具链

### VLMDiscoverFinding

`vlm_discover` 工具单条发现结果。VLM 对全帧主动扫描后，将无法被 CV 规则量化的异常以此结构返回。**不包含** `region_mask` / `bbox`（VLM 无像素级定位能力），仅包含语义描述和估计影响。

| Field | Type | Required | Validation |
|-------|------|----------|------------|
| `degradation_type` | `str` | ✅ | e.g. `hand_extra_finger` / `face_generation_error` / `background_overlap` |
| `region_description` | `str` | ✅ | 自然语言描述异常区域（如"左下角人物左手出现 6 根手指"） |
| `severity` | `str` | ✅ | `minor` \| `moderate` \| `critical` |
| `confidence` | `float` | ✅ | VLM 自报告置信度 [0.0, 1.0] |
| `reasoning` | `str` | ✅ | VLM 详细推理（中文） |
| `mos_impact_estimate` | `float` | ✅ | ≤ 0；**仅参考，不参与最终 MOS 计算** |

```python
@dataclass
class VLMDiscoverFinding:
    degradation_type: str
    region_description: str
    severity: str
    confidence: float
    reasoning: str
    mos_impact_estimate: float
```

### VLM_DISCOVER_PROMPT（提示词契约）

`vlm_discover` 调用 VLM 时使用的 system/user prompt（定义于 `src/lqdd/agent/prompts.py`）。要求 VLM 输出严格 JSON：

```json
{
  "findings": [
    {
      "degradation_type": "hand_extra_finger",
      "region_description": "左下角人物左手出现6根手指",
      "severity": "moderate",
      "confidence": 0.82,
      "reasoning": "清晰可见手部有多余手指",
      "mos_impact_estimate": -0.4
    }
  ],
  "overall_assessment": "对全图整体质量的一句话总结"
}
```

- `severity` 仅允许 `minor` / `moderate` / `critical`
- `confidence` 范围 [0.0, 1.0]
- `mos_impact_estimate` 仅作参考记录，不参与最终 MOS 计算（MOS 由 `compute_mos()` 统一负责）
- 无问题时 `findings` 输出空数组

### AgentAction 白名单扩展

V1 白名单 `{vlm_analyze, rerun_detector, dispatch_compression, accept}` 扩展为：

```python
action: Literal["vlm_analyze", "rerun_detector", "dispatch_compression", "vlm_discover", "accept"]
```

| action | Parameters | Constraints |
|--------|------------|-------------|
| `vlm_discover` | `reason?` | 对全帧主动扫描；每帧最多 1 次；结果写入 `agent_meta.vlm_discover_findings`（不进 `degradations[]`） |

### AgentContext / AgentMeta 扩展

| Field | Type | Notes |
|-------|------|-------|
| `AgentContext.vlm_discover_findings` | `list[VLMDiscoverFinding]` | 主动发现结果（不进 `merged_degradations`，单独记录） |
| `AgentMeta.vlm_discover_findings` | `list[dict]` | 序列化到报告；未触发时为 `[]` |

### TraceEntry 扩展

`decision_trace` 新增 stage：

| stage | module | decision |
|-------|--------|----------|
| `vlm_discover` | `VLMDiscover` | `vlm_discover_complete` / `vlm_discover_failed` |

---

## 2. VideoClipRunner 聚合层

### VideoClipReport

`VideoClipRunner.run()` 的聚合输出。

| Field | Type | Notes |
|-------|------|-------|
| `clip_id` | `str` | clip 标识符（写入每帧报告的 `video_id`） |
| `frame_count` | `int` | 处理帧数 |
| `frame_reports` | `list[QualityReport]` | 逐帧单帧报告（V1 结构） |
| `flicker_result` | `TemporalFlickerResult` | 帧间闪烁检测结果 |
| `aggregate_mos` | `float \| None` | 逐帧 `overall_mos` 的均值；全部不可用时为 None |
| `worst_frame_mos` | `float \| None` | 最差帧 MOS；无可用 MOS 时为 None |
| `worst_frame_index` | `int` | 最差帧在 `frame_reports` 中的索引；无可用 MOS 时为 -1 |
| `degradation_summary` | `dict[str, int]` | 各 `degradation_type` 跨帧出现次数 |

```python
@dataclass
class VideoClipReport:
    clip_id: str
    frame_count: int
    frame_reports: list[QualityReport]
    flicker_result: TemporalFlickerResult
    aggregate_mos: float
    worst_frame_mos: float
    worst_frame_index: int
    degradation_summary: dict
```

### VideoClipRunner

外层包装器，不修改单帧 pipeline 内部接口。

| Method | Signature | Notes |
|--------|-----------|-------|
| `__init__` | `(pipeline, flicker_luma_threshold=8.0, flicker_hue_threshold=6.0, flicker_min_ratio=0.15)` | `pipeline` 需实现 `.run(SingleFrameInput) -> QualityReport` |
| `run` | `(frames: list[np.ndarray], clip_id="clip", frame_ids=None) -> VideoClipReport` | 逐帧调 `pipeline.run`；帧间调 `detect_temporal_flicker`；聚合 MOS 与汇总 |

### sample_frames_from_video

| Signature | Notes |
|-----------|-------|
| `(video_path, max_frames=8, start_sec=0.0, end_sec=None) -> list[np.ndarray]` | 均匀抽取 ≤ `max_frames` 帧；返回 BGR uint8 列表；文件不存在抛 `FileNotFoundError`，无法打开抛 `RuntimeError` |

---

## 3. TemporalFlicker 检测器

### FlickerSegment

一段闪烁区间的描述。

| Field | Type | Notes |
|-------|------|-------|
| `start_frame` | `int` | 起始帧索引 |
| `end_frame` | `int` | 结束帧索引（= start + 1） |
| `max_delta` | `float` | 该段最大跳变值 |
| `metric` | `str` | `luma_delta` \| `hue_delta` |
| `severity` | `str` | `minor` \| `moderate` \| `critical` |

### TemporalFlickerResult

对一组连续帧的时域闪烁检测结果。

| Field | Type | Notes |
|-------|------|-------|
| `frame_count` | `int` | 输入帧数 |
| `flicker_segments` | `list[FlickerSegment]` | 检出的闪烁区间 |
| `mean_luma_delta` | `float` | 相邻帧灰度均值差的均值 |
| `max_luma_delta` | `float` | 最大灰度均值差 |
| `flicker_ratio` | `float` | 发生闪烁的帧间比例 [0, 1] |
| `is_flickering` | `bool` | `flicker_ratio >= min_flicker_ratio` |
| `method` | `str` | `temporal_luma_hue_delta` |

### detect_temporal_flicker

| Signature | Notes |
|-----------|-------|
| `(frames, luma_delta_threshold=8.0, hue_delta_threshold=6.0, min_flicker_ratio=0.15) -> TemporalFlickerResult` | 至少 2 帧；<2 帧返回空结果 |

### 算法

1. 对每帧计算灰度均值（`cv2.COLOR_BGR2GRAY`）与 HSV-H 通道均值（`cv2.COLOR_BGR2HSV`）
2. 相邻帧灰度均值差 = `|luma_means[i+1] - luma_means[i]|`；色相同理
3. 灰度差 ≥ `luma_delta_threshold` → 记 `luma_delta` 段；否则色相差 ≥ `hue_delta_threshold` → 记 `hue_delta` 段
4. 严重度：按跳变值相对阈值的倍数分级（≥ 2.5× → critical，≥ 1.5× → moderate，否则 minor）
5. `flicker_ratio` = 闪烁帧间数 / (n-1)；`is_flickering` = `flicker_ratio >= min_flicker_ratio`

### 包结构约束

- `src/lqdd/temporal_flicker/detector.py` — 实现
- **不**在 `src/lqdd/detectors/registry.py` 的 `ALL_DETECTOR_NAMES` / `build_detector_registry` 中注册（单帧 pipeline 无法调用）
- 仅由 `VideoClipRunner` 在帧间聚合层 import 调用

---

## 4. QualityReport v1 Schema 扩展

V2 不新建独立报告 schema，而是**扩展** v1 schema（`002-v1-agent-layer/contracts/quality-report.v1.schema.json`）：

| 字段 | 变更 |
|------|------|
| `agent_meta.agent_steps[].action` enum | 新增 `vlm_discover` |
| `agent_meta.vlm_discover_findings` | 新增可选属性，`list[VLMDiscoverFinding]`，未触发时为 `[]` |
| `decision_trace[].stage` | 新增 `vlm_discover`（enum 已含 `agent_step`） |

**向后兼容**：新字段均为可选；V1 报告（无 `vlm_discover_findings`）仍通过 v1 schema。
