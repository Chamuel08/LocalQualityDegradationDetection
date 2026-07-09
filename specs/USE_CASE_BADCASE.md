# USE_CASE_BADCASE.md — 离线 Badcase 用例规范

> Offline Badcase Use Case Specification v1.1.0

---

## 1. 文档定位

定义 **Local Quality Degradation Detection** 的首批输入形态与工作流：

- 离线 badcase 单帧的输入边界
- 劣化 taxonomy 与检测器映射
- UI overlay 处理规则
- 质检 / 算法 badcase 工作流

| 文档 | 关系 |
|------|------|
| [`VERSION_ROADMAP.md`](VERSION_ROADMAP.md) | 版本边界与 Agent 层 |
| [`method_selection.md`](method_selection.md) | 子检测器算法选型 |
| [`001-v0-fast-mvp/spec.md`](001-v0-fast-mvp/spec.md) | v0.1 交付切片 |
| [`002-v1-agent-layer/spec.md`](002-v1-agent-layer/spec.md) | V1 Agent 层 |

---

## 2. 场景定义

### 2.1 什么是「badcase 帧」

**Badcase 帧**指被质检或人工认定为「画面存在局部质量问题」的单帧截图，通常来自录屏抽帧、手动截图或问题回放。

| 来源 | 说明 | V1 |
|------|------|-----|
| 录屏抽帧 | 从视频/VOD 按时间戳截取 | ✅ 单帧 |
| 手动截图 | 质检保存的问题帧 | ✅ 单帧 |
| 批量 session | 同一段素材的多帧目录 | ✅ 批量 CLI |
| 实时流接入 | RTMP/HLS 等在线流协议 | ❌ 不在 V1 范围 |

### 2.2 系统边界

**不含实时流处理**。上游负责解码与抽帧；本系统分析已保存的图像文件。

---

## 3. Badcase 输入元数据

```python
class BadcaseMetadata(TypedDict):
    frame_id: str
    session_id: str | None
    timestamp_ms: float | None
    platform: str | None               # 可选：来源平台标识
    source_pipeline: str | None        # 可选：生成/渲染方案标识
    resolution: tuple[int, int]
    codec: str | None
    bitrate_kbps: int | None
    frame_type: Literal["I", "P", "B", "unknown"] | None
    source_type: Literal["video_replay", "manual_screenshot", "issue_report"]
    has_overlay: bool
    overlay_types: list[str] | None    # subtitle, watermark, sticker, etc.
```

---

## 4. Badcase 工作流

### v0.1（GitHub MVP）

```
badcase/frames/ → detect.py --mode fast --legacy-fixed → JSON/HTML → 人工复核
```

- 无 VLM / LLM Judge；见 [`VERSION_ROADMAP.md`](VERSION_ROADMAP.md)

### V1（产品默认）

```
badcase/frames/ → detect.py --mode fast (Agent)
    → Round1: GlobalScan → SubDetectors
    → LLM Judge 全帧整合
    → [灰区 VLM Confirm] → [Round2 白名单补检]
    → JSON/HTML → 人工复核 → 标注回流
```

| 场景 | 模式 | V1 Agent 介入 |
|------|------|----------------|
| 批量筛图 | Fast（默认） | LLM Judge + 灰区 VLM Confirm |
| 单条深度归因 | Deep（可选） | VLM 粗分 + 子检测器量化 |

Deep Mode 失败自动 fallback Fast，报告标注 `mode_degraded`。详见 [`002-v1-agent-layer/spec.md`](002-v1-agent-layer/spec.md)。

---

## 5. 劣化 Taxonomy（与检测器映射）

| 现象 | 检测器 | 版本 |
|------|--------|------|
| 码率不足 block 噪点 | compression_artifact | v0.1 / V1 |
| 头发/面部纹理 blur | hair_texture / face_artifact | V1 |
| 边缘 spill / 绿边 | edge_bleed | v0.1 / V1 |
| UI overlay 干扰 | global_scan text_ui | V1 |
| 低码率缩放模糊 | hair_texture + 分辨率归一化 | V1 |
| 面部过曝 / 恐怖谷 | face_artifact | V1 |
| 手部异常 | hand_anomaly | V1 |
| 背景伪影 | background_artifact | V1 |
| 时序闪烁 | temporal_flicker | V2 |

---

## 6. Overlay 处理规则

字幕、水印、贴纸等 **UI overlay** 区域通过 `text_ui` 启发式或 `--ignore-regions` JSON 排除，不参与 face/background 劣化判定。

规则要点：

- `ignore_regions` 与 `text_ui` 检测结果取并集，形成 `overlay_mask`
- overlay 像素不参与 face/hair/background 的 anomaly_score 计算
- 大面积 overlay（如画面 ≥60%）可触发 fast_reject，MOS 上限收紧
- 部分遮挡（头发/手遮挡面部）时，对应 ROI 降级为 skip 而非硬判 anomaly

---

## 7. 分辨率与编码

常见 720p/1080p；低码率 H.264 block 由 `compression_artifact` 覆盖。ROI 指标计算前可按基准分辨率归一化。

---

## 8. V1 vs V2

| 能力 | V1 | V2 |
|------|----|----|
| 单帧 badcase | ✅ | ✅ |
| 视频 clip / 时序 | ❌ | ✅ |

---

## 9. 验收（场景级）

- [ ] 单帧 CLI 输出合法 JSON
- [ ] overlay 区域不误报
- [ ] block / edge 类 sample 可检出
