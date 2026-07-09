# LocalQualityDegradationDetection

**可解释的局部画质劣化检测（Local Quality Degradation Detection）**

*Explainable local quality degradation detection for offline badcase frames*

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Status](https://img.shields.io/badge/code-v0.1%20MVP-green)]()
[![Scene](https://img.shields.io/badge/scene-badcase-orange)](specs/USE_CASE_BADCASE.md)

> 输入：离线 badcase 单帧（录屏抽帧 / 手动截图）  
> 输出：带 **bbox + 数值 Evidence + MOS 影响 + Decision Trace** 的 JSON / HTML 报告

---

## Demo

| 输入帧 | 检测报告（bbox + Evidence） |
|--------|---------------------------|
| *待补充：`examples/demo_input.png`* | *待补充：`examples/demo_report.png`* |

```bash
python detect.py --image data/sample/frames/edge/edge_01.png --mode fast
python detect.py --image data/sample/frames/edge/edge_01.png --mode fast --output report.html
```

> **当前状态**：v0.1 Fast Mode MVP 已实现（固定 pipeline + CLI + 样例数据）；V1 Agent 层已实现（`--mode fast` 默认）。

---

## 为什么做（Problem）

局部画质问题**高频、需定位、需归因**，全图 IQA 不够用：

| 痛点 | 现有方案不足 | 本项目 |
|------|-------------|--------|
| badcase 难批量分析 | BRISQUE/NIQE 无定位 | **局部**检测 + bbox |
| 糊 / 绿边 / block 混在一起 | 黑箱总分 | **分类型**检测器 + Evidence |
| block vs 生成 blur | 无法区分成因 | CompressionArtifact + root_cause |
| UI overlay | 易误报 | text_ui mask + ignore_regions |
| 质检复核 | 只要分数不够 | HTML + 中文 detail |

**边界**：**离线单帧**，不接 RTMP 实时流。

---

## 做了什么（Solution）

Coarse-to-Fine：Global Scan → 按需子检测器 → Report（JSON/HTML）。

| 版本 | 形态 |
|------|------|
| **v0.1 MVP** | 固定 Pipeline：GlobalScan → EdgeBleed + Compression，`--legacy-fixed` 回退 |
| **V1**（默认） | **Agent 编排**：小模型先行 + **VLM 灰区兜底** + **LLM Judge** 整合与 Round 2 |

详见 [`specs/VERSION_ROADMAP.md`](specs/VERSION_ROADMAP.md)。

---

## 方法亮点

见 [`specs/method_selection.md`](specs/method_selection.md)：FFT 头发纹理、ArcFace 面部、Lab ΔE 边缘、DCT blockiness 等。

---

## 实现进度

| 模块 | Spec | Code |
|------|:----:|:----:|
| Badcase 用例 | ✅ | — |
| **v0.1** GlobalScan + EdgeBleed + Compression | ✅ | ✅ |
| **v0.1** CLI `detect.py`（`--legacy-fixed` 回退） | ✅ | ✅ |
| **V1** Agent + VLM 灰区 + LLM Judge | ✅ | ✅ |
| **V1** `--mode fast` Agent 默认路径 | ✅ | ✅ |
| 全量子检测器（face / hair / hand 等） | — | 📋 |

Spec Kit：`001-v0-fast-mvp`（v0.1）· `002-v1-agent-layer`（V1 Agent）

---

## 版本路线图

| 版本 | 内容 | 文档 |
|------|------|------|
| v0.1 | 固定 pipeline，2 检测器，零外部依赖 | [`001-v0-fast-mvp/spec.md`](specs/001-v0-fast-mvp/spec.md) |
| V1 | Agent + VLM 兜底 + LLM Judge | [`002-v1-agent-layer/spec.md`](specs/002-v1-agent-layer/spec.md) |
| V2 | 视频 + 时序闪烁检测 | [`VERSION_ROADMAP.md`](specs/VERSION_ROADMAP.md) |

---

## 快速开始

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp config.example.yaml config.yaml
python scripts/generate_synthetic_samples.py
python detect.py --image data/sample/frames/edge/edge_01.png --mode fast
python detect.py --image data/sample/frames/edge/edge_01.png --mode fast --legacy-fixed
pytest tests/ -m "not vlm" -q
```

可选：`pip install -e ".[full]"` 安装 MediaPipe（Python 3.10–3.12 推荐）；未安装时自动使用 GrabCut 回退分割。

---

## 文档（公开集）

| 文档 | 内容 |
|------|------|
| [`specs/VERSION_ROADMAP.md`](specs/VERSION_ROADMAP.md) | v0.1 / V1 / V2 边界与 Agent 层 |
| [`specs/USE_CASE_BADCASE.md`](specs/USE_CASE_BADCASE.md) | Badcase 工作流与 overlay 规则 |
| [`specs/method_selection.md`](specs/method_selection.md) | 子检测器算法选型 |
| [`specs/001-v0-fast-mvp/`](specs/001-v0-fast-mvp/) | v0.1 feature spec、plan、contracts |
| [`specs/002-v1-agent-layer/`](specs/002-v1-agent-layer/) | V1 Agent feature spec、plan、contracts |

---

## License

Licensed under the [Apache License, Version 2.0](LICENSE). See [NOTICE](NOTICE) for third-party dependency attributions.
