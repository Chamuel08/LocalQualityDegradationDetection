# LocalQualityDegradationDetection

**可解释的局部画质劣化检测（Local Quality Degradation Detection）**

Reference-free, explainable local quality degradation detection for offline badcase frames.

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

输入离线单帧，输出不规则 **region mask**、数值 **Evidence**、**MOS 影响**与 **Decision Trace**（JSON / HTML）。无参考图 diff，面向 badcase 定位与归因。

---

## Architecture

```
GlobalScan → 9 detectors → [Agent: nomination routing] → [VLM grey-zone] → [Judge] → Report
```

| Detector | Target | Method |
|----------|--------|--------|
| `edge_bleed` | 轮廓绿边 | 绿通道偏置 + ΔE |
| `compression_artifact` | 全图/ROI 压缩 | DCT 8×8 块边界 + Laplacian 纹理损失 |
| `blur_artifact` | 主体模糊 | 前景 ROI Laplacian |
| `mosaic_artifact` | 马赛克 | 下采样-上采样块一致性 |
| `banding_artifact` | 背景色带 | 梯度量化台阶 |
| `background_artifact` | 背景劣化 | 背景块效应 / 色彩漂移 |
| `hair_texture` | 发丝 | FFT 高频能量比 |
| `face_artifact` | 面部 | 过曝 + Laplacian（可扩展 ArcFace） |
| `hand_anomaly` | 手部 | MediaPipe Hands 几何（可选 `[mediapipe]`） |

**v0.1**（`--legacy-fixed`）：固定 Fast Pipeline，全量检测器。  
**V1**（默认）：GlobalScan 提名 → 路由 → 灰区 VLM Confirm → Judge Round 2。

报告含 `region_mask_rle`（RLE 像素 mask）、`bbox`、`evidence`、`decision_trace`。

---

## Quick Start

**Requirements:** Python 3.10+

```bash
git clone https://github.com/Chamuel08/LocalQualityDegradationDetection.git
cd LocalQualityDegradationDetection

python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp config.example.yaml config.yaml

# 内置合成样例位于 data/sample/；如需重新生成：
python scripts/generate_synthetic_samples.py

python detect.py \
  --image data/sample/frames/edge/edge_01.png \
  --mode fast --legacy-fixed \
  --output report.html
```

可选 V1 Agent（需 [Ollama](https://ollama.com/) + 视觉模型）：

```bash
ollama pull qwen2.5-vl:7b
python detect.py --image data/sample/frames/edge/edge_01.png --mode fast --output report.html
```

无 Ollama 时 VLM/Judge 优雅降级，`decision_trace` 记录 skip 原因。

---

## Tests

```bash
pytest tests/ -m "not vlm" -q
```

---

## Benchmark

自备 GT manifest 时可批量评测（结果默认 gitignore，不随仓库分发）：

```bash
python benchmark/run_eval.py --manifest /path/to/manifest.json
```

见 [`benchmark/README.md`](benchmark/README.md)。

---

## Specs

| Document | Description |
|----------|-------------|
| [`specs/USE_CASE_BADCASE.md`](specs/USE_CASE_BADCASE.md) | Badcase 场景与边界 |
| [`specs/VERSION_ROADMAP.md`](specs/VERSION_ROADMAP.md) | v0.1 / V1 / V2 路线 |
| [`specs/001-v0-fast-mvp/`](specs/001-v0-fast-mvp/) | v0.1 契约与 schema |
| [`specs/002-v1-agent-layer/`](specs/002-v1-agent-layer/) | V1 Agent / VLM 契约 |

---

## License

[Apache License 2.0](LICENSE)
