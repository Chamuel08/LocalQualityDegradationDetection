# LocalQualityDegradationDetection

**无参考、可解释的局部画质劣化检测**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)

输入离线单帧图像，输出不规则 **region mask**、数值 **Evidence**、**MOS 影响估计**与完整 **Decision Trace**（JSON / HTML）。无需参考图或 diff，适用于 badcase 定位与归因、离线质检与后处理排查等场景。

---

## 特性

- **九个专用检测器**：覆盖压缩、模糊、马赛克、色带、绿幕溢色、发丝、面部、背景等；`hand_anomaly` 为**实验性 MVP**（见下表）
- **可解释输出**：RLE 像素 mask、bbox、各检测器 evidence、结构化 decision trace
- **两种流水线**：v0.1 确定性快速路径；V1 Agent 层（提名路由 + VLM 灰区确认 + LLM Judge）
- **优雅降级**：Ollama 不可用时，VLM / Judge 步骤自动跳过并在 trace 中记录原因
- **可选评测工具链**：本地合成带 pixel GT 的劣化样本并批量评测（数据不随仓库分发）

---

## 架构

```
GlobalScan → 9 detectors → [Agent: nomination routing] → [VLM grey-zone] → [Judge] → Report
```

| 检测器 | 目标 | 方法 | 成熟度 |
|--------|------|------|--------|
| `edge_bleed` | 边缘 / 溢色 | 绿通道偏置 + ΔE | 主力 |
| `compression_artifact` | 压缩块效应 | DCT 8×8 边界 + Laplacian 纹理损失 | 主力 |
| `blur_artifact` | 主体模糊 | 前景 ROI Laplacian | 主力 |
| `mosaic_artifact` | 马赛克 / 像素化 | 下采样–上采样块一致性 | 主力 |
| `banding_artifact` | 色带 | 梯度量化台阶 | 主力 |
| `background_artifact` | 背景劣化 | 背景块效应 / 色彩漂移 | 主力 |
| `hair_texture` | 发丝细节损失 | FFT 高频能量比 | 主力 |
| `face_artifact` | 面部过曝 / 模糊 | 亮度 + Laplacian（可扩展 ArcFace） | 主力 |
| `hand_anomaly` | 手部几何异常 | 启发式 ROI + 边缘密度；**可选** mediapipe 增强关键点几何 | **实验性 MVP** |

> **`hand_anomaly` 说明**：检测器已接入流水线，但 ROI 为前景 bbox 下半段的启发式区域（非真实手部分割）；未装 `mediapipe` 时走边缘密度 fallback，**仍会运行**但能力有限。spec 中的多指计数、粘连、手部模糊等**尚未实现**；benchmark **无** hand 类 GT。详见 [`specs/hand_anomaly.spec.md`](specs/hand_anomaly.spec.md)。

**v0.1**（`--legacy-fixed`）：固定 Fast Pipeline，全量检测器确定性运行，无 Agent / VLM / Judge。  
**V1**（**默认**：`config.yaml` 中 `agent.enabled: true` 且不加 `--legacy-fixed`）：完整 Agent 编排流水线，见下文 [V1 Agent 流水线](#v1-agent-流水线)。

报告字段包括 `region_mask_rle`、`bbox`、`evidence`、`decision_trace`；V1 额外包含 `agent_meta`、`vlm_reasoning`、`vlm_reasoning_summary`。

---

## 环境要求与安装

### 环境要求

| 项目 | 要求 | 说明 |
|------|------|------|
| **Python** | **3.10 ~ 3.12**（推荐 3.11） | `requires-python = ">=3.10"`，见 `pyproject.toml` |
| **操作系统** | macOS / Linux / Windows | 无特殊限制 |
| **磁盘** | ≥ 2 GB（不含模型） | 代码 + venv + 依赖 |
| **Ollama（V1 推荐）** | 另需 **~6 GB+** | VLM `qwen2.5vl:7b` 约 4.7 GB；Judge `qwen2.5:3b` 约 2 GB |
| **内存（V1 + VLM）** | 建议 **16 GB+** | 7B 视觉模型本地推理时占用较高 |

- **仅跑 v0.1**（`--legacy-fixed`）：只需 Python 环境，**不需要 Ollama**
- **跑 V1 Agent**（默认）：需要 Python + **Ollama 已安装、已拉模型、服务可访问**

### Python 依赖

依赖清单以 [`requirements.txt`](requirements.txt) 为准（与 [`pyproject.toml`](pyproject.toml) 同步）：

| 文件 | 内容 |
|------|------|
| [`requirements.txt`](requirements.txt) | 运行时依赖（OpenCV、NumPy、YAML、JSON Schema、httpx） |
| [`requirements-dev.txt`](requirements-dev.txt) | 开发环境：`-r requirements.txt` + pytest + Pillow + 可编辑安装 |
| [`requirements-optional.txt`](requirements-optional.txt) | **可选增强**：`mediapipe`（仅提升实验性 `hand_anomaly`，非必需） |

| 包 | 版本要求 | 用途 |
|----|----------|------|
| `opencv-python-headless` | ≥ 4.8 | 图像读写与 CV 算子 |
| `numpy` | ≥ 1.24 | 数值计算 |
| `PyYAML` | ≥ 6.0 | 读取 `config.yaml` |
| `jsonschema` | ≥ 4.20 | 报告 schema 校验 |
| `httpx` | ≥ 0.27 | 调用 Ollama HTTP API（VLM / Judge） |
| `pytest` | ≥ 7.4 | 测试（仅 dev） |
| `Pillow` | ≥ 10.0 | 测试与 Demo（仅 dev） |
| `mediapipe` | ≥ 0.10 | **可选**：为实验性 `hand_anomaly` 提供 MediaPipe 关键点路径（不装也能跑 fallback） |

### 1. 安装 Python 项目

```bash
git clone https://github.com/Chamuel08/LocalQualityDegradationDetection.git
cd LocalQualityDegradationDetection

# 创建虚拟环境（推荐）
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 升级 pip
pip install -U pip setuptools wheel

# 开发环境（推荐：依赖 + 可编辑安装 + pytest）
pip install -r requirements-dev.txt

# 仅运行时依赖 + 手动安装包：
# pip install -r requirements.txt
# pip install -e .

# 可选：增强实验性 hand_anomaly（非必需；不装 mediapipe 时该检测器仍可用 fallback）
# pip install -r requirements-optional.txt

# 复制配置模板（gitignore，本地修改）
cp config.example.yaml config.yaml
```

验证 Python 包是否可用：

```bash
python -c "import cv2, numpy, yaml, httpx; print('ok', cv2.__version__)"
python detect.py --help
pip check
```

### 2. 安装 Ollama（V1 Agent 需要）

V1 的 VLM 灰区确认与 LLM Judge 通过 **本地 Ollama** 调用，不依赖云端 API。

#### 2.1 安装 Ollama 本体

按官方文档安装：[https://ollama.com/download](https://ollama.com/download)

**macOS**

```bash
# 方式 A：官网下载 .dmg 安装（推荐）
# 方式 B：Homebrew
brew install ollama
```

**Linux**

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

**Windows**

从官网下载安装包安装；安装后 Ollama 通常作为系统服务在后台运行。

#### 2.2 启动 Ollama 服务

macOS / Linux 若未自动启动，可手动：

```bash
ollama serve
```

默认监听 **`http://localhost:11434`**（与 `config.yaml` 中 `vlm.host` / `judge.host` 一致）。

验证服务是否正常：

```bash
curl http://localhost:11434/api/tags
# 或
ollama list
```

#### 2.3 拉取模型

模型名称须与 `config.yaml` 一致（见 `config.example.yaml` 默认值）：

```bash
# VLM：灰区 ROI 二次确认（约 4.7 GB）
ollama pull qwen2.5vl:7b

# Judge：全帧结论审查（约 2 GB）
ollama pull qwen2.5:3b
```

确认已下载：

```bash
ollama list
# 应能看到 qwen2.5vl:7b 与 qwen2.5:3b
```

快速冒烟（可选，确认模型能推理）：

```bash
ollama run qwen2.5:3b "reply ok"
```

#### 2.4 与 LQDD 的对接方式

LQDD 通过 `httpx` 请求 Ollama REST API，**不**需要额外安装 Python 的 `ollama` SDK。

`config.yaml` 关键项（已从 `config.example.yaml` 复制后一般无需改）：

```yaml
vlm:
  provider: ollama
  model: qwen2.5vl:7b
  host: http://localhost:11434
  timeout_ms: 120000        # VLM 首次加载可能较慢，建议 ≥ 120s

judge:
  provider: ollama
  model: qwen2.5:3b
  host: http://localhost:11434
  timeout_ms: 45000
```

环境变量覆盖（可选）：

| 变量 | 作用 |
|------|------|
| `OLLAMA_HOST` | 覆盖 VLM / Judge 的 Ollama 地址（如 `http://127.0.0.1:11434`） |
| `LQDD_VLM_MODEL` | 覆盖 VLM 模型名 |
| `LQDD_JUDGE_MODEL` | 覆盖 Judge 模型名 |
| `LQDD_AGENT_ENABLED=0` | 关闭 Agent，等效 v0.1 |

**代理注意**：若系统设置了 `HTTP_PROXY` 且指向不可用代理，可能导致访问 `localhost` 失败。本项目已对 Ollama 客户端设置 `trust_env=False`；若仍异常，可临时 `unset HTTP_PROXY HTTPS_PROXY` 后再跑。

#### 2.5 Ollama 常见问题

| 现象 | 处理 |
|------|------|
| `vlm_skipped: service_unavailable` | 确认 `ollama serve` 在跑，`curl localhost:11434/api/tags` 有响应 |
| 首次 VLM 很慢 / 超时 | 调大 `vlm.timeout_ms`（如 `180000`）；确保模型已 `ollama pull` 完成 |
| `model not found` | 模型名与 `ollama list` 不一致时，改 `config.yaml` 或重新 `ollama pull` |
| 内存不足 | 换更小 Judge 模型，或仅跑 v0.1（`--legacy-fixed`） |

### 3. 验证完整环境

```bash
# 1) Python 依赖
pytest tests/ -m "not vlm" -q

# 2) Ollama（V1）
ollama list | grep -E 'qwen2.5vl|qwen2.5'

# 3) v0.1 冒烟（无需 Ollama）
python detect.py --image /path/to/frame.png --legacy-fixed --output /tmp/report_v01.json

# 4) V1 冒烟（需 Ollama）
python detect.py --image /path/to/frame.png --config config.yaml --output /tmp/report_v1.json
python -c "import json; d=json.load(open('/tmp/report_v1.json')); print('vlm_calls', d.get('agent_meta',{}).get('vlm_calls'))"
```

`agent_meta.vlm_calls > 0` 表示 VLM 已被调用（需该帧存在灰区检出且 Ollama 正常）。

---

## 快速开始

完成 [环境要求与安装](#环境要求与安装) 后，**默认即 V1 Agent 流水线**（`agent.enabled: true`）。下面分两条路径说明。

### V1 完整流程（推荐）

前提：Ollama 已安装、已拉取模型、服务可访问（步骤见上文 [环境要求与安装 → 安装 Ollama](#环境要求与安装)）。

```bash
source .venv/bin/activate

python detect.py \
  --image /path/to/frame.png \
  --config config.yaml \
  --output report_v1.html
```

- **不要**加 `--legacy-fixed`
- 输出 HTML 含 bbox 可视化；改 `--output report_v1.json` 可得 JSON
- Ollama 未运行时流水线仍会完成，但 VLM / Judge 会 skip（原因写入 `decision_trace`）

验证 Agent 是否真正跑通，见 [如何确认 V1 已生效](#如何确认-v1-已生效)。

### v0.1 基线（无 Agent）

**无需 Ollama**，适用于 CI、阈值调参、与 V1 对比：

```bash
python detect.py \
  --image /path/to/frame.png \
  --legacy-fixed \
  --output report_v01.html
```

- `--legacy-fixed` — 跳过 Agent / VLM / Judge，仅规则检测器
- `--output report.html` — HTML 报告；`-` 表示 JSON 打印到 stdout

批量模式（v0.1）：

```bash
python detect.py \
  --image-dir /path/to/frames/ \
  --legacy-fixed \
  --output-dir reports/
```

---

## V1 Agent 流水线

V1 在规则检测器之上增加 **Agent 编排层**：对「检出了但置信度拿不准」的结果，用本地 VLM 做 ROI 二次确认，再由 LLM Judge 审查全帧结论，必要时进入 Round 2 补检。

> **定位说明**：当前 VLM 是 **Confirm（二次确认）**——仅对检测器已产出、且 confidence 落在灰区的条目发起视觉确认；**不是** Discover（从零主动发现规则检不出的劣化）。

### 阶段一览

| 阶段 | 模块 | 作用 |
|------|------|------|
| 1. GlobalScan | `GlobalScanner` | 全帧扫描，产出区域提名（nominations） |
| 2. Routing | `FastRouter` | 按提名置信度决定 dispatch / skip；低置信提名不跑对应检测器 |
| 3. Detection | 9 个 SubDetectors | 规则检测，产出 preliminary degradations + confidence |
| 4. VLM Confirm | `VLMConfirm` | 对**灰区检出**裁剪 ROI，调用 Ollama VLM 二次确认 |
| 5. Fusion | `fuse_all` | 融合检测器置信度与 VLM 结论，写入 `vlm_reasoning` |
| 6. Judge | `LLMJudge` | 审查全帧 degradations + MOS 是否自洽，决定是否 Round 2 |
| 7. Round 2（可选） | `Round2Executor` | 按 Judge 白名单 action 补检或重跑（如 `dispatch_compression`） |
| 8. Report | `ReportGenerator` | 聚合 MOS、mask、bbox、完整 `decision_trace` |

### 置信度分级与 VLM 触发条件

检测器产出每条 degradation 后，按 `confidence` 分三档（阈值见 `config.yaml` → `agent`）：

| 档位 | 默认范围 | 行为 |
|------|----------|------|
| **高置信** | ≥ `high_confidence_threshold`（0.7） | 直接采纳检测器结果，**不调 VLM** |
| **灰区** | `[grey_zone_lower, grey_zone_upper)`（0.4～0.7） | 裁剪 bbox ROI → **调用 VLM Confirm** |
| **低置信** | < `grey_zone_lower`（0.4） | **不调 VLM**，结果原样进入后续 Judge |

单帧 VLM 调用上限由 `vlm.max_calls_per_frame` 控制（默认 3）；超出后 trace 记录 `vlm_skipped: quota_exceeded`。

VLM 与检测器结论融合策略（`vlm/fuse.py`）：

- **agree** — 加权融合置信度（检测器 40% + VLM 60%）
- **vlm_override** — VLM 强置信（>0.8）时以 VLM 为准
- **detector_override** — 检测器强置信（>0.8）时保留检测器
- **uncertain** — 取较低置信度，MOS 影响保守下调

### Judge 与 Round 2

Judge 输入为融合后的全帧 degradations 与 MOS，输出 `assessment`（如 `consistent` / `uncertain` / `inconsistent`）及可选 `actions`。

当 `needs_round2: true` 且未达 `agent.max_rounds` 时，执行 Round 2 白名单动作（如 `dispatch_compression`、`rerun_detector`、`accept`），trace 中出现 `round2_complete`。

### 如何确认 V1 已生效

打开 JSON 报告（或 HTML 内嵌 JSON），检查以下字段：

**1. `decision_trace` 阶段链**

典型 V1 链路应包含：

```text
mode_select → global_scan → routing → detection
  → vlm_confirm（decision: vlm_confirmed 或 vlm_skipped:*）
  → judge（decision: judge_consistent / judge_uncertain / …）
  → [round2_complete]（若 Judge 触发 Round 2）
  → aggregation
```

**2. `agent_meta`**

```json
{
  "rounds_executed": 1,
  "vlm_calls": 1,
  "judge_assessment": "uncertain"
}
```

- `vlm_calls > 0` 表示 VLM 实际被调用（Ollama 正常且存在灰区检出）
- `vlm_calls: 0` 可能是：全为高/低置信、Ollama 不可用、或 `--legacy-fixed`

**3. 单条 degradation 的 `vlm_reasoning`**

灰区条目经 VLM 确认后会有：

```json
{
  "reasoning": "…",
  "vlm_confidence": 0.85,
  "fusion_decision": "agree"
}
```

**4. 常见 skip 原因（Ollama 不可用时）**

| trace decision | 含义 |
|----------------|------|
| `vlm_skipped: service_unavailable` | Ollama 未启动或请求失败 |
| `vlm_skipped: quota_exceeded` | 单帧 VLM 调用达上限 |

### V1 与 v0.1 对比示例

同一张图分别跑两条路径，对比 Agent 层贡献：

```bash
# v0.1 基线
python detect.py --image /path/to/frame.png --legacy-fixed --output report_v01.json

# V1 完整（需 Ollama）
python detect.py --image /path/to/frame.png --config config.yaml --output report_v1.json
```

关注差异：`confidence` 是否经 VLM 融合、`decision_trace` 是否多出 vlm/judge/round2 阶段、`vlm_reasoning_summary` 是否有内容。

---

## Demo 报告

以下所有素材均为本项目独立生成（AI 合成人像 + 程序化劣化），不依赖任何第三方数据集。

### 源图与劣化样本

干净源图（AI 生成）及其 7 种程序化劣化版本，均存放在 `docs/demo/`：

![干净合成人像](docs/demo/synthetic_portrait.png)

### 多 case 检测结果一览

对同一张干净源图分别施加 7 种劣化，用 v0.1 `--legacy-fixed` 检测，结果如下：

| 劣化样本 | 施加的劣化 | 目标检测器 | 是否检出 | 置信度 | 严重度 |
|----------|-----------|-----------|---------|--------|--------|
| `case_compression.png` | 边缘高频压缩 | `compression_artifact` | ✅ | 0.867 | moderate |
| `case_block.png` | JPEG 块效应 | `compression_artifact` | ✅ | 0.920 | severe |
| `case_blur.png` | 高斯模糊 | `blur_artifact` | ✅ | 0.740 | moderate |
| `case_mosaic.png` | 下采样像素化 | `mosaic_artifact` | ✅ | 0.759 | moderate |
| `case_overexposure.png` | 亮度过曝 | `face_artifact` | ✅ | 0.720 | minor |
| `case_banding.png` | 色深降低 | `banding_artifact` | ❌ | — | — |
| `case_green_spill.png` | 边缘绿溢色 | `edge_bleed` | ✅ | 0.797 | critical |

> `banding` 检测器当前阈值偏保守，对合成色带的检出率较低（已知限制，见 [项目状态](#项目状态)）。其余 6 种劣化均被对应检测器成功检出。

### V1 Agent 完整示例（`case_blur`）

选取 `case_blur`（区域模糊）展示 V1 Agent 完整流程，因为该样本中 `face_artifact` 原始置信度 0.68 落在灰区 [0.4, 0.7)，**触发了 VLM 二次确认**。

#### 1. 报告总览

```json
{
  "system_version": "1.0.0",
  "overall_mos": 3.673,
  "severity": "moderate",
  "agent_meta": {
    "rounds_executed": 2,
    "max_rounds_reached": true,
    "vlm_calls": 1,
    "judge_assessment": "uncertain"
  }
}
```

`vlm_calls: 1` 表示有 1 条灰区检出经 VLM 确认。

#### 2. VLM 确认的 degradation — 三层可解释结构

`face_artifact` 原始置信度 0.68（灰区），经 VLM 确认后融合置信度提升至 0.782：

```json
{
  "detector": "face_artifact",
  "degradation_type": "face_blur",
  "severity": "minor",
  "confidence": 0.782,
  "bbox": [315, 255, 393, 522],
  "evidence": {
    "method": "face_roi_exposure_laplacian",
    "metric": "face_laplacian_var",
    "value": 10.96,
    "threshold": 85.0,
    "detail": "面部 Laplacian 方差 11 ≤ 85（偏糊）"
  },
  "root_cause_hypothesis": {
    "cause": "generation_artifact",
    "confidence": 0.45
  },
  "vlm_reasoning": {
    "reasoning": "面部区域存在明显的模糊现象，细节损失严重，导致面部特征不清晰。",
    "vlm_confidence": 0.85,
    "ux_impact": "面部模糊使得观看者难以辨认人物身份和表情，影响整体视觉体验。",
    "fusion_decision": "agree"
  }
}
```

- **L1 `evidence`**：Laplacian 方差 11 ≤ 85，数值化判定面部偏糊
- **L2 `root_cause_hypothesis`**：根因为生成伪影
- **L3 `vlm_reasoning`**：VLM 语义确认 + UX 影响 + 融合策略 `agree`（0.4×0.68 + 0.6×0.85 = 0.782）

#### 3. decision_trace — Agent 编排全链路（8 阶段）

```json
[
  { "stage": "mode_select",  "module": "AgentOrchestrator", "decision": "mode_fast_agent",    "duration_ms": 0.0    },
  { "stage": "global_scan",  "module": "GlobalScanner",     "decision": "scan_complete",      "duration_ms": 3357.9 },
  { "stage": "routing",      "module": "FastRouter",        "decision": "route_complete",     "duration_ms": 0.0    },
  { "stage": "detection",    "module": "SubDetectors",      "decision": "detectors_complete", "duration_ms": 456.5  },
  { "stage": "vlm_confirm",  "module": "VLMConfirm",        "decision": "vlm_confirmed",      "duration_ms": 9878.3,
    "input_summary":  { "detector": "face_artifact" },
    "output_summary": { "is_degraded": true, "vlm_confidence": 0.85 } },
  { "stage": "judge",        "module": "LLMJudge",          "decision": "judge_uncertain",    "duration_ms": 3436.5 },
  { "stage": "detection",    "module": "Round2Executor",    "decision": "round2_complete",    "duration_ms": 0.0    },
  { "stage": "aggregation",  "module": "ReportGenerator",   "decision": "aggregate_mos",      "duration_ms": 0.0    }
]
```

**阶段链**：

```text
mode_select → global_scan → routing → detection
  → vlm_confirm       (灰区 face_artifact 经 VLM 确认，置信度 0.68→0.78)
  → judge             (LLM 全帧审查，判定 uncertain)
  → round2            (Judge 触发补检)
  → aggregation       (MOS 聚合)
```

#### 4. degradation_summary

```json
{
  "total_count": 4,
  "by_severity":   { "moderate": 3, "minor": 1 },
  "by_detector":   { "compression_artifact": 1, "mosaic_artifact": 1, "blur_artifact": 1, "face_artifact": 1 },
  "by_root_cause": { "encoding_loss": 3, "generation_artifact": 1 },
  "top_issues": [
    "画面出现块状压缩伪影/高频纹理损失",
    "画面出现马赛克或过度像素化块效应",
    "主体区域出现区域性模糊/纹理损失"
  ]
}
```

#### 5. mos_breakdown — MOS 衰减建模

```json
{
  "base_mos": 4.5,
  "total_penalty": -0.8273,
  "penalties": [
    { "source": "compression_artifact", "penalty": -0.35, "effective_penalty": -0.3500, "decay_index": 0 },
    { "source": "mosaic_artifact",      "penalty": -0.35, "effective_penalty": -0.2450, "decay_index": 1 },
    { "source": "blur_artifact",        "penalty": -0.32, "effective_penalty": -0.1568, "decay_index": 2 },
    { "source": "face_artifact",        "penalty": -0.22, "effective_penalty": -0.0755, "decay_index": 3 }
  ]
}
```

`effective_penalty = penalty × 0.7^decay_index`：第一条全量扣分，后续递减。

#### 6. performance — 各阶段耗时

```json
{
  "total_ms":       17136,
  "global_scan_ms":  3358,
  "detection_ms":     457,
  "vlm_ms":          9883,
  "judge_ms":        3437
}
```

VLM 是主要瓶颈（占总耗时 58%），单次 ROI 确认约 10 秒（7B 模型本地推理）。

### v0.1 vs V1 关键字段对比（`case_blur`）

| 字段 | v0.1 基线 | V1 Agent |
|------|-----------|----------|
| `system_version` | `0.1.0` | `1.0.0` |
| `overall_mos` | 3.673 | 3.673 |
| degradation 数 | 4 | 4 |
| `face_artifact` confidence | 0.680（灰区） | **0.782**（VLM 融合后） |
| `vlm_reasoning` | ❌ 无 | ✅ 含中文语义 + UX 影响 |
| `agent_meta.vlm_calls` | — | **1** |
| `agent_meta.judge_assessment` | — | `uncertain` |
| `decision_trace` 阶段数 | 5 | **8**（多出 vlm_confirm + judge + round2） |
| VLM Confirm 阶段 | ❌ | ✅ `vlm_confirmed` |
| Judge + Round 2 阶段 | ❌ | ✅ `judge_uncertain` + `round2_complete` |

### 复现命令

```bash
# Demo 素材已随仓库提供（docs/demo/*.png）
# 无需任何外部数据集

# v0.1 基线（对任一 case 跑）
python detect.py \
  --image docs/demo/case_blur.png \
  --legacy-fixed \
  --output report_v01.json

# V1 完整（需 Ollama）
python detect.py \
  --image docs/demo/case_blur.png \
  --config config.yaml \
  --output report_v1.json

# HTML 可视化报告
python detect.py \
  --image docs/demo/case_blur.png \
  --config config.yaml \
  --output docs/demo/case_blur_v1.html

# 批量跑所有 case
for case in docs/demo/case_*.png; do
  python detect.py --image "$case" --legacy-fixed --output "${case%.png}_report.json"
done
```

> 完整 JSON 报告含 `region_mask_rle`（RLE 像素 mask）等字段，此处为可读性裁剪。实际输出可通过 `--output -` 打印到 stdout 查看。

---

## 使用说明

### CLI 参数

| 参数 | 说明 |
|------|------|
| `--image PATH` | 单张输入图像 |
| `--image-dir PATH` | 图像目录（非递归） |
| `--mode fast` | 检测模式（v0.1 仅支持 `fast`） |
| `--config PATH` | YAML 配置文件（存在时默认读 `config.yaml`） |
| `--legacy-fixed` | v0.1 固定流水线，跳过 Agent 层 |
| `--output PATH` | 输出文件（`-` = stdout JSON） |
| `--output-dir PATH` | 批量输出目录 |
| `--metadata PATH` | 可选 metadata JSON 侧车文件 |
| `--ignore-regions PATH` | 可选 ignore-regions JSON 侧车文件 |
| `--frame-id ID` | 覆盖报告中的 `frame_id` |
| `--verbose` | 详细日志 |

安装后也可直接使用：

```bash
# V1（默认）
lqdd --image /path/to/frame.png --config config.yaml --output report_v1.html

# v0.1 基线
lqdd --image /path/to/frame.png --legacy-fixed --output report_v01.html
```

### 流水线模式

| 模式 | 用法 | 默认？ | 适用场景 |
|------|------|--------|----------|
| **V1 agent** | 不加 `--legacy-fixed`，`agent.enabled: true` | **是** | 灰区 VLM 确认、Judge 审查、Round 2 |
| **v0.1 fast** | 加 `--legacy-fixed` | 否 | 可复现基线、CI、阈值调参 |

关闭 Agent 而不加 CLI 参数时，可在 `config.yaml` 中设置 `agent.enabled: false`（等效于始终走 v0.1 编排外的 FastPipeline）。

---

## 配置

将 `config.example.yaml` 复制为 `config.yaml`（已 gitignore），按需调整检测器阈值、Agent 路由及 VLM / Judge 参数。

### 检测器

- `global_scan` — 提名阈值与 ROI 提示
- `edge_bleed`、`compression`、`blur` 等 — 各检测器参数
- `report` — MOS 基线与衰减系数

### V1 Agent / VLM / Judge（`config.example.yaml` 默认值）

```yaml
agent:
  enabled: true
  max_rounds: 2
  high_confidence_threshold: 0.7   # ≥ 此值：不调 VLM
  grey_zone_lower: 0.4             # 灰区下界
  grey_zone_upper: 0.7             # 灰区上界（不含）
  max_detectors_per_frame: 9
  hard_decision_threshold: 0.55

vlm:
  provider: ollama
  model: qwen2.5vl:7b
  host: http://localhost:11434
  timeout_ms: 120000
  max_calls_per_frame: 3

judge:
  provider: ollama
  model: qwen2.5:3b
  host: http://localhost:11434
  timeout_ms: 45000
```

可通过环境变量覆盖：`OLLAMA_HOST`、`LQDD_VLM_MODEL`、`LQDD_JUDGE_MODEL`、`LQDD_AGENT_ENABLED=0`（详见 [环境要求与安装 → 与 LQDD 的对接方式](#环境要求与安装)）。

---

## 评测（可选，本地数据）

本仓库**不包含**评测图像或 GT mask。如需可复现指标，可在本地生成合成 benchmark：

1. 一组**干净源 PNG 帧**（人像 / 抠图合成类内容效果较好）
2. 数据目录下的独立 **`degradation` 合成工具包**（见 `scripts/generate_benchmark_dataset.py`）

```bash
export LQDD_DATA_DIR=/path/to/data

# 需要：$LQDD_DATA_DIR/degradation/synthesize.py 与干净源帧目录
python scripts/generate_benchmark_dataset.py \
  --input  $LQDD_DATA_DIR/source_frames \
  --output $LQDD_DATA_DIR/synthetic_benchmark \
  --samples-per-type 5 \
  --clean-count 8
```

生成目录结构：

```
synthetic_benchmark/
├── images/          # 劣化图像
├── masks/           # pixel GT（像素值 = class_id）
└── manifest.json    # bbox、primary_type、source 等元数据
```

批量评测：

```bash
python benchmark/run_eval.py \
  --manifest $LQDD_DATA_DIR/synthetic_benchmark/manifest.json \
  --output benchmark/runs/results.json

python benchmark/run_eval_smoke.py   # 按 GT 类型快速 smoke
```

GT 类型与检测器映射见 [`benchmark/README.md`](benchmark/README.md)、[`data/sample/README.md`](data/sample/README.md)。

从 manifest 生成 Demo HTML 报告（输出到 gitignore 的 `examples/`）：

```bash
python scripts/generate_demo_assets.py
```

> **注意**：`benchmark/run_eval.py` 与 `scripts/generate_demo_assets.py` 当前走 **v0.1 `--legacy-fixed` 路径**，用于可复现的检测器基线评测与可视化 Demo。要评测 V1 Agent 效果，请对单张或批量帧使用不带 `--legacy-fixed` 的 `detect.py`，并对比 JSON 中的 `decision_trace` / `agent_meta`。

---

## 开发

### 运行测试

```bash
pytest tests/ -m "not vlm" -q
```

依赖 VLM 服务的测试标记为 `@pytest.mark.vlm`，默认跳过。

### 项目结构

```
├── detect.py                 # CLI 入口
├── requirements.txt          # 运行时依赖
├── requirements-dev.txt      # 开发依赖（含 -e .）
├── requirements-optional.txt # mediapipe（可选，增强实验性 hand_anomaly）
├── config.example.yaml       # 配置模板
├── src/lqdd/
│   ├── detectors/            # 九个劣化检测器
│   ├── agent/                # 提名路由、Judge 客户端
│   ├── vlm/                  # Ollama VLM 客户端
│   ├── pipeline/             # v0.1 + V1 编排
│   └── report/               # JSON / HTML 报告构建
├── benchmark/                # 批量评测脚本
├── scripts/                  # benchmark 生成、Demo 资源
├── tests/
└── specs/                    # 设计文档与契约
```

---

## 文档

| 文档 | 说明 |
|------|------|
| [`specs/USE_CASE_BADCASE.md`](specs/USE_CASE_BADCASE.md) | Badcase 场景与边界 |
| [`specs/VERSION_ROADMAP.md`](specs/VERSION_ROADMAP.md) | v0.1 / V1 / V2 路线 |
| [`specs/001-v0-fast-mvp/`](specs/001-v0-fast-mvp/) | v0.1 契约与 schema |
| [`specs/002-v1-agent-layer/`](specs/002-v1-agent-layer/) | V1 Agent / VLM 契约 |

---

## 项目状态

### 已完成

| 模块 | 状态 | 说明 |
|------|------|------|
| v0.1 固定流水线 | ✅ | GlobalScan → 9 检测器 → Report（`--legacy-fixed`） |
| V1 Agent 编排 | ✅ | 路由 → VLM 灰区确认 → Judge → Round 2（默认） |
| CLI + JSON/HTML 报告 | ✅ | 单图 / 批量 / schema 校验 |
| 6/8 检测器可用 | ✅ | edge_bleed / compression / blur / mosaic / overexposure / green_spill |
| Ollama VLM + Judge | ✅ | `qwen2.5vl:7b` + `qwen2.5:3b`，优雅降级 |

### 已知限制

| 限制 | 说明 | 影响 |
|------|------|------|
| `banding_artifact` 检出率低 | 阈值偏保守，对合成色带不敏感 | Demo 中 banding case 未检出 |
| `hand_anomaly` 实验性 MVP | 启发式 ROI + 边缘密度 fallback；spec 中的多指/粘连未实现 | 已标注为实验性，不建议作为主力能力展示 |
| 干净帧误报率偏高 | `compression` / `blur` / `mosaic` 在干净帧上有交叉敏感 | Benchmark clean FPR 未达标 |
| Deep Mode 未实现 | `--mode deep` 返回 exit 2 | V1 仅 Fast Mode 可用 |
| 实机 VLM 测试缺失 | 无 `@pytest.mark.vlm` 标注的真实 Ollama 集成测试 | VLM 验证依赖手动 E2E |

### V2 规划（未实现）

- 视频 clip 输入 + 时序采样
- `TemporalFlicker` 时域闪烁检测
- Deep Mode（VLM 先行 + 子检测器量化）

详见 [`specs/VERSION_ROADMAP.md`](specs/VERSION_ROADMAP.md)。

---

## License

[Apache License 2.0](LICENSE)
