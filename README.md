# LocalQualityDegradationDetection

**无参考、可解释的局部画质劣化检测**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)

输入离线单帧图像，输出不规则 **region mask**、数值 **Evidence**、帧级 **MOS 总分**与完整 **Decision Trace**（JSON / HTML）。无需参考图或 diff，适用于 badcase 定位与归因、离线质检与后处理排查等场景。

---

## 特性

- **九个专用检测器**：覆盖压缩、模糊、马赛克、色带、绿幕溢色、发丝、面部、背景等；`hand_anomaly` 为**实验性 MVP**（见下表）
- **可解释输出**：归因来自 `degradations[]`——RLE 像素 mask、bbox、各检测器 evidence、root_cause、VLM 语义确认、结构化 decision trace（**与 MOS 计算解耦**）
- **两种流水线**：v0.1 确定性快速路径；V1 ReAct Agent（LLM 自主决策驱动 VLM 调用与补检）
- **真正的 Agent 自主决策**：LLM 在看到 CV 结果后，自主判断是否调用 VLM、是否补检——而非硬编码阈值触发
- **MOS 打分**：默认 rule 启发式（零依赖）；可选 `mos_model=clip_iqa` 用 CLIP-IQA 无参考感知预测。**MOS 只是一个总分，per-item 罚分明细非感知归因**（详见 [MOS 与归因](#mos-与归因)）
- **优雅降级**：Ollama 不可用时，Agent 步骤自动走规则降级并在 trace 中记录原因
- **可选评测工具链**：本地合成带 pixel GT 的劣化样本并批量评测（数据不随仓库分发）

---

## 架构

### V1 ReAct Agent 架构（默认）

```
GlobalScan → 9 detectors → ReAct Agent Loop → Report
                                   │
                    ┌──────────────▼──────────────┐
                    │  Observe: CV 检测结果         │
                    │  Think:   LLM 推理分析        │  ← LLM 是决策中心
                    │  Act:     自主选择工具调用     │
                    │    ├─ vlm_analyze   (VLM确认) │
                    │    ├─ rerun_detector(重检)    │
                    │    ├─ dispatch_compression    │
                    │    └─ accept       (终止)     │
                    └─────────────────────────────-┘
```

**核心改变**：VLM 是否调用不再由硬编码阈值决定，而由 **LLM Agent 观察 CV 结果后自主决策**。

### v0.1 基线架构（`--legacy-fixed`）

```
GlobalScan → 9 detectors → Report
```

无 Agent / VLM / Judge，全量检测器确定性运行。

---

## 检测器

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

> **`hand_anomaly` 说明**：检测器已接入流水线，但 ROI 为前景 bbox 下半段的启发式区域（非真实手部分割）；未装 `mediapipe` 时走边缘密度 fallback，**仍会运行**但能力有限。spec 中的多指计数、粘连、手部模糊等**尚未实现**；benchmark **无** hand 类 GT。

**v0.1**（`--legacy-fixed`）：固定 Fast Pipeline，全量检测器确定性运行，无 Agent / VLM。  
**V1**（**默认**）：ReAct Agent 编排，LLM 自主决策驱动工具调用，见下文 [V1 ReAct Agent 流水线](#v1-react-agent-流水线)。

报告字段包括 `region_mask_rle`、`bbox`、`evidence`、`decision_trace`；V1 额外包含 `agent_meta`（含 `agent_steps` 决策轨迹）、`vlm_reasoning`、`vlm_reasoning_summary`。

---

## 环境要求与安装

### 环境要求

| 项目 | 要求 | 说明 |
|------|------|------|
| **Python** | **3.10 ~ 3.12**（推荐 3.11） | `requires-python = ">=3.10"`，见 `pyproject.toml` |
| **操作系统** | macOS / Linux / Windows | 无特殊限制 |
| **磁盘** | ≥ 2 GB（不含模型） | 代码 + venv + 依赖 |
| **Ollama（V1 推荐）** | 另需 **~6 GB+** | VLM `qwen2.5vl:7b` 约 4.7 GB；Judge `qwen2.5:1.5b` 约 1 GB |
| **内存（V1 + VLM）** | 建议 **16 GB+** | 7B 视觉模型本地推理时占用较高 |

- **仅跑 v0.1**（`--legacy-fixed`）：只需 Python 环境，**不需要 Ollama**
- **跑 V1 Agent**（默认）：需要 Python + **Ollama 已安装、已拉模型、服务可访问**

### Python 依赖

依赖清单以 [`requirements.txt`](requirements.txt) 为准（与 [`pyproject.toml`](pyproject.toml) 同步）：

| 文件 | 内容 |
|------|------|
| [`requirements.txt`](requirements.txt) | 运行时依赖（OpenCV、NumPy、YAML、JSON Schema、httpx） |
| [`requirements-dev.txt`](requirements-dev.txt) | 开发环境：`-r requirements.txt` + pytest + Pillow + 可编辑安装 |
| [`requirements-optional.txt`](requirements-optional.txt) | **可选增强**：`mediapipe`（提升实验性 `hand_anomaly`）；`pyiqa + torch + torchvision`（启用 `mos_model=clip_iqa`，均非必需） |

| 包 | 版本要求 | 用途 |
|----|----------|------|
| `opencv-python-headless` | ≥ 4.8 | 图像读写与 CV 算子 |
| `numpy` | ≥ 1.24 | 数值计算 |
| `PyYAML` | ≥ 6.0 | 读取 `config.yaml` |
| `jsonschema` | ≥ 4.20 | 报告 schema 校验 |
| `httpx` | ≥ 0.27 | 调用 Ollama HTTP API（VLM / Agent LLM） |
| `pytest` | ≥ 7.4 | 测试（仅 dev） |
| `Pillow` | ≥ 10.0 | 测试与 Demo（仅 dev） |
| `mediapipe` | ≥ 0.10 | **可选**：为实验性 `hand_anomaly` 提供 MediaPipe 关键点路径（不装也能跑 fallback） |
| `pyiqa` | ≥ 0.1.10 | **可选**：启用 `mos_model="clip_iqa"` — 使用 CLIP-IQA（ICCV 2023）无参考画质预测，替代硬编码衰减公式 |
| `torch` | ≥ 2.0 | **可选**：`pyiqa` 推理后端（CLIP-IQA 依赖） |
| `torchvision` | ≥ 0.15 | **可选**：`pyiqa` 图像预处理（CLIP-IQA 依赖） |

### 1. 安装 Python 项目

```bash
git clone https://github.com/Chamuel08/LocalQualityDegradationDetection.git
cd LocalQualityDegradationDetection

# 创建虚拟环境（推荐，必须使用虚拟环境）
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 升级 pip
pip install -U pip setuptools wheel

# 开发环境（推荐：依赖 + 可编辑安装 + pytest）
pip install -r requirements-dev.txt

# 仅运行时依赖 + 手动安装包：
# pip install -r requirements.txt
# pip install -e .

# 可选：增强实验性 hand_anomaly + 启用 CLIP-IQA MOS 模式（均非必需）
# pip install -r requirements-optional.txt
#
# 或只装 CLIP-IQA 相关依赖（启用 mos_model="clip_iqa"）：
# pip install "lqdd[clip_iqa]"   # 即 pyiqa + torch + torchvision + setuptools
# 首次运行时会自动下载 CLIP-IQA 权重（~260 MB，缓存到 ~/.cache/torch/hub/pyiqa/）

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

V1 的 LLM Agent 决策与 VLM 视觉确认通过 **本地 Ollama** 调用，不依赖云端 API。

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
# VLM：Agent 决策触发时的 ROI 视觉确认（约 4.7 GB）
ollama pull qwen2.5vl:7b

# Judge/Agent LLM：自主决策（约 1 GB）
ollama pull qwen2.5:1.5b
```

确认已下载：

```bash
ollama list
# 应能看到 qwen2.5vl:7b 与 qwen2.5:1.5b
```

快速冒烟（可选，确认模型能推理）：

```bash
ollama run qwen2.5:1.5b "reply ok"
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
  max_calls_per_frame: 3    # Agent 单帧最多触发 VLM 次数

judge:
  provider: ollama
  model: qwen2.5:1.5b
  host: http://localhost:11434
  timeout_ms: 1500          # Agent 决策 LLM 超时
```

环境变量覆盖（可选）：

| 变量 | 作用 |
|------|------|
| `OLLAMA_HOST` | 覆盖 VLM / Agent LLM 的 Ollama 地址（如 `http://127.0.0.1:11434`） |
| `LQDD_VLM_MODEL` | 覆盖 VLM 模型名 |
| `LQDD_JUDGE_MODEL` | 覆盖 Agent LLM 模型名 |
| `LQDD_AGENT_ENABLED=0` | 关闭 Agent，等效 v0.1 |

**代理注意**：若系统设置了 `HTTP_PROXY` 且指向不可用代理，可能导致访问 `localhost` 失败。本项目已对 Ollama 客户端设置 `trust_env=False`；若仍异常，可临时 `unset HTTP_PROXY HTTPS_PROXY` 后再跑。

#### 2.5 Ollama 常见问题

| 现象 | 处理 |
|------|------|
| `vlm_failed: service_unavailable` | 确认 `ollama serve` 在跑，`curl localhost:11434/api/tags` 有响应 |
| 首次 VLM 很慢 / 超时 | 调大 `vlm.timeout_ms`（如 `180000`）；确保模型已 `ollama pull` 完成 |
| `model not found` | 模型名与 `ollama list` 不一致时，改 `config.yaml` 或重新 `ollama pull` |
| 内存不足 | 换更小 Judge 模型，或仅跑 v0.1（`--legacy-fixed`） |

### 3. 验证完整环境

```bash
# 激活虚拟环境（必须）
source .venv/bin/activate

# 1) Python 依赖
python -c "import cv2, numpy, yaml, httpx; print('ok')"

# 2) Ollama（V1）
ollama list | grep -E 'qwen2.5'

# 3) v0.1 冒烟（无需 Ollama）
python detect.py --image docs/demo/case_blur.png --legacy-fixed --output /tmp/report_v01.json

# 4) V1 ReAct Agent 冒烟（需 Ollama）
python detect.py --image docs/demo/case_blur.png --config config.yaml --output /tmp/report_v1.json
python -c "import json; d=json.load(open('/tmp/report_v1.json')); steps=d.get('agent_meta',{}).get('agent_steps',[]); print('agent_steps:', len(steps)); [print(f'  step{s[\"step\"]}: {s[\"action\"]} - {s[\"thought\"][:40]}') for s in steps]"
```

`agent_meta.agent_steps` 中出现 `vlm_analyze` 条目，表示 LLM Agent 自主决策调用了 VLM（需该帧存在不确定检出且 Ollama 正常）。

---

## 快速开始

完成 [环境要求与安装](#环境要求与安装) 后，**默认即 V1 ReAct Agent 流水线**（`agent.enabled: true`）。

> ⚠️ **必须使用虚拟环境**：所有命令执行前请先 `source .venv/bin/activate`

### V1 ReAct Agent 完整流程（推荐）

前提：Ollama 已安装、已拉取模型、服务可访问。

```bash
source .venv/bin/activate

python detect.py \
  --image /path/to/frame.png \
  --config config.yaml \
  --output report_v1.html
```

- **不要**加 `--legacy-fixed`
- 输出 HTML 含 bbox 可视化；改 `--output report_v1.json` 可得 JSON
- Ollama 未运行时，Agent 自动降级为规则决策，流水线仍会完成

验证 Agent 是否真正跑通，见 [如何确认 V1 ReAct Agent 已生效](#如何确认-v1-react-agent-已生效)。

### v0.1 基线（无 Agent）

**无需 Ollama**，适用于 CI、阈值调参、与 V1 对比：

```bash
source .venv/bin/activate

python detect.py \
  --image /path/to/frame.png \
  --legacy-fixed \
  --output report_v01.html
```

- `--legacy-fixed` — 跳过 Agent / VLM，仅规则检测器
- `--output report.html` — HTML 报告；`-` 表示 JSON 打印到 stdout

批量模式（v0.1）：

```bash
python detect.py \
  --image-dir /path/to/frames/ \
  --legacy-fixed \
  --output-dir reports/
```

---

## V1 ReAct Agent 流水线

V1 在规则检测器之上增加 **ReAct Agent 编排层**：LLM 观察所有 CV 检测结果，自主决定是否调用 VLM 进行视觉确认、是否补充运行检测器，直到输出 `accept` 终止。

> **架构核心变化**：之前 VLM 是否调用由硬编码的置信度阈值（灰区判断）决定；现在由 **LLM Agent 自主推理后决定**——它可以拒绝 VLM 调用（直接 accept），也可以主动要求补检。

### 阶段一览

| 阶段 | 模块 | 作用 |
|------|------|------|
| 1. GlobalScan | `GlobalScanner` | 全帧扫描，产出区域提名（nominations） |
| 2. Routing | `FastRouter` | 按提名置信度决定 dispatch / skip；派发 CV 检测器 |
| 3. Detection | 9 个 SubDetectors | 规则检测，产出 preliminary degradations + confidence |
| 4. **ReAct Agent Loop** | `ReactAgent` | **LLM 自主决策循环**（见下文） |
| 5. Report | `ReportGenerator` | 聚合 MOS、mask、bbox、完整 `decision_trace` |

### ReAct Agent 决策循环

```
while not done (最多 max_steps 步):
    Observe  → 将 CV 结果、历史步骤格式化给 LLM
    Think    → LLM 分析当前状态，输出 thought（推理过程）
    Act      → LLM 选择一个工具调用
    Execute  → 执行工具，获取 observation
    → 下一轮
```

**LLM 可调用的工具（Tools）**：

| 工具 | 作用 | 典型触发场景 |
|------|------|------------|
| `vlm_analyze` | 对指定检测项调用 VLM 视觉确认 | 置信度模糊、证据不足、需视觉辨别 |
| `rerun_detector` | 用调整后阈值重新运行指定检测器 | 怀疑漏检、需更细粒度扫描 |
| `dispatch_compression` | 补充运行压缩伪影检测器 | 全局 MOS 偏低但压缩未检出 |
| `accept` | 接受当前结果，终止循环 | 结果已充分可信 |

**VLM 触发条件**：由 LLM 自主判断，不再是固定阈值。LLM 的决策依据来自其推理，包括但不限于：
- 某检测项置信度偏低，视觉确认能显著提升可信度
- 高置信度结果（LLM 认为已足够）可跳过 VLM，直接 accept

### 如何确认 V1 ReAct Agent 已生效

打开 JSON 报告，检查以下字段：

**1. `decision_trace` 阶段链**

典型 V1 ReAct 链路应包含：

```text
mode_select → global_scan → routing → detection
  → agent_step（decision: agent_vlm_analyze_step1）   ← LLM 自主决策调 VLM
  → agent_step（decision: agent_accept_step2）        ← LLM 自主决策终止
  → aggregation
```

**2. `agent_meta`**

```json
{
  "rounds_executed": 1,
  "vlm_calls": 3,
  "judge_assessment": null,
  "agent_driven_vlm": true,
  "agent_steps": [
    {
      "step": 1,
      "thought": "检测到低置信度项，视觉确认能提升可信度",
      "action": "vlm_analyze",
      "reason": "置信度 0.68 处于模糊区间，需要 VLM 确认",
      "observation": "vlm_analyze 完成：确认了 3 个检测项",
      "latency_ms": 9450.2
    },
    {
      "step": 2,
      "thought": "VLM 已确认，结果可信",
      "action": "accept",
      "reason": "无需进一步分析",
      "observation": "Agent 终止：无需进一步分析",
      "latency_ms": 312.1
    }
  ]
}
```

关键字段说明：

| 字段 | 含义 |
|------|------|
| `agent_driven_vlm: true` | VLM 是 Agent 自主决策调用的（而非硬编码触发） |
| `agent_steps` | 每一步 LLM 的完整推理轨迹（thought + action + observation） |
| `judge_assessment: null` | ReAct 模式下没有独立的 Judge 审查阶段 |

**3. 单条 degradation 的 `vlm_reasoning`**

Agent 触发 VLM 后经确认的条目会有：

```json
{
  "reasoning": "面部区域存在明显的模糊现象，细节损失严重",
  "vlm_confidence": 0.85,
  "ux_impact": "面部模糊使观看者难以辨认人物，影响视觉体验",
  "fusion_decision": "agree"
}
```

**4. 常见降级原因（Ollama 不可用时）**

| trace decision | 含义 |
|----------------|------|
| `vlm_failed: service_unavailable` | Ollama 未启动或请求失败，Agent 降级为规则决策 |
| `vlm_skipped: quota_exceeded` | 单帧 VLM 调用达 `max_calls_per_frame` 上限 |

### V1 ReAct Agent vs v0.1 对比示例

同一张图分别跑两条路径，对比 Agent 层贡献：

```bash
source .venv/bin/activate

# v0.1 基线（无 Agent，无 VLM）
python detect.py --image docs/demo/case_blur.png --legacy-fixed --output report_v01.json

# V1 ReAct Agent（需 Ollama）
python detect.py --image docs/demo/case_blur.png --config config.yaml --output report_v1.json
```

| 字段 | v0.1 基线 | V1 ReAct Agent |
|------|-----------|----------------|
| `system_version` | `0.1.0` | `1.0.0` |
| `overall_mos` | 3.673 | 3.673 |
| degradation 数 | 4 | 4 |
| `face_artifact` confidence | 0.680 | **0.782**（VLM 融合后） |
| `vlm_reasoning` | ❌ 无 | ✅ 含中文语义 + UX 影响 |
| `agent_meta.agent_steps` | — | ✅ 完整决策轨迹 |
| `agent_meta.agent_driven_vlm` | — | `true` |
| `decision_trace` 阶段数 | 5 | **7**（含 agent_step × N） |
| VLM 调用决策方 | — | **LLM Agent 自主判断** |

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

> `banding` 检测器当前阈值偏保守，对合成色带的检出率较低（已知限制）。其余 6 种劣化均被对应检测器成功检出。

### V1 ReAct Agent 完整示例（`case_blur`）

选取 `case_blur`（区域模糊）展示 V1 ReAct Agent 完整流程。该样本中 `face_artifact` 置信度 0.68，**LLM Agent 自主决策触发了 VLM 确认**。

#### 1. 报告总览

```json
{
  "system_version": "1.0.0",
  "overall_mos": 3.673,
  "severity": "moderate",
  "agent_meta": {
    "rounds_executed": 1,
    "vlm_calls": 3,
    "agent_driven_vlm": true,
    "agent_steps": [
      {
        "step": 1,
        "thought": "检测到低置信度项，需要 VLM 视觉确认",
        "action": "vlm_analyze",
        "reason": "置信度 0.68 不足，需要 VLM 确认",
        "observation": "vlm_analyze 完成：确认了 3 个检测项，当前 VLM 调用总计 3 次"
      },
      {
        "step": 2,
        "thought": "已执行过 VLM 确认，接受当前结果",
        "action": "accept",
        "reason": "VLM 已确认，结果可信",
        "observation": "Agent 终止：VLM 已确认，结果可信"
      }
    ]
  }
}
```

`agent_driven_vlm: true` 表示 VLM 由 LLM Agent 自主触发，而非硬编码路由。

#### 2. VLM 确认的 degradation — 三层可解释结构

`face_artifact` 原始置信度 0.68，经 Agent 触发 VLM 确认后融合置信度提升至 0.782：

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
- **L3 `vlm_reasoning`**：LLM Agent 自主决策 → VLM 语义确认 + UX 影响 + 融合策略 `agree`（0.4×0.68 + 0.6×0.85 = 0.782）

#### 3. decision_trace — ReAct Agent 编排全链路

```json
[
  { "stage": "mode_select",  "module": "AgentOrchestrator", "decision": "mode_fast_react_agent" },
  { "stage": "global_scan",  "module": "GlobalScanner",     "decision": "scan_complete"         },
  { "stage": "routing",      "module": "FastRouter",        "decision": "route_complete"        },
  { "stage": "detection",    "module": "SubDetectors",      "decision": "detectors_complete"    },
  { "stage": "agent_step",   "module": "ReactAgent",        "decision": "agent_vlm_analyze_step1",
    "input_summary":  { "thought": "检测到低置信度项，需要 VLM 视觉确认" },
    "output_summary": { "action": "vlm_analyze", "observation": "vlm_analyze 完成：确认了 3 个检测项" } },
  { "stage": "agent_step",   "module": "ReactAgent",        "decision": "agent_accept_step2",
    "input_summary":  { "thought": "已执行过 VLM 确认，接受当前结果" },
    "output_summary": { "action": "accept" } },
  { "stage": "aggregation",  "module": "ReportGenerator",   "decision": "aggregate_mos"         }
]
```

**阶段链**：

```text
mode_select → global_scan → routing → detection
  → agent_step(vlm_analyze)  ← LLM 自主决策：调用 VLM 确认
  → agent_step(accept)       ← LLM 自主决策：结果可信，终止
  → aggregation
```

#### 4. mos_breakdown — MOS 总分

**MOS 与归因是两件事**：
- **归因**（劣化是什么 / 在哪 / 为什么）看 `degradations[]`（detector、bbox、evidence、root_cause、vlm_reasoning），与 MOS 无关。
- **MOS** 只是帧级一个总分。推荐 `mos_model=clip_iqa`，由 CLIP-IQA 无参考感知预测直接给出，`penalties` 为空：

```json
{
  "base_mos": 3.673,
  "total_penalty": 0.0,
  "cap_applied": false,
  "cap_reason": "mos_model=clip_iqa，分数由 CLIP-IQA 直接预测",
  "penalties": []
}
```

未装 pyiqa/torch 时降级到 `rule` 后端：`MOS = base_mos + Σ(penalty_i × decay_factor^i)`（零依赖兜底，非感知归因，`penalties` 仅为求和明细）。

### 复现命令

```bash
# Demo 素材已随仓库提供（docs/demo/*.png）
# 无需任何外部数据集

source .venv/bin/activate

# v0.1 基线（对任一 case 跑）
python detect.py \
  --image docs/demo/case_blur.png \
  --legacy-fixed \
  --output report_v01.json

# V1 ReAct Agent（需 Ollama）
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

# 查看 Agent 决策轨迹
python -c "
import json
d = json.load(open('report_v1.json'))
for step in d.get('agent_meta', {}).get('agent_steps', []):
    print(f'Step {step[\"step\"]}: [{step[\"action\"]}] {step[\"thought\"]}')
    print(f'  → {step[\"observation\"]}')
"
```

> 完整 JSON 报告含 `region_mask_rle`（RLE 像素 mask）等字段，此处为可读性裁剪。实际输出可通过 `--output -` 打印到 stdout 查看。

---

## Python API 调用

除 CLI 外，也可直接在 Python 代码中使用：

### V1 ReAct Agent（推荐）

```python
import cv2
from lqdd.config.loader import load_config
from lqdd.models.inputs import SingleFrameInput
from lqdd.agent.orchestrator import AgentOrchestrator

# 激活虚拟环境后再运行（source .venv/bin/activate）

config = load_config()   # 读取 config.yaml

frame = cv2.imread("path/to/frame.png")
fi = SingleFrameInput(frame=frame, frame_id="my_frame", mode="fast")

orc = AgentOrchestrator(config)   # 使用 OllamaJudgeClient（需 Ollama）
report = orc.run(fi)

print("MOS:", report.overall_mos)
print("degradations:", len(report.degradations))

# 查看 Agent 自主决策轨迹（agent_meta 在 Agent 禁用或不可用时为 None，需做保护）
if report.agent_meta:
    for step in report.agent_meta["agent_steps"]:
        print(f"Step {step['step']}: [{step['action']}] {step['thought']}")
        print(f"  → {step['observation']}")
```

### V1 使用 MockJudgeClient（无需 Ollama，适合测试）

```python
from lqdd.agent.orchestrator import AgentOrchestrator
from lqdd.agent.judge_client import MockJudgeClient

# MockJudgeClient：模拟 LLM 决策，不需要 Ollama
orc = AgentOrchestrator(config, judge_client=MockJudgeClient())
report = orc.run(fi)

# agent_driven_vlm 为 True 时表示 Agent 自主决策触发了 VLM
if report.agent_meta:
    print("agent_driven_vlm:", report.agent_meta["agent_driven_vlm"])
```

### V1 使用 RuleBasedJudgeClient（规则降级，无需 Ollama）

```python
from lqdd.agent.judge_client import RuleBasedJudgeClient

# RuleBasedJudgeClient：基于规则模拟 Agent 决策（Ollama 不可用时的降级路径）
orc = AgentOrchestrator(config, judge_client=RuleBasedJudgeClient(config.agent))
report = orc.run(fi)
```

### v0.1 基线 FastPipeline

```python
from lqdd.pipeline.fast_pipeline import FastPipeline

pipeline = FastPipeline(config)
report = pipeline.run(fi)
```

### 自定义 LLM Agent（实现 JudgeClient 接口）

```python
from lqdd.agent.judge_client import JudgeClient
from typing import Any

class MyCustomAgent(JudgeClient):
    def review(self, prompt: str) -> dict[str, Any] | None:
        # 旧式 judge 接口（向后兼容）
        return {"assessment": "consistent", "actions": [], "needs_round2": False}

    def decide(self, system_prompt: str, user_prompt: str) -> dict[str, Any] | None:
        # ReAct Agent 核心决策接口
        # system_prompt: AGENT_SYSTEM_PROMPT（工具定义 + 决策原则）
        # user_prompt:   当前 CV 结果 + 历史步骤（AGENT_OBSERVE_TEMPLATE）
        # 返回格式：{"thought": "...", "action": "vlm_analyze|accept|...", "reason": "..."}
        ...

orc = AgentOrchestrator(config, judge_client=MyCustomAgent())
```

---

## 使用说明

### CLI 参数

| 参数 | 说明 |
|------|------|
| `--image PATH` | 单张输入图像 |
| `--image-dir PATH` | 图像目录（非递归） |
| `--mode fast` | 检测模式（当前仅支持 `fast`） |
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
# V1 ReAct Agent（默认）
lqdd --image /path/to/frame.png --config config.yaml --output report_v1.html

# v0.1 基线
lqdd --image /path/to/frame.png --legacy-fixed --output report_v01.html
```

### 流水线模式

| 模式 | 用法 | 默认？ | 适用场景 |
|------|------|--------|----------|
| **V1 ReAct Agent** | 不加 `--legacy-fixed`，`agent.enabled: true` | **是** | LLM 自主决策驱动 VLM / 补检 |
| **v0.1 fast** | 加 `--legacy-fixed` | 否 | 可复现基线、CI、阈值调参 |

关闭 Agent 而不加 CLI 参数时，可在 `config.yaml` 中设置 `agent.enabled: false`。

---

## 配置

将 `config.example.yaml` 复制为 `config.yaml`（已 gitignore），按需调整检测器阈值、Agent 参数及 VLM 参数。

### 检测器

- `global_scan` — 提名阈值与 ROI 提示
- `edge_bleed`、`compression`、`blur` 等 — 各检测器参数
- `report` — MOS 基线、衰减系数、`mos_model`（`rule` / `clip_iqa` / `internal`，默认 `rule`）

### V1 ReAct Agent / VLM（`config.example.yaml` 默认值）

```yaml
agent:
  enabled: true
  max_rounds: 2             # Agent 最大循环步数 = max_rounds × 3
  high_confidence_threshold: 0.7   # 供规则降级参考（LLM 模式下由 LLM 自主判断）
  grey_zone_lower: 0.4             # 供规则降级参考
  grey_zone_upper: 0.7             # 供规则降级参考
  max_detectors_per_frame: 9
  hard_decision_threshold: 0.55

vlm:
  provider: ollama
  model: qwen2.5vl:7b
  host: http://localhost:11434
  timeout_ms: 120000
  max_calls_per_frame: 3    # Agent 单帧最多可触发 VLM 的次数（防止超额调用）

judge:
  provider: ollama
  model: qwen2.5:1.5b       # Agent 决策 LLM
  host: http://localhost:11434
  timeout_ms: 1500
```

可通过环境变量覆盖：`OLLAMA_HOST`、`LQDD_VLM_MODEL`、`LQDD_JUDGE_MODEL`、`LQDD_AGENT_ENABLED=0`。

---

## 评测（可选，本地数据）

本仓库**不包含**评测图像或 GT mask。如需可复现指标，可在本地生成合成 benchmark：

1. 一组**干净源 PNG 帧**（人像 / 抠图合成类内容效果较好）
2. 数据目录下的独立 **`degradation` 合成工具包**（见 `scripts/generate_benchmark_dataset.py`）

```bash
source .venv/bin/activate
export LQDD_DATA_DIR=/path/to/data

python scripts/generate_benchmark_dataset.py \
  --input  $LQDD_DATA_DIR/source_frames \
  --output $LQDD_DATA_DIR/synthetic_benchmark \
  --samples-per-type 5 \
  --clean-count 8
```

批量评测：

```bash
python benchmark/run_eval.py \
  --manifest $LQDD_DATA_DIR/synthetic_benchmark/manifest.json \
  --output benchmark/runs/results.json

python benchmark/run_eval_smoke.py
```

GT 类型与检测器映射见 [`benchmark/README.md`](benchmark/README.md)、[`data/sample/README.md`](data/sample/README.md)。

> **注意**：`benchmark/run_eval.py` 当前走 **v0.1 `--legacy-fixed` 路径**，用于可复现的检测器基线评测。要评测 V1 ReAct Agent 效果，请对单张或批量帧使用不带 `--legacy-fixed` 的 `detect.py`，并对比 JSON 中的 `decision_trace` / `agent_meta.agent_steps`。

---

## 开发

### 运行测试

```bash
source .venv/bin/activate
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
│   ├── agent/
│   │   ├── orchestrator.py   # ReAct Agent 编排核心
│   │   ├── judge_client.py   # LLM Agent 客户端（OllamaJudgeClient / MockJudgeClient / RuleBasedJudgeClient）
│   │   ├── prompts.py        # AGENT_SYSTEM_PROMPT + AGENT_OBSERVE_TEMPLATE（ReAct 提示词）
│   │   ├── router.py         # CV 检测器派发路由（不再负责 VLM 决策）
│   │   ├── actions.py        # Round 2 工具执行器（v0.1 兼容保留）
│   │   └── context.py        # AgentContext 工厂
│   ├── vlm/
│   │   ├── client.py         # Ollama VLM 客户端
│   │   ├── confirm.py        # VLM 确认执行（run_vlm_confirm_for_item：单项确认，供 Agent 调用）
│   │   └── fuse.py           # CV + VLM 置信度融合
│   ├── models/
│   │   ├── agent.py          # AgentStep / AgentAction / AgentMeta 等数据结构
│   │   └── report.py         # QualityReport / DegradationItem 等
│   ├── mos/                  # MOS 预测后端（按 ReportConfig.mos_model 分发）
│   │   └── clip_iqa.py       # 可选：CLIP-IQA 无参考画质预测（需 pyiqa + torch）
│   ├── pipeline/             # v0.1 FastPipeline + V1 AgentPipeline 入口
│   └── report/               # JSON / HTML 报告构建（含 compute_mos rule 衰减公式）
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
| V1 ReAct Agent 编排 | ✅ | LLM 自主决策循环（Observe→Think→Act），工具：vlm_analyze / rerun_detector / dispatch_compression / accept |
| CLI + JSON/HTML 报告 | ✅ | 单图 / 批量 / schema 校验 |
| 6/8 检测器可用 | ✅ | edge_bleed / compression / blur / mosaic / overexposure / green_spill |
| Ollama VLM + Agent LLM | ✅ | `qwen2.5vl:7b` + `qwen2.5:1.5b`，规则降级 |
| Agent 决策轨迹记录 | ✅ | `agent_meta.agent_steps`：完整 thought + action + observation |

### 已知限制

| 限制 | 说明 | 影响 |
|------|------|------|
| `banding_artifact` 检出率低 | 阈值偏保守，对合成色带不敏感 | Demo 中 banding case 未检出 |
| `hand_anomaly` 实验性 MVP | 启发式 ROI + 边缘密度 fallback；spec 中的多指/粘连未实现 | 已标注为实验性，不建议作为主力能力展示 |
| 干净帧误报率偏高 | `compression` / `blur` / `mosaic` 在干净帧上有交叉敏感 | Benchmark clean FPR 未达标 |
| Deep Mode 未实现 | `--mode deep` 返回 exit 2 | V1 仅 Fast Mode 可用 |
| Agent 小模型推理质量 | `qwen2.5:1.5b` 参数量小，复杂场景决策质量有限 | 可换更大模型（如 `qwen2.5:7b`）提升 Agent 推理质量 |

### V2 规划（未实现）

- 视频 clip 输入 + 时序采样
- `TemporalFlicker` 时域闪烁检测
- Deep Mode（VLM 先行 + 子检测器量化）
- Agent 主动发现（VLM Discover 模式，从零主动识别规则检不出的劣化）

详见 [`specs/VERSION_ROADMAP.md`](specs/VERSION_ROADMAP.md)。

---

## License

[Apache License 2.0](LICENSE)
