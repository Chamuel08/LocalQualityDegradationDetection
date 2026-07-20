# LocalQualityDegradationDetection

**无参考、可解释的局部画质劣化检测**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)

输入离线单帧图像（或视频 clip），输出劣化区域的像素 **mask**、**bbox**、数值 **Evidence**、帧级 **MOS 总分**与完整 **Decision Trace**（JSON / HTML）。无需参考图或 diff，用 ReAct Agent（本地 LLM + VLM，经 Ollama 推理）在 CV 规则检测器之上自主决策「是否做视觉二次确认 / 是否补检 / 是否主动发现」，给出可回溯的决策轨迹。全程不依赖云端 API。

---

## 特性

- **九个专用 CV 检测器**：压缩、模糊、马赛克、色带、绿幕溢色、发丝、面部、背景、手部几何（`hand_anomaly` 实验性 MVP）
- **可解释归因**：`degradations[]` 给出 RLE 像素 mask、bbox、各检测器 evidence、root_cause、VLM 语义确认（**与 MOS 计算解耦**）
- **ReAct Agent 自主决策**：LLM 观察 CV 结果后自主判断是否调用 VLM、是否补检——而非硬编码阈值触发
- **VLM 主动发现（`vlm_discover`）**：Agent 可让 VLM 对全帧主动扫描，发现 CV 规则检不到的语义异常（AI 生成多指、面部生成错误等），结果写入 `agent_meta.vlm_discover_findings`
- **视频 clip 多帧输入（V2）**：`VideoClipRunner` 外层包装多帧，逐帧跑单帧 pipeline + 帧间 `TemporalFlicker`（亮度/色相跳变）聚合，不改单帧接口
- **MOS 打分**：默认 rule 启发式（零依赖）；可选 `mos_model=clip_iqa` 用 CLIP-IQA 无参考感知预测。MOS 只是帧级一个总分，per-item 罚分明细非感知归因
- **优雅降级**：Ollama 不可用时，Agent 步骤自动走规则降级并在 trace 中记录原因
- **两种流水线**：v0.1 确定性快速路径（`--legacy-fixed`）；V1 ReAct Agent（默认）

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
                    │    ├─ vlm_discover  (VLM主动发现) │
                    │    ├─ rerun_detector(重检)    │
                    │    ├─ dispatch_compression    │
                    │    └─ accept       (终止)     │
                    └──────────────────────────────┘
```

### v0.1 基线架构（`--legacy-fixed`）

```
GlobalScan → 9 detectors → Report
```

无 Agent / VLM，全量检测器确定性运行。

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
| `face_artifact` | 面部过曝 / 模糊 | 亮度 + Laplacian | 主力 |
| `hand_anomaly` | 手部几何异常 | 启发式 ROI + 边缘密度；**可选** mediapipe 0.10+ Tasks API 关键点几何 | **实验性 MVP** |

> **`hand_anomaly` 说明**：CV 路径的多指计数 / 粘连尚未实现；但 **`vlm_discover` 工具可由 VLM 主动发现手部语义异常**（如多指），结果写入 `agent_meta.vlm_discover_findings`。

---

## 安装

```bash
git clone https://github.com/Chamuel08/LocalQualityDegradationDetection.git
cd LocalQualityDegradationDetection

python3 -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -U pip setuptools wheel
pip install -r requirements-dev.txt                    # 依赖 + 可编辑安装 + pytest
cp config.example.yaml config.yaml                     # 本地配置（已 gitignore）

# 可选增强（均非必需）
pip install -r requirements-optional.txt               # mediapipe(手部) + pyiqa/torch(CLIP-IQA MOS)
# 或只启用 CLIP-IQA MOS：
pip install "lqdd[clip_iqa]"                            # 首次运行自动下载权重 ~260MB
```

**V1 Agent 需要 Ollama**（v0.1 不需要）：

```bash
# 安装：https://ollama.com/download  →  启动：ollama serve
ollama pull qwen2.5vl:7b      # VLM 视觉确认（~4.7 GB）
ollama pull qwen2.5:1.5b      # Agent 决策 LLM（~1 GB）
ollama list                   # 确认两个模型已就位
```

> 代理注意：若 `HTTP_PROXY` 指向不可用代理，可能影响访问 localhost。本项目 Ollama 客户端已设 `trust_env=False`；仍异常可 `unset HTTP_PROXY HTTPS_PROXY`。

验证：

```bash
python -c "import cv2, numpy, yaml, httpx; print('ok')"
python detect.py --image docs/demo/case_blur.png --legacy-fixed --output /tmp/r.json && echo "v0.1 ok"
python detect.py --image docs/demo/case_blur.png --config config.yaml --output /tmp/r_v1.json && echo "V1 ok"
```

---

## 使用

### CLI — 单帧检测

```bash
source .venv/bin/activate

# V1 ReAct Agent（默认，需 Ollama）— 输出 HTML 可视化
python detect.py --image docs/demo/case_blur.png --config config.yaml --output report.html

# 输出 JSON
python detect.py --image docs/demo/case_blur.png --config config.yaml --output report.json

# v0.1 基线（无 Agent / VLM，无需 Ollama）
python detect.py --image docs/demo/case_blur.png --legacy-fixed --output report_v01.json
```

### CLI — 批量检测

```bash
python detect.py --image-dir /path/to/frames/ --legacy-fixed --output-dir reports/
```

### CLI 参数

| 参数 | 说明 |
|------|------|
| `--image PATH` | 单张输入图像 |
| `--image-dir PATH` | 图像目录（非递归） |
| `--config PATH` | YAML 配置（默认读 `config.yaml`） |
| `--legacy-fixed` | v0.1 固定流水线，跳过 Agent 层 |
| `--output PATH` | 输出文件（`-` = stdout JSON；`.html` = HTML 可视化） |
| `--output-dir PATH` | 批量输出目录 |
| `--metadata PATH` | 可选 metadata JSON 侧车文件 |
| `--ignore-regions PATH` | 可选 ignore-regions JSON 侧车文件 |
| `--frame-id ID` | 覆盖报告中的 `frame_id` |
| `--verbose` | 详细日志 |

安装后也可直接用 `lqdd` 入口（等价于 `python detect.py`）。

### Python API — V1 ReAct Agent（推荐）

```python
import cv2
from lqdd.config.loader import load_config
from lqdd.models.inputs import SingleFrameInput
from lqdd.agent.orchestrator import AgentOrchestrator

config = load_config()                              # 读取 config.yaml
frame = cv2.imread("path/to/frame.png")
fi = SingleFrameInput(frame=frame, frame_id="my_frame", mode="fast")

orc = AgentOrchestrator(config)                     # 需 Ollama
report = orc.run(fi)

print("MOS:", report.overall_mos)
print("degradations:", len(report.degradations))

# Agent 自主决策轨迹（agent_meta 在 Agent 禁用/不可用时为 None）
if report.agent_meta:
    for step in report.agent_meta["agent_steps"]:
        print(f"Step {step['step']}: [{step['action']}] {step['thought']}")
        print(f"  → {step['observation']}")
```

### Python API — 无需 Ollama 的两种降级

```python
from lqdd.agent.orchestrator import AgentOrchestrator
from lqdd.agent.judge_client import MockJudgeClient, RuleBasedJudgeClient

# MockJudgeClient：模拟 LLM 决策（测试用）
orc = AgentOrchestrator(config, judge_client=MockJudgeClient())

# RuleBasedJudgeClient：基于规则模拟 Agent 决策（Ollama 不可用时降级）
orc = AgentOrchestrator(config, judge_client=RuleBasedJudgeClient(config.agent))
report = orc.run(fi)
```

### Python API — v0.1 基线 FastPipeline

```python
from lqdd.pipeline.fast_pipeline import FastPipeline

pipeline = FastPipeline(config)
report = pipeline.run(fi)
```

### Python API — V2 视频 clip 多帧输入

`VideoClipRunner` 是单帧 pipeline 的外层包装器：对每帧独立调用 `pipeline.run()`，再在帧间聚合层运行 `TemporalFlicker`（亮度/色相跳变检测），返回 `VideoClipReport`（逐帧报告 + 闪烁摘要 + 聚合 MOS）。**不修改任何单帧内部接口**。

```python
import cv2
from lqdd.config.loader import load_config
from lqdd.pipeline.agent_pipeline import AgentPipeline
from lqdd.pipeline.video_clip_runner import VideoClipRunner, sample_frames_from_video

config = load_config()
pipeline = AgentPipeline(config)                   # 也可用 FastPipeline（无需 Ollama）
runner = VideoClipRunner(pipeline)

# 从视频均匀抽帧（最多 8 帧）
frames = sample_frames_from_video("input.mp4", max_frames=8)
result = runner.run(frames, clip_id="clip_001")

print("aggregate_mos:", result.aggregate_mos)
print("worst_frame:", result.worst_frame_mos, "@ idx", result.worst_frame_index)
print("flicker:", result.flicker_result.is_flickering, "ratio:", result.flicker_result.flicker_ratio)
print("degradation_summary:", result.degradation_summary)   # 各 degradation_type 跨帧出现次数
```

> `TemporalFlicker` 依赖多帧输入，不进入单帧 `ALL_DETECTOR_NAMES`，仅在 `VideoClipRunner` 层调用。`vlm_discover` 的主动发现结果在每帧报告的 `agent_meta.vlm_discover_findings` 中。

### Python API — 自定义 LLM Agent

实现 `JudgeClient` 接口即可替换 Agent 决策后端（如换 OpenAI 兼容模型）：

```python
from lqdd.agent.judge_client import JudgeClient
from lqdd.agent.orchestrator import AgentOrchestrator
from typing import Any

class MyCustomAgent(JudgeClient):
    def review(self, prompt: str) -> dict[str, Any] | None:
        return {"assessment": "consistent", "actions": [], "needs_round2": False}

    def decide(self, system_prompt: str, user_prompt: str) -> dict[str, Any] | None:
        # ReAct Agent 核心决策接口
        # 返回格式：{"thought": "...", "action": "vlm_analyze|vlm_discover|accept|...", "reason": "..."}
        ...

orc = AgentOrchestrator(config, judge_client=MyCustomAgent())
```

### 图形界面（`lqdd-gui`）

基于 Gradio 的 web GUI + pywebview 原生窗口，复用 pipeline 与 mask 叠加渲染，覆盖单帧检测与 V2 视频多帧输入。

```bash
pip install "lqdd[gui]"          # gradio + pywebview（可选依赖）

lqdd-gui                         # pywebview 原生窗口（默认）
lqdd-gui --browser               # 浏览器回退（http://127.0.0.1:7860）
lqdd-gui --port 8000             # 自定义端口
```

界面布局：
- **左侧输入区**：单帧图片上传 / 视频上传（Tab 切换）+ 模式 Radio（V1 Agent / v0.1 基线）+ 配置路径
- **右侧结果区**：mask 叠加预览图 + MOS/严重度总览 + 劣化列表表 + Agent 决策轨迹表 + `vlm_discover` 主动发现 + 视频 flicker 聚合 + 完整 JSON（可折叠）

> V1 模式需 Ollama 已运行；不可用时 Agent 自动规则降级，GUI 不阻塞。

### 打包可执行文件（`build/`）

把 GUI 打包成独立可执行文件（PyInstaller），运行时启动本地 server 并打开 pywebview 原生窗口。

```bash
bash build/build.sh              # 产出 dist/lqdd-gui/lqdd-gui
./dist/lqdd-gui/lqdd-gui         # 启动原生窗口
```

打包说明：
- `build/build.sh`：venv-pack → pip install → pyinstaller → `dist/lqdd-gui/`
- `build/app.spec`：`collect_all` 收集 gradio/pywebview/cv2；**排除** torch/pyiqa/mediapipe（重依赖，体积太大）
- **不含 Ollama / 模型权重**：需另装 Ollama 并 `ollama pull qwen2.5vl:7b` / `qwen2.5:1.5b`
- **降级**：打包后 MOS 用 `rule` 后端，`hand_anomaly` 用边缘密度 fallback

---

## 结果展示

所有素材均为本项目独立生成（AI 合成人像 + 程序化劣化），不依赖任何第三方数据集。

### 源图与劣化样本

干净源图（AI 生成）及其 7 种程序化劣化版本，存放在 `docs/demo/`：

![干净合成人像](docs/demo/synthetic_portrait.png)

### 多 case 检测结果一览（v0.1）

对同一张干净源图分别施加 7 种劣化，用 v0.1 `--legacy-fixed` 检测：

| 劣化样本 | 施加的劣化 | 目标检测器 | 是否检出 | 置信度 | 严重度 |
|----------|-----------|-----------|---------|--------|--------|
| `case_compression.png` | 边缘高频压缩 | `compression_artifact` | ✅ | 0.867 | moderate |
| `case_block.png` | JPEG 块效应 | `compression_artifacts` | ✅ | 0.920 | severe |
| `case_blur.png` | 高斯模糊 | `blur_artifact` | ✅ | 0.740 | moderate |
| `case_mosaic.png` | 下采样像素化 | `mosaic_artifact` | ✅ | 0.759 | moderate |
| `case_overexposure.png` | 亮度过曝 | `face_artifact` | ✅ | 0.720 | minor |
| `case_banding.png` | 色深降低 | `banding_artifact` | ❌ | — | — |
| `case_green_spill.png` | 边缘绿溢色 | `edge_bleed` | ✅ | 0.797 | critical |

> `banding` 检测器当前阈值偏保守，对合成色带检出率较低（已知限制）。其余 6 种劣化均被对应检测器成功检出。

### V1 ReAct Agent 完整示例（`case_blur`）

该样本中 `face_artifact` 置信度 0.68，**LLM Agent 自主决策触发了 VLM 确认**。

**1. 报告总览**

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
        "observation": "vlm_analyze 完成：确认了 3 个检测项"
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

**2. VLM 确认的 degradation — 三层可解释结构**

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
  "root_cause_hypothesis": { "cause": "generation_artifact", "confidence": 0.45 },
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

**3. decision_trace — ReAct Agent 编排全链路**

```text
mode_select → global_scan → routing → detection
  → agent_step(vlm_analyze)  ← LLM 自主决策：调用 VLM 确认
  → agent_step(accept)       ← LLM 自主决策：结果可信，终止
  → aggregation
```

**4. mos_breakdown — MOS 总分**

**MOS 与归因是两件事**：归因看 `degradations[]`（detector、bbox、evidence、root_cause、vlm_reasoning），与 MOS 无关；MOS 只是帧级一个总分。推荐 `mos_model=clip_iqa`，由 CLIP-IQA 无参考感知预测直接给出：

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

### V1 vs v0.1 对比

同一张图分别跑两条路径：

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

### V2 视频结果（`VideoClipRunner`）

```python
result = runner.run(frames, clip_id="clip_001")
# VideoClipReport:
#   clip_id            = "clip_001"
#   frame_count        = 5
#   aggregate_mos      = 3.82            # 逐帧 MOS 均值
#   worst_frame_mos    = 2.91
#   worst_frame_index  = 3               # 最差帧
#   flicker_result:
#     is_flickering    = True
#     flicker_ratio    = 0.25
#     flicker_segments = [FlickerSegment(start=2, end=3, metric='luma_delta', max_delta=18.4, severity='moderate')]
#   degradation_summary = {"face_blur": 3, "compression_artifact": 1, "edge_bleed": 2}
```

`TemporalFlicker` 检出帧 2→3 间存在亮度跳变（18.4 > 阈值 8.0），标记为 `moderate`；`degradation_summary` 跨帧汇总各劣化类型出现次数。

### `vlm_discover` 主动发现示例

当 MOS 低且 CV 检出为 0 时，Agent 可自主选择 `vlm_discover` 让 VLM 对全帧主动扫描：

```json
{
  "agent_meta": {
    "agent_steps": [
      { "step": 1, "action": "vlm_discover", "thought": "MOS=2.1 且 detection_count=0，疑似 AI 生成语义异常" }
    ],
    "vlm_discover_findings": [
      {
        "degradation_type": "hand_extra_finger",
        "region_description": "左下角人物左手出现6根手指",
        "severity": "moderate",
        "confidence": 0.82,
        "reasoning": "清晰可见手部有多余手指",
        "mos_impact_estimate": -0.4
      }
    ]
  }
}
```

> `vlm_discover_findings` **不**合并进 `degradations[]`（VLM 无像素级定位），仅写入 `agent_meta`；`mos_impact_estimate` 仅参考，不参与 MOS 计算。

### 复现命令

```bash
source .venv/bin/activate

# v0.1 基线（对任一 case，无需 Ollama）
python detect.py --image docs/demo/case_blur.png --legacy-fixed --output report_v01.json

# V1 ReAct Agent（需 Ollama）
python detect.py --image docs/demo/case_blur.png --config config.yaml --output report_v1.json

# HTML 可视化报告
python detect.py --image docs/demo/case_blur.png --config config.yaml --output docs/demo/case_blur_v1.html

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

---

## 配置

将 `config.example.yaml` 复制为 `config.yaml`（已 gitignore），按需调整检测器阈值、Agent 参数及 VLM 参数。

### V1 ReAct Agent / VLM 关键配置

```yaml
agent:
  enabled: true
  max_rounds: 2                    # Agent 最大循环步数 = max_rounds × 3
  high_confidence_threshold: 0.7   # 供规则降级参考（LLM 模式下由 LLM 自主判断）
  grey_zone_lower: 0.4
  grey_zone_upper: 0.7
  max_detectors_per_frame: 9
  hard_decision_threshold: 0.55

vlm:
  provider: ollama
  model: qwen2.5vl:7b               # VLM 视觉确认 / 主动发现
  host: http://localhost:11434
  timeout_ms: 120000
  max_calls_per_frame: 3           # Agent 单帧最多触发 VLM 次数

judge:
  provider: ollama
  model: qwen2.5:1.5b               # Agent 决策 LLM
  host: http://localhost:11434
  timeout_ms: 45000

report:
  mos_model: "rule"                # rule（默认零依赖）/ clip_iqa（CLIP-IQA 感知预测）/ internal（预留）
```

### 环境变量覆盖

| 变量 | 作用 |
|------|------|
| `OLLAMA_HOST` | 覆盖 VLM / Agent LLM 的 Ollama 地址 |
| `LQDD_VLM_MODEL` | 覆盖 VLM 模型名 |
| `LQDD_JUDGE_MODEL` | 覆盖 Agent LLM 模型名 |
| `LQDD_AGENT_ENABLED=0` | 关闭 Agent，等效 v0.1 |

---

## 项目结构

```
├── detect.py                 # CLI 入口
├── config.example.yaml       # 配置模板
├── requirements.txt          # 运行时依赖
├── requirements-dev.txt      # 开发依赖（含 -e .）
├── requirements-optional.txt # mediapipe + pyiqa/torch（可选）
├── src/lqdd/
│   ├── detectors/            # 九个劣化检测器
│   ├── agent/
│   │   ├── orchestrator.py   # ReAct Agent 编排核心
│   │   ├── judge_client.py   # LLM Agent 客户端（Ollama / Mock / RuleBased）
│   │   ├── prompts.py        # AGENT_SYSTEM_PROMPT + AGENT_OBSERVE_TEMPLATE + VLM_DISCOVER_PROMPT
│   │   ├── router.py         # CV 检测器派发路由（不负责 VLM 决策）
│   │   └── context.py        # AgentContext 工厂
│   ├── vlm/
│   │   ├── client.py         # Ollama VLM 客户端
│   │   ├── confirm.py        # VLM 确认执行（供 Agent 调用）
│   │   └── fuse.py           # CV + VLM 置信度融合
│   ├── models/
│   │   ├── agent.py          # AgentStep / AgentAction / AgentMeta / VLMDiscoverFinding
│   │   └── report.py         # QualityReport / DegradationItem 等
│   ├── mos/                  # MOS 预测后端（按 mos_model 分发）
│   │   └── clip_iqa.py       # 可选：CLIP-IQA 无参考画质预测
│   ├── temporal_flicker/     # V2：时域闪烁检测器（帧间聚合层）
│   │   └── detector.py       # detect_temporal_flicker：相邻帧亮度/色相跳变
│   ├── pipeline/
│   │   ├── fast_pipeline.py     # v0.1
│   │   ├── agent_pipeline.py    # V1
│   │   └── video_clip_runner.py # V2：多帧输入包装器 + TemporalFlicker 聚合
│   ├── ui/                     # 图形界面（可选，gradio + pywebview）
│   │   └── app.py              # lqdd-gui 入口：单帧/视频 GUI + 原生窗口
│   └── report/               # JSON / HTML 报告构建（含 compute_mos）
├── build/                     # PyInstaller 打包体系（可执行文件）
│   ├── build.sh               # 一键构建脚本
│   ├── app.spec               # PyInstaller spec
│   └── requirements_pack.txt  # 打包专用依赖
├── docs/demo/                # 演示素材（合成人像 + 7 种劣化）
├── benchmark/                # 批量评测脚本
├── scripts/                  # benchmark 生成、Demo 资源
├── tests/                     # 单元 / 集成 / 契约测试
└── specs/                     # 设计文档与契约（spec-kit 驱动开发）
    ├── 001-v0-fast-mvp/       # v0.1 契约与 schema
    ├── 002-v1-agent-layer/    # V1 ReAct Agent 契约
    └── 003-v2-video-and-discover/  # V2 视频 + vlm_discover 契约
```

---

## 项目状态

### 已完成

| 模块 | 状态 | 说明 |
|------|------|------|
| v0.1 固定流水线 | ✅ | GlobalScan → 9 检测器 → Report（`--legacy-fixed`） |
| V1 ReAct Agent 编排 | ✅ | LLM 自主决策循环（Observe→Think→Act），工具：vlm_analyze / vlm_discover / rerun_detector / dispatch_compression / accept |
| V2 视频与主动发现 | ✅ | `VideoClipRunner` 多帧包装 + `TemporalFlicker` 帧间聚合；`vlm_discover` VLM 全帧主动发现 |
| CLI + JSON/HTML 报告 | ✅ | 单图 / 批量 / schema 校验 |
| 6/8 检测器可用 | ✅ | edge_bleed / compression / blur / mosaic / overexposure / green_spill |
| Ollama VLM + Agent LLM | ✅ | `qwen2.5vl:7b` + `qwen2.5:1.5b`，规则降级 |
| Agent 决策轨迹记录 | ✅ | `agent_meta.agent_steps`：完整 thought + action + observation；`vlm_discover_findings` 主动发现 |

### 已知限制

| 限制 | 说明 | 影响 |
|------|------|------|
| `banding_artifact` 检出率低 | 阈值偏保守，对合成色带不敏感 | Demo 中 banding case 未检出 |
| `hand_anomaly` 实验性 MVP | 启发式 ROI + 边缘密度 fallback；装 mediapipe 走 HandLandmarker 关键点几何；CV 路径多指/粘连未实现 | 已标注为实验性；多指等语义异常可由 `vlm_discover` 经 VLM 主动发现 |
| 干净帧误报率偏高 | `compression` / `blur` / `mosaic` 在干净帧上有交叉敏感 | Benchmark clean FPR 未达标 |
| Deep Mode 未实现 | `--mode deep` 返回 exit 2 | V1 仅 Fast Mode 可用 |
| Agent 小模型推理质量 | `qwen2.5:1.5b` 参数量小，复杂场景决策质量有限 | 可换更大模型（如 `qwen2.5:7b`）提升 Agent 推理质量 |

### V2 状态

| 功能 | 状态 | 说明 |
|------|------|------|
| `vlm_discover` Agent 工具 | ✅ **已实现** | Agent 可对全帧主动扫描，发现 AI 生成伪影等 CV 盲区；结果写入 `agent_meta.vlm_discover_findings` |
| `VideoClipRunner` 包装器 | ✅ **已实现** | 外层包装多帧输入，不改单帧接口；支持 `sample_frames_from_video()` 抽帧 |
| `TemporalFlicker` 检测器 | ✅ **已实现** | 帧间亮度/色相跳变检测，在 `VideoClipRunner` 层调用，不进单帧流水线 |
| Deep Mode | ⏸ 推迟 | `vlm_discover` 已满足核心需求；Deep Mode 重构整个调度层代价过高 |
| PatchCore 异常检测 | ❌ 不实现 | 违反系统「无参考」核心原则，需维护正常样本 memory bank |

详见 [`specs/VERSION_ROADMAP.md`](specs/VERSION_ROADMAP.md)。

---

## 评测（可选，本地数据）

本仓库**不包含**评测图像或 GT mask。如需可复现指标，可在本地生成合成 benchmark：

```bash
source .venv/bin/activate
export LQDD_DATA_DIR=/path/to/data

python scripts/generate_benchmark_dataset.py \
  --input  $LQDD_DATA_DIR/source_frames \
  --output $LQDD_DATA_DIR/synthetic_benchmark \
  --samples-per-type 5 --clean-count 8

python benchmark/run_eval.py \
  --manifest $LQDD_DATA_DIR/synthetic_benchmark/manifest.json \
  --output benchmark/runs/results.json
```

> `benchmark/run_eval.py` 当前走 **v0.1 `--legacy-fixed` 路径**，用于可复现的检测器基线评测。评测 V1 ReAct Agent 效果请用不带 `--legacy-fixed` 的 `detect.py`，对比 JSON 中的 `decision_trace` / `agent_meta.agent_steps`。

---

## 开发

```bash
source .venv/bin/activate
pytest tests/ -m "not vlm" -q       # 依赖 VLM 服务的测试标记为 @pytest.mark.vlm，默认跳过
```

---

## 文档

| 文档 | 说明 |
|------|------|
| [`specs/USE_CASE_BADCASE.md`](specs/USE_CASE_BADCASE.md) | Badcase 场景与边界 |
| [`specs/VERSION_ROADMAP.md`](specs/VERSION_ROADMAP.md) | v0.1 / V1 / V2 路线 |
| [`specs/001-v0-fast-mvp/`](specs/001-v0-fast-mvp/) | v0.1 契约与 schema |
| [`specs/002-v1-agent-layer/`](specs/002-v1-agent-layer/) | V1 ReAct Agent / VLM 契约 |
| [`specs/003-v2-video-and-discover/`](specs/003-v2-video-and-discover/) | V2 视频 + `vlm_discover` 契约 |

---

## License

[Apache License 2.0](LICENSE)
